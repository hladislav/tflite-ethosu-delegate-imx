/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Copyright 2022-2023 NXP
Copyright 2025 Ladislav Hano

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/

#include <utility>
#include <string.h>
#include <vector>

#include <stdio.h>

#include "flatbuffers/flexbuffers.h"

#include "ethosu_delegate.h"
#include "simple_delegate.h"
#include "ethosu_delegate_utils.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/subgraph.h"

#include "ethosu.h"

using namespace std;

namespace tflite {
namespace ethosu {

struct TfLiteEthosuContext{
  EthosU::Device* device;
  shared_ptr<EthosU::Buffer> arena_buffer;  //Input buffer for input/ouput/scratch tensor
  shared_ptr<EthosU::Buffer> flash_buffer;  //Input buffer for weight tensor
  shared_ptr<EthosU::Buffer> qread_buffer;  //Ouput buffer for profiling qread data
  size_t arena_size;
  size_t flash_size;
};

// Ethosu delegate kernel.
class EthosuDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit EthosuDelegateKernel(const EthosuDelegateOptions& opt)
      : options(opt) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params,
                    void* data) override {
    const char* meta_ptr = nullptr;
    ethosu_context = reinterpret_cast<TfLiteEthosuContext*>(data);

    //Check if vela compiled model
    int vela_node = 0;
    for (int i = 0; i < params->nodes_to_replace->size; i ++) {
        TfLiteNode* node;
        TfLiteRegistration* reg;
        auto node_idx = params->nodes_to_replace->data[i];
        context->GetNodeAndRegistration(context, node_idx, &node, &reg);
        if (reg->builtin_code == kTfLiteBuiltinCustom &&
            strcmp(reg->custom_name, ETHOSU_CUSTOM_NAME) == 0) {
            vela_node ++;
        }
    }
    if (vela_node == 0) {
        //Online model compile
        try {
	    string cache_file = options.cache_file_path + "/" + to_string(params->nodes_to_replace->data[0]);
            if (options.cache_file_path != "" &&
                access(cache_file.c_str(), F_OK) == 0) {
                //Cache file exist, read model from cache file.
                TFLITE_LOG(TFLITE_LOG_INFO, "Ethosu delegate: Read model from cache file %s",
                                   options.cache_file_path.c_str());
                model = readTFLiteModel(cache_file);
            } else {
                auto model_converter = ModelConverter::GetSingleton();
                model = model_converter->convert(context, params);
                if (options.cache_file_path != "") {
                    // Write to cache file
                    TFLITE_LOG(TFLITE_LOG_INFO, "Ethosu delegate: Write model to cache file %s",
                                       options.cache_file_path.c_str());
                    writeTFLiteModel(model.get(), cache_file);
                }
            }
        } catch (const char* msg) {
            TF_LITE_KERNEL_LOG(context, "Failed to build model, %s\n", msg);
            return kTfLiteDelegateError;
        }
        TF_LITE_ENSURE_EQ(context, model->subgraphs.size(), 1);
        TF_LITE_ENSURE_EQ(context, model->subgraphs[0]->operators.size(), 1);
        operations.resize(model->subgraphs[0]->operators.size());

        // Get the address offsets for each tensor from metadata
        for (auto &metadata : model->metadata) {
            if (metadata->name == OFFLINE_MEM_ALLOC_METADATA) {
                auto &metadata_buffer = model->buffers[metadata->buffer];
                meta_ptr = (const char*)metadata_buffer->data.data();
                break;
            }
        }
        TF_LITE_ENSURE(context, meta_ptr != nullptr);
        address_offsets = METADATA_TO_OFFSET(meta_ptr);

        size_t arena_size = 0, flash_size = 0;
        for (int i = 0; i < METADATA_SIZE(meta_ptr); i ++){
            auto tensor_size = GetTensorDataSize(model->subgraphs[0]->tensors[i]);
            if (address_offsets[i] >= 0 && address_offsets[i] + tensor_size > arena_size)
                arena_size = address_offsets[i] + tensor_size;
        }
        arena_offset = ethosu_context->arena_size;
        ethosu_context->arena_size = arena_offset + ALIGN_SIZE(arena_size);

        auto flash_idx = model->subgraphs[0]->operators[0]->inputs[FLASH_TENSOR_INDEX];
        auto &flash_tensor = model->subgraphs[0]->tensors[flash_idx];
        flash_size = model->buffers[flash_tensor->buffer]->data.size();
        flash_offset = ethosu_context->flash_size;
        ethosu_context->flash_size = flash_offset + ALIGN_SIZE(flash_size);
    } else if (vela_node == params->nodes_to_replace->size) {
        //Offline model compile
        operations.resize(params->nodes_to_replace->size);
        for (int i = 0; i < params->nodes_to_replace->size; i++) {
            TfLiteNode* node;
            TfLiteRegistration* reg;
            int node_idx = params->nodes_to_replace->data[i];
            context->GetNodeAndRegistration(context, node_idx, &node, &reg);
            tflite::TfLiteIntArrayView inputs(node->inputs);
            tflite::TfLiteIntArrayView outputs(node->outputs);

            auto& op = operations[i];
            copy(inputs.begin(), inputs.end(), back_inserter(op.inputs));
            copy(outputs.begin(), outputs.end(), back_inserter(op.outputs));
        }

        //Get arena offset for each tensor from meta data
        size_t bytes;
        TF_LITE_ENSURE_OK(context, context->GetModelMetadata(context,
                          OFFLINE_MEM_ALLOC_METADATA, &meta_ptr, &bytes));
        address_offsets = METADATA_TO_OFFSET(meta_ptr);

        size_t arena_size = 0;
        for (int i = 0; i < METADATA_SIZE(meta_ptr); i ++){
            auto tensor = &context->tensors[i];
            if (address_offsets[i] >= 0 && address_offsets[i] + tensor->bytes > arena_size)
                arena_size = address_offsets[i] + tensor->bytes;
        }
        ethosu_context->arena_size = ALIGN_SIZE(arena_size);
        flash_offset = 0;
        arena_offset = 0;
    } else {
        TF_LITE_KERNEL_LOG(context, "Unsupported vela compiled model.\n");
        return kTfLiteDelegateError;
    }

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i ++) {
        if (options.pmu_counter_config[i] != 0)
            pmu_counter_config.push_back(options.pmu_counter_config[i]);
    }

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    if(model == nullptr) {
        //Offline model compile
        TF_LITE_ENSURE_OK(context, PrepareOfflineCompiledModel(context, node));
    } else {
        //Online model compile
        TF_LITE_ENSURE_OK(context, PrepareOnlineCompiledModel(context, node));
    }

    for (auto& op : operations) {
        vector<shared_ptr<EthosU::Buffer>> ifm {ethosu_context->arena_buffer};
        vector<shared_ptr<EthosU::Buffer>> ofm {};
        if (options.enable_profiling) {
            ofm.push_back(ethosu_context->qread_buffer);
            ethosu_context->qread_buffer->resize(0);
        }
        if (op.need_flash)
            ifm.push_back(ethosu_context->flash_buffer);

        op.memory_layout.flash_offset = flash_offset;
        op.memory_layout.arena_offset = arena_offset;
        op.inference = make_shared<EthosU::Inference>(op.ethosu_network, ifm.begin(), ifm.end(),
		ofm.begin(), ofm.end(), pmu_counter_config, options.enable_cycle_counter, op.memory_layout);
    }

    return kTfLiteOk;
  }

  TfLiteStatus PrepareOfflineCompiledModel(TfLiteContext* context, TfLiteNode* node) {
    try {
        if (ethosu_context->arena_buffer == nullptr) {
            ethosu_context->arena_buffer = make_shared<EthosU::Buffer>(*ethosu_context->device,
			    ethosu_context->arena_size);
        }
        Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
        for (auto& op : operations) {
            op.memory_layout.input_count = op.inputs.size() - 4;
            op.memory_layout.output_count = op.outputs.size();

            // Get command stream data size and create buffer
            auto cms_idx = op.inputs[CMS_TENSOR_INDEX];
            auto cms_tensor = &context->tensors[cms_idx];
            size_t cms_data_size = cms_tensor->bytes;

            op.ethosu_net_buffer = make_shared<EthosU::Buffer>(*ethosu_context->device, cms_data_size);
            op.ethosu_net_buffer->resize(cms_data_size);
            memcpy(op.ethosu_net_buffer->data(), cms_tensor->data.raw, cms_data_size);
            op.ethosu_network = make_shared<EthosU::Network>(*ethosu_context->device, op.ethosu_net_buffer);

            // Get flash tensor data size
            auto flash_idx = op.inputs[FLASH_TENSOR_INDEX];
            auto flash_tensor = &context->tensors[flash_idx];
            size_t flash_data_size = flash_tensor->bytes;
            op.need_flash = (flash_data_size != 0);
            if (flash_data_size != 0 && ethosu_context->flash_buffer == nullptr) {
                ethosu_context->flash_buffer =
			make_shared<EthosU::Buffer>(*ethosu_context->device, flash_data_size);
                memcpy(ethosu_context->flash_buffer->data(), flash_tensor->data.raw, flash_data_size);
            }

            // Get addresses of outputs data
            for (int i = 0; i < op.outputs.size(); ++i) {
                auto idx = op.outputs[i];
                auto tensor = &context->tensors[idx];
                op.memory_layout.output_offset[i] = address_offsets[idx];
                op.memory_layout.output_size[i] = tensor->bytes;

                TfLiteCustomAllocation allocation({ethosu_context->arena_buffer->data() + address_offsets[idx],
                                tensor->bytes});
                this_subgraph->SetCustomAllocationForTensor(idx, allocation,
                                kTfLiteCustomAllocationFlagsSkipAlignCheck);
            }
            // Get addresses to inputs data
            for (int i = SCRATCH_TENSOR_INDEX; i < op.inputs.size(); ++i) {
                auto idx = op.inputs[i];
                auto tensor = &context->tensors[idx];
                if (i >= INPUT_TENSOR_INDEX) {
                    op.memory_layout.input_offset[i - INPUT_TENSOR_INDEX] = address_offsets[idx];
                    op.memory_layout.input_size[i - INPUT_TENSOR_INDEX] = tensor->bytes;
                }

                TfLiteCustomAllocation allocation({ethosu_context->arena_buffer->data() + address_offsets[idx],
                                tensor->bytes});
                this_subgraph->SetCustomAllocationForTensor(idx, allocation,
                                kTfLiteCustomAllocationFlagsSkipAlignCheck);
            }
        }
    } catch (exception &e) {
        TF_LITE_KERNEL_LOG(context, "Failed to alloc ethos_u buffer.\n");
        return kTfLiteDelegateError;
    }

    return kTfLiteOk;
  }

  TfLiteStatus PrepareOnlineCompiledModel(TfLiteContext* context, TfLiteNode* node){
    // Map the ethosu tensor index to tflite tensor index.
    auto FindTfliteTensorIndex = [&](std::string name) -> int32_t {
      for (int i = 0; i < context->tensors_size; i ++){
         if (context->tensors[i].name && context->tensors[i].name == name)
           return i;
      }
      return -1;
    };

    // Preare the buffers for ethosu device
    try {
      Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
      if (ethosu_context->arena_buffer == nullptr) {
          ethosu_context->arena_buffer = make_shared<EthosU::Buffer>(*ethosu_context->device,
			 ethosu_context->arena_size);
      }
      if (ethosu_context->flash_buffer == nullptr && ethosu_context->flash_size != 0) {
          ethosu_context->flash_buffer = make_shared<EthosU::Buffer>(*ethosu_context->device,
			 ethosu_context->flash_size);
      }

      size_t arena_data_size = 0;
      auto &ethosu_tensors = model->subgraphs[0]->tensors;
      for (int i = 0; i < model->subgraphs[0]->operators.size(); i ++){
        auto &op = operations[i];
        auto &ethosu_op = model->subgraphs[0]->operators[i];
        op.memory_layout.input_count = ethosu_op->inputs.size() - 4;
        op.memory_layout.output_count = ethosu_op->outputs.size();

        // Get command stream data size and create buffer
        auto cms_idx = ethosu_op->inputs[CMS_TENSOR_INDEX];
        auto &cms_tensor = ethosu_tensors[cms_idx];
        auto &cms_buffer = model->buffers[cms_tensor->buffer];
        size_t cms_data_size = cms_buffer->data.size();
        op.ethosu_net_buffer = make_shared<EthosU::Buffer>(*ethosu_context->device, cms_data_size);
        memcpy(op.ethosu_net_buffer->data(), cms_buffer->data.data(), cms_data_size);
        op.ethosu_network = make_shared<EthosU::Network>(*ethosu_context->device, op.ethosu_net_buffer);

        // Get flash tensor data size
        auto flash_idx = ethosu_op->inputs[FLASH_TENSOR_INDEX];
        auto &flash_tensor = ethosu_tensors[flash_idx];
        auto &flash_buffer = model->buffers[flash_tensor->buffer];
        auto flash_data_size = flash_buffer->data.size();
        op.need_flash = (flash_data_size != 0);
        if (flash_data_size != 0) {
            memcpy(ethosu_context->flash_buffer->data() + flash_offset, flash_buffer->data.data(), flash_data_size);
        }

        // Get addresses of outputs data
        for (int i = 0; i < ethosu_op->outputs.size(); ++i) {
            auto ethosu_idx = ethosu_op->outputs[i];
            auto &tensor = model->subgraphs[0]->tensors[ethosu_idx];
            op.memory_layout.output_offset[i] = address_offsets[ethosu_idx] + arena_offset;
            op.memory_layout.output_size[i] = GetTensorDataSize(tensor);

            auto tflite_idx = FindTfliteTensorIndex(tensor->name);
            TF_LITE_ENSURE(context, tflite_idx != -1);
            TfLiteCustomAllocation allocation({ethosu_context->arena_buffer->data() + op.memory_layout.output_offset[i],
			    op.memory_layout.output_size[i]});
            this_subgraph->SetCustomAllocationForTensor(tflite_idx, allocation,
			    kTfLiteCustomAllocationFlagsSkipAlignCheck);
        }
        // Get addresses to inputs data
        for (int i = 0; i < ethosu_op->inputs.size() - INPUT_TENSOR_INDEX; ++i) {
            auto ethosu_idx = ethosu_op->inputs[i + INPUT_TENSOR_INDEX];
            auto &tensor = model->subgraphs[0]->tensors[ethosu_idx];
            op.memory_layout.input_offset[i] = address_offsets[ethosu_idx] + arena_offset;
            op.memory_layout.input_size[i] = GetTensorDataSize(tensor);

            auto tflite_idx = FindTfliteTensorIndex(tensor->name);
            TF_LITE_ENSURE(context, tflite_idx != -1);
            TfLiteCustomAllocation allocation({ethosu_context->arena_buffer->data() + op.memory_layout.input_offset[i],
			    op.memory_layout.input_size[i]});
            this_subgraph->SetCustomAllocationForTensor(tflite_idx, allocation,
			    kTfLiteCustomAllocationFlagsSkipAlignCheck);
        }
      }
    } catch (exception &e) {
        TF_LITE_KERNEL_LOG(context, "Failed to alloc ethos_u buffer.\n");
        return kTfLiteDelegateError;
    }

    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    try {
      for (auto& op : operations) {
        /* make sure the wait completes ok */
        if (op.inference->invoke(options.timeout) <= 0) {
          TF_LITE_KERNEL_LOG(context, "Ethos_u inference failed\n");
          return kTfLiteDelegateError;
        }

        /* Read out PMU counters if configured */
        if (pmu_counter_config.size() > 0) {
          const vector<uint32_t> pmus = op.inference->getPmuCounters();
          cout << "Ethos_u PMUs : [";
          for (auto p : pmus) {
            cout << " " << p;
          }
          cout << " ]" << endl;
        }
        if (options.enable_cycle_counter) {
          cout << "Ethos-u cycle counter: " << op.inference->getCycleCounter() << endl;
        }
        if (options.enable_profiling) {
          auto count = ethosu_context->qread_buffer->size() / sizeof(EthosuQreadEvent);
          if (count == options.profiling_buffer_size) {
            cout << "Ethos_u profiling_buffer_size is too small, please set a larger number" << endl;
          } else {
            auto qread_buffer =
		    reinterpret_cast<EthosuQreadEvent*>(ethosu_context->qread_buffer->data());
            cout << "Ethos_u qread profiling data." << endl;
            for (int i =0; i < count; i ++) {
                cout << "index:" << i << ",qread:0x" << hex << qread_buffer[i].qread
                     << ",status:0x" << hex << qread_buffer[i].status
                     << ",cycle count:" << dec << qread_buffer[i].cycleCount << endl;
            }
          }
        }
      }
    } catch (exception &e) {
      TF_LITE_KERNEL_LOG(context, "Failed to invoke ethos_u op.\n");
      return kTfLiteDelegateError;
    }

    return kTfLiteOk;
  }

 private:
  const EthosuDelegateOptions options;
  TfLiteEthosuContext* ethosu_context;
  struct OperationDataType {
    shared_ptr<EthosU::Buffer> ethosu_net_buffer;  //Buffer for cms tensor
    shared_ptr<EthosU::Network> ethosu_network;
    shared_ptr<EthosU::Inference> inference;
    EthosU::MemoryLayout memory_layout;
    bool need_flash;
    //for vela model
    vector<int> inputs;
    vector<int> outputs;
  };

  //for none vela model
  std::unique_ptr<ModelT> model;

  vector<OperationDataType> operations;
  const int32_t *address_offsets;
  uint32_t flash_offset;
  uint32_t arena_offset;
  vector<uint32_t> pmu_counter_config;
};

// EthosuDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class EthosuDelegate : public SimpleDelegateInterface {
 public:
  explicit EthosuDelegate(const EthosuDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    if (registration->builtin_code == kTfLiteBuiltinCustom &&
        strcmp(registration->custom_name, ETHOSU_CUSTOM_NAME) == 0)
      return true;
    return IsNodeSupportedByEthosU(context, node, registration->builtin_code);
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    try {
        TfLiteEthosuContext *ethosu_context = new TfLiteEthosuContext;
        ethosu_context->device =
		EthosU::Device::GetSingleton(options_.device_name.c_str());

        if (options_.enable_profiling && options_.profiling_buffer_size != 0){
            size_t size = sizeof(EthosuQreadEvent) * options_.profiling_buffer_size;
            ethosu_context->qread_buffer =
		    make_shared<EthosU::Buffer>(*ethosu_context->device, size);
            ethosu_context->qread_buffer->resize(0);
        } else {
            ethosu_context->qread_buffer = nullptr;
        }
	ethosu_context->arena_buffer = nullptr;
	ethosu_context->flash_buffer = nullptr;
	ethosu_context->arena_size = 0;
	ethosu_context->flash_size = 0;
	context_map_[context] = ethosu_context;
    } catch (exception &e) {
        TF_LITE_KERNEL_LOG(context, "Failed to create ethos_u driver.\n");
        return kTfLiteDelegateError;
    }

    return kTfLiteOk;
  }

  void *GetDelegateContext(TfLiteContext* context) const{
      return context_map_.at(context);
  }

  const char* Name() const override {
    static constexpr char kName[] = "EthosuDelegate";
    return kName;
  }

  unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return make_unique<EthosuDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

  ~EthosuDelegate() {
    std::map<TfLiteContext*, void*>::iterator itr;
    for (itr = context_map_.begin(); itr != context_map_.end(); ++itr) {
       delete (TfLiteEthosuContext*)itr->second;
    }
  }


 private:
  const EthosuDelegateOptions options_;
  std::map<TfLiteContext*, void*> context_map_;
};

}  // namespace ethosu
}  // namespace tflite

EthosuDelegateOptions EthosuDelegateOptionsDefault() {
  EthosuDelegateOptions options;

  options.device_name = ETHOSU_DEFAULT_DEVICE_NAME;
  options.cache_file_path = "";
  options.timeout = ETHOSU_DEFAULT_TIMEOUT;
  options.enable_cycle_counter = false;
  options.enable_profiling = false;
  options.profiling_buffer_size = DEFAULT_QREAD_BUFFER_SIZE;
  memset(options.pmu_counter_config, 0, sizeof(options.pmu_counter_config));

  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteEthosuDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* EthosuDelegateCreate(const EthosuDelegateOptions* options) {
  auto delegate = make_unique<tflite::ethosu::EthosuDelegate>(
          options ? *options : EthosuDelegateOptionsDefault());
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(move(delegate), 
             kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `EthosuDelegateCreate` call.
void EthosuDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
