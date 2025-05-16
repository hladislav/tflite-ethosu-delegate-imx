/*
 * Copyright 2025 Ladislav Hano
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#ifndef ETHOSU_H
#define ETHOSU_H

/****************************************************************************
 * Includes
 ****************************************************************************/

#include "ethosu_resmgr_api.h"
#include "ethosu_drv.h"

#include <stdlib.h>
#include <stdint.h>

#include <sys/ioctl.h>
#include <sys/types.h>

#include <devctl.h>
#include <unistd.h>
#include <poll.h>
#include <sys/types.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#undef major
#undef minor

namespace EthosU {
	typedef ethosu_resmgr_memory_layout_t MemoryLayout;

	template<typename T>
	int edevctl(int fd, int dcmd, void* msg = nullptr) {
		int ret = -1;
		int res = devctl(fd, dcmd, msg, sizeof(T), &ret);
		if (res != EOK) {
			perror("Failed to call devctl");
			return res;
		}
	
		return ret;
	}

	class Device {
		public:
			static Device *GetSingleton(const char *device = "/dev/ethosu0");
			virtual ~Device();
		
			int ioctl(int cmd, void *data = nullptr) const;
			Capabilities capabilities() const;
		
		private:
			Device(const char *device);
			int fd;
	};

	class Buffer {
        public:
			// Device is only used here for parity with Linux implementation
            Buffer(const Device &device, const size_t capacity);
            ~Buffer();

            size_t capacity() const;
            void extend(const Device &device, const size_t capacity);
            void clear();
            char *data() const;
            void resize(size_t size, size_t offset = 0);
            size_t offset() const;
            size_t size() const;

            ethosu_resmgr_buffer_t construct_api_buffer();

			// ! Unlike in Linux implementation this returns fd of the allocated memory
            int getFd();

        private:
            size_t bufferCapacity;
            size_t bufferOffset;
            size_t bufferSize;

            off_t pAddr_offset;

            int fd;
            char *dataPtr;

            void allocateBuffer(size_t capacity);
    };

	class Network {
		public:
			Network(const Device &device, std::shared_ptr<Buffer> &buffer) : device(device), buffer(buffer) {};

			std::shared_ptr<Buffer> getBuffer();
			int ioctl(int cmd, void *data = nullptr);
	
			ethosu_resmgr_buffer_t construct_api_network();
		
		private:
			int fd;
			const Device	&device;
			std::shared_ptr<Buffer> buffer;
			// TODO might need to be shared
	};

	class Inference {
		public:
			template <typename T>
			Inference(const std::shared_ptr<Network> &network,
						const T &ifmBegin,
						const T &ifmEnd,
						const T &ofmBegin,
						const T &ofmEnd,
						MemoryLayout layout) :
				network(network), status(std::nullopt), type(ETHOSU_API_INFERENCE_OP) {
				std::copy(ifmBegin, ifmEnd, std::back_inserter(ifmBuffers));
				std::copy(ofmBegin, ofmEnd, std::back_inserter(ofmBuffers));
				std::vector<uint32_t> counterConfigs = initializeCounterConfig();
		
				create(counterConfigs, false, layout);
			}
			template <typename T, typename U>
			Inference(const std::shared_ptr<Network> &network,
						const T &ifmBegin,
						const T &ifmEnd,
						const T &ofmBegin,
						const T &ofmEnd,
						const U &counters,
						bool enableCycleCounter,
						MemoryLayout layout) :
				network(network), status(std::nullopt), type(ETHOSU_API_INFERENCE_OP) {
				std::copy(ifmBegin, ifmEnd, std::back_inserter(ifmBuffers));
				std::copy(ofmBegin, ofmEnd, std::back_inserter(ofmBuffers));
				std::vector<uint32_t> counterConfigs = initializeCounterConfig();
		
				if (counters.size() > counterConfigs.size())
					throw EthosU::Exception("PMU Counters argument to large.");
		
				std::copy(counters.begin(), counters.end(), counterConfigs.begin());
				create(counterConfigs, enableCycleCounter, layout);
			}
	
			int invoke(int64_t timeoutNanos = -1);
			const std::vector<uint32_t> getPmuCounters() const;
			uint64_t getCycleCounter() const;
			bool failed() const;
			const std::shared_ptr<Network> getNetwork() const;
			std::vector<std::shared_ptr<Buffer>> &getIfmBuffers();
			std::vector<std::shared_ptr<Buffer>> &getOfmBuffers();
		
			static uint32_t getMaxPmuEventCounters();
		
		private:
			void create(std::vector<uint32_t> &counterConfigs, bool enableCycleCounter, MemoryLayout layout);
			std::vector<uint32_t> initializeCounterConfig();

			const std::shared_ptr<Network> network;
			std::vector<std::shared_ptr<Buffer>> ifmBuffers;
			std::vector<std::shared_ptr<Buffer>> ofmBuffers;
			MemoryLayout layout;

			std::optional<uint32_t> status;
	
			uint64_t pmu_config_cycle_count;
			uint32_t pmu_config_events[ETHOSU_PMU_EVENT_MAX];
	
			uint64_t pmu_count_cycle_count;
			uint32_t pmu_count_events[ETHOSU_PMU_EVENT_MAX];
	
			enum ethosu_api_inference_type type;
	};

}	// namespace EthosU

#endif
