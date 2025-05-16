/*
 * Copyright 2025 Ladislav Hano
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "ethosu_resmgr_api.h"
#include "ethosu.h"

#include <memory>
#include <vector>
#include <stdlib.h>

namespace EthosU {

void Inference::create(std::vector<uint32_t> &counterConfigs, bool cycleCounterEnable, EthosU::MemoryLayout layout) {
    if (ifmBuffers.size() > ETHOSU_FD_MAX) {
        throw Exception("IFM buffer overflow");
    }

    if (ofmBuffers.size() > ETHOSU_FD_MAX) {
        throw Exception("OFM buffer overflow");
    }

    if (counterConfigs.size() != ETHOSU_PMU_EVENT_MAX) {
        throw Exception("Wrong size of counter configurations");
    }

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i++) {
        pmu_config_events[i] = counterConfigs.at(i);
    }

    pmu_config_cycle_count = cycleCounterEnable;

    this->layout = layout;
}

std::vector<uint32_t> Inference::initializeCounterConfig() {
    return std::vector<uint32_t>(ETHOSU_PMU_EVENT_MAX, 0);
}

uint32_t Inference::getMaxPmuEventCounters() {
    return ETHOSU_PMU_EVENT_MAX;
}

int Inference::invoke(int64_t timeoutNanos) {
    ethosu_resmgr_inference_msg_t msg = { 0 };

    pmu_config_cycle_count = 0;

    msg.tx.ifm_count = 0;
    for (auto buf : ifmBuffers) {
        msg.tx.ifm[msg.tx.ifm_count++] = buf->construct_api_buffer();
    }

    msg.tx.ofm_count = 0;
    for (auto buf : ofmBuffers) {
        msg.tx.ofm[msg.tx.ofm_count++] = buf->construct_api_buffer();
    }

    msg.tx.network_buffer = network.get()->construct_api_network();

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; ++i) {
        msg.tx.pmu_config[i] = pmu_config_events[i];
    }

    msg.tx.arena_offset = layout.arena_offset;
    msg.tx.flash_offset = layout.flash_offset;
    msg.tx.inference_type = type;

    int ret = network->ioctl(ETHOSU_IOCTL_INFERENCE_INVOKE, &msg);
    if (ret != 0) {
        throw Exception("Problem with ioctl call");
    }

    if (msg.rx.status == ETHOSU_STATUS_OK) {
        pmu_count_cycle_count = msg.rx.pmu_cycle_counter_count;
        for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; ++i) {
            pmu_count_events[i] = msg.rx.pmu_event_count[i];
        }
    }

    status.emplace(msg.rx.status);

    // return 1 as success since the delegate considers <= 0 as error
    return msg.rx.status == ETHOSU_STATUS_OK ? 1 : 0;
}

bool Inference::failed() const {
    return status.value_or(ETHOSU_STATUS_ERROR) != ETHOSU_STATUS_OK;
}

const std::vector<uint32_t> Inference::getPmuCounters() const {
    std::vector<uint32_t> counterValues = std::vector<uint32_t>(ETHOSU_PMU_EVENT_MAX);

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i++) {
        if (pmu_config_events[i]) {
            counterValues.at(i) = pmu_count_events[i];
        }
    }

    return counterValues;
}

uint64_t Inference::getCycleCounter() const {
    return pmu_count_cycle_count;
}

const std::shared_ptr<Network> Inference::getNetwork() const {
    return network;
}

std::vector<std::shared_ptr<Buffer>> &Inference::getIfmBuffers() {
    return ifmBuffers;
}

std::vector<std::shared_ptr<Buffer>> &Inference::getOfmBuffers() {
    return ofmBuffers;
}

} // namespace EthosU
