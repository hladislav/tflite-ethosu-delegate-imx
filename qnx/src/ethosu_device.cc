/*
 * Copyright 2025 Ladislav Hano
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "ethosu_resmgr_api.h"
#include "ethosu_drv.h"
#include "ethosu.h"

#include <utility>
#include <string.h>
#include <vector>

#include <algorithm>
#include <queue>
#include <exception>
#include <iostream>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace EthosU;
using namespace std;

namespace EthosU {

Device::Device(const char *device) {
    fd = eopen(device, O_RDWR | O_NONBLOCK);

    if (eioctl(fd, ETHOSU_DEVCTL_VERSION_REQ) != 0) {
        throw "Resource manager and core versions differ!";
    }
}

Device::~Device() {
    eclose(fd);
}

Device* Device::GetSingleton(const char *device){
    static Device dev(device);
    return &dev;
}

int Device::ioctl(int cmd, void *data) const {
    switch (cmd)
    {
        case ETHOSU_IOCTL_PING:
        case ETHOSU_IOCTL_VERSION_REQ:
            return eioctl(fd, cmd, NULL);
        case ETHOSU_IOCTL_CAPABILITIES_REQ:
            if (data == nullptr) {
                throw Exception("ETHOSU_IOCTL_CAPABILITIES_REQ requires data, was nullptr");
            }

            return edevctl<ethosu_api_device_capabilities_t>(fd, cmd, data);

        case ETHOSU_IOCTL_INFERENCE_INVOKE:
            if (data == nullptr) {
                throw Exception("ETHOSU_IOCTL_INFERENCE_INVOKE requires data, was nullptr");
            }

            return edevctl<ethosu_resmgr_inference_msg_t>(fd, cmd, data);
    }

    return ENOTSUP;
}

Capabilities Device::capabilities() const {
    ethosu_api_device_capabilities_t msg = { 0 };

    // TODO do something with the result value
    this->ioctl( ETHOSU_DEVCTL_CAPABILITIES_REQ, &msg);

    Capabilities capabilities(
        HardwareId(msg.hw_id.version_status,
                SemanticVersion(msg.hw_id.version_major, msg.hw_id.version_minor),
                SemanticVersion(msg.hw_id.product_major),
                SemanticVersion(msg.hw_id.arch_major_rev, msg.hw_id.arch_minor_rev, msg.hw_id.arch_patch_rev)),
        HardwareConfiguration(msg.hw_cfg.macs_per_cc, msg.hw_cfg.cmd_stream_version, bool(msg.hw_cfg.custom_dma)),
        SemanticVersion(msg.driver_major_rev, msg.driver_minor_rev, msg.driver_patch_rev));
    return capabilities;
}

}