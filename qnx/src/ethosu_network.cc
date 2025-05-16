/*
 * Copyright 2025 Ladislav Hano
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

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

ethosu_resmgr_buffer_t Network::construct_api_network() {
    return buffer.get()->construct_api_buffer();
}

shared_ptr<Buffer> Network::getBuffer() {
    return buffer;
}

int Network::ioctl(int cmd, void *data) {
    return device.ioctl(cmd, data);
}

}   // namespace EthosU
