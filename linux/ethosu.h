/*
 * Copyright 2025 Ladislav Hano
 * Copyright 2022-2023 NXP
 * Copyright (c) 2020-2021 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <poll.h>
#include <signal.h>

#include "ethosu_drv.h"
#include "linux/ethosu.h"
/*
 *The following undef are necessary to avoid clash with macros in GNU C Library
 * if removed the following warning/error are produced:
 *
 *  In the GNU C Library, "major" ("minor") is defined
 *  by <sys/sysmacros.h>. For historical compatibility, it is
 *  currently defined by <sys/types.h> as well, but we plan to
 *  remove this soon. To use "major" ("minor"), include <sys/sysmacros.h>
 *  directly. If you did not intend to use a system-defined macro
 *  "major" ("minor"), you should undefine it after including <sys/types.h>.
 */
#undef major
#undef minor

namespace EthosU {

int eioctl(int fd, int cmd, void *data = nullptr);
int eopen(const char *pathname, int flags);
int eppoll(struct pollfd *fds, nfds_t nfds, const struct timespec *tmo_p, const sigset_t *sigmask);
int eclose(int fd);
void *emmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int emunmap(void *addr, size_t length);

typedef ethosu_uapi_memory_layout MemoryLayout;

class Device {
public:
    static Device *GetSingleton(const char *device = "/dev/ethosu0");
    virtual ~Device();

    int ioctl(unsigned long cmd, void *data = nullptr) const;
    Capabilities capabilities() const;

private:
    Device(const char *device);
    int fd;
};

class Buffer {
public:
    Buffer(const Device &device, const size_t capacity);
    virtual ~Buffer();

    size_t capacity() const;
    void extend(const Device &device, const size_t capacity);
    void clear() const;
    char *data() const;
    void resize(size_t size, size_t offset = 0) const;
    size_t offset() const;
    size_t size() const;

    int getFd() const;

private:
    int fd;
    char *dataPtr;
    size_t dataCapacity;
};

class Network {
public:
    Network(const Device &device, std::shared_ptr<Buffer> &buffer);
    virtual ~Network();

    int ioctl(unsigned long cmd, void *data = nullptr);
    std::shared_ptr<Buffer> getBuffer();

private:
    int fd;
    std::shared_ptr<Buffer> buffer;
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
        network(network) {
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
        network(network) {
        std::copy(ifmBegin, ifmEnd, std::back_inserter(ifmBuffers));
        std::copy(ofmBegin, ofmEnd, std::back_inserter(ofmBuffers));
        std::vector<uint32_t> counterConfigs = initializeCounterConfig();

        if (counters.size() > counterConfigs.size())
            throw EthosU::Exception("PMU Counters argument to large.");

        std::copy(counters.begin(), counters.end(), counterConfigs.begin());
        create(counterConfigs, enableCycleCounter, layout);
    }

    virtual ~Inference();

    int wait(int64_t timeoutNanos = -1) const;
    int invoke(int64_t timeoutNanos = -1) const;
    const std::vector<uint32_t> getPmuCounters() const;
    uint64_t getCycleCounter() const;
    bool failed() const;
    int getFd() const;
    const std::shared_ptr<Network> getNetwork() const;
    std::vector<std::shared_ptr<Buffer>> &getIfmBuffers();
    std::vector<std::shared_ptr<Buffer>> &getOfmBuffers();

    static uint32_t getMaxPmuEventCounters();

private:
    void create(std::vector<uint32_t> &counterConfigs, bool enableCycleCounter, MemoryLayout layout);
    std::vector<uint32_t> initializeCounterConfig();

    int fd;
    const std::shared_ptr<Network> network;
    std::vector<std::shared_ptr<Buffer>> ifmBuffers;
    std::vector<std::shared_ptr<Buffer>> ofmBuffers;
};

} // namespace EthosU
