/*
 * Copyright 2025 Ladislav Hano
 * Copyright 2022-2023 NXP
 * Copyright (c) 2020-2021 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "ethosu_drv.h"

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

using namespace std;

namespace EthosU {
__attribute__((weak)) int eioctl(int fd, unsigned long cmd, void *data) {
    int ret = ::ioctl(fd, cmd, data);
    if (ret < 0) {
        throw Exception("IOCTL failed");
    }

    return ret;
}

__attribute__((weak)) int eopen(const char *pathname, int flags) {
    int fd = ::open(pathname, flags);
    if (fd < 0) {
        throw Exception("Failed to open device");
    }

    return fd;
}

__attribute__((weak)) int
eppoll(struct pollfd *fds, nfds_t nfds, const struct timespec *tmo_p, const sigset_t *sigmask) {
    int result = ::ppoll(fds, nfds, tmo_p, sigmask);
    if (result < 0) {
        throw Exception("Failed to wait for ppoll event or signal");
    }

    return result;
}

__attribute__((weak)) int eclose(int fd) {
    int result = ::close(fd);
    if (result < 0) {
        throw Exception("Failed to close file");
    }

    return result;
}
__attribute((weak)) void *emmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void *ptr = ::mmap(addr, length, prot, flags, fd, offset);
    if (ptr == MAP_FAILED) {
        throw Exception("Failed to mmap file");
    }

    return ptr;
}

__attribute__((weak)) int emunmap(void *addr, size_t length) {
    int result = ::munmap(addr, length);
    if (result < 0) {
        throw Exception("Failed to munmap file");
    }

    return result;
}

} // namespace EthosU

namespace EthosU {

/****************************************************************************
 * Exception
 ****************************************************************************/

Exception::Exception(const char *msg) : msg(msg) {}

Exception::~Exception() throw() {}

const char *Exception::what() const throw() {
    return msg.c_str();
}

/****************************************************************************
 * Semantic Version
 ****************************************************************************/

bool SemanticVersion::operator==(const SemanticVersion &other) {
    return other.major == major && other.minor == minor && other.patch == patch;
}

bool SemanticVersion::operator<(const SemanticVersion &other) {
    if (other.major > major)
        return true;
    if (other.minor > minor)
        return true;
    return other.patch > patch;
}

bool SemanticVersion::operator<=(const SemanticVersion &other) {
    return *this < other || *this == other;
}

bool SemanticVersion::operator!=(const SemanticVersion &other) {
    return !(*this == other);
}

bool SemanticVersion::operator>(const SemanticVersion &other) {
    return !(*this <= other);
}

bool SemanticVersion::operator>=(const SemanticVersion &other) {
    return !(*this < other);
}

ostream &operator<<(ostream &out, const SemanticVersion &v) {
    return out << "{ major=" << unsigned(v.major) << ", minor=" << unsigned(v.minor) << ", patch=" << unsigned(v.patch)
               << " }";
}

} // namespace EthosU
