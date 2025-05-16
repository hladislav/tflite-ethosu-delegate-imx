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

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
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

int eioctl(int fd, unsigned long cmd, void *data = nullptr);
int eopen(const char *pathname, int flags);
int eppoll(struct pollfd *fds, nfds_t nfds, const struct timespec *tmo_p, const sigset_t *sigmask);
int eclose(int fd);
void *emmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int emunmap(void *addr, size_t length);

class Exception : public std::exception {
public:
    Exception(const char *msg);
    virtual ~Exception() throw();
    virtual const char *what() const throw();

private:
    std::string msg;
};

/**
 * Sematic Version : major.minor.patch
 */
class SemanticVersion {
public:
    SemanticVersion(uint32_t _major = 0, uint32_t _minor = 0, uint32_t _patch = 0) :
        major(_major), minor(_minor), patch(_patch){};

    bool operator==(const SemanticVersion &other);
    bool operator<(const SemanticVersion &other);
    bool operator<=(const SemanticVersion &other);
    bool operator!=(const SemanticVersion &other);
    bool operator>(const SemanticVersion &other);
    bool operator>=(const SemanticVersion &other);

    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

std::ostream &operator<<(std::ostream &out, const SemanticVersion &v);

/*
 * Hardware Identifier
 * @versionStatus:             Version status
 * @version:                   Version revision
 * @product:                   Product revision
 * @architecture:              Architecture revison
 */
struct HardwareId {
public:
    HardwareId(uint32_t _versionStatus,
               const SemanticVersion &_version,
               const SemanticVersion &_product,
               const SemanticVersion &_architecture) :
        versionStatus(_versionStatus),
        version(_version), product(_product), architecture(_architecture) {}

    uint32_t versionStatus;
    SemanticVersion version;
    SemanticVersion product;
    SemanticVersion architecture;
};

/*
 * Hardware Configuration
 * @macsPerClockCycle:         MACs per clock cycle
 * @cmdStreamVersion:          NPU command stream version
 * @customDma:                 Custom DMA enabled
 */
struct HardwareConfiguration {
public:
    HardwareConfiguration(uint32_t _macsPerClockCycle, uint32_t _cmdStreamVersion, bool _customDma) :
        macsPerClockCycle(_macsPerClockCycle), cmdStreamVersion(_cmdStreamVersion), customDma(_customDma) {}

    uint32_t macsPerClockCycle;
    uint32_t cmdStreamVersion;
    bool customDma;
};

/**
 * Device capabilities
 * @hwId:                      Hardware
 * @driver:                    Driver revision
 * @hwCfg                      Hardware configuration
 */
class Capabilities {
public:
    Capabilities(const HardwareId &_hwId, const HardwareConfiguration &_hwCfg, const SemanticVersion &_driver) :
        hwId(_hwId), hwCfg(_hwCfg), driver(_driver) {}

    HardwareId hwId;
    HardwareConfiguration hwCfg;
    SemanticVersion driver;
};

} // namespace EthosU
