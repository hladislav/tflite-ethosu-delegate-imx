/*
 * Copyright 2025 Ladislav Hano
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#ifndef ETHOSU_API_H
#define ETHOSU_API_H

/**
 * Includes
 */

#include <stdint.h>
#include <devctl.h>

/**
 * Defines
 */

#define ETHOSU_BUFFER_MAX           4

#define ETHOSU_FD_MAX               16
#define ETHOSU_PMU_EVENT_MAX        4

/**
 * Types
 */

enum ethosu_resmgr_status
{
    ETHOSU_STATUS_OK,
    ETHOSU_STATUS_ERROR,
};

typedef struct ethosu_resmgr_memory_layout
{
    uint32_t flash_offset;
    uint32_t arena_offset;

    uint32_t input_count;
    uint32_t input_offset[ETHOSU_FD_MAX];
    uint32_t input_size[ETHOSU_FD_MAX];

    uint32_t output_count;
    uint32_t output_offset[ETHOSU_FD_MAX];
    uint32_t output_size[ETHOSU_FD_MAX];
} ethosu_resmgr_memory_layout_t;

typedef struct ethosu_resmgr_buffer
{
    uint32_t paddr;
    uint32_t size;
} ethosu_resmgr_buffer_t;

typedef struct ethosu_resmgr_inference_request
{
    size_t timeout;

    uint32_t                ifm_count;
    ethosu_resmgr_buffer_t  ifm[ETHOSU_BUFFER_MAX];

    uint32_t                ofm_count;
    ethosu_resmgr_buffer_t  ofm[ETHOSU_BUFFER_MAX];

    ethosu_resmgr_buffer_t  network_buffer;

    uint8_t                 pmu_config_count;
    uint8_t                 pmu_config[ETHOSU_PMU_EVENT_MAX];
    uint8_t                 pmu_cycle_counter_enable;

    uint32_t                flash_offset;
    uint32_t                arena_offset;

    uint32_t                inference_type;
} ethosu_resmgr_inference_request_t;

typedef struct ethosu_resmgr_inference_response
{
    uint64_t user_data;
    uint32_t ofm_count;
    uint32_t ofm_size[ETHOSU_BUFFER_MAX];
    uint32_t status;
    uint8_t  pmu_event_config[ETHOSU_PMU_EVENT_MAX];
    uint32_t pmu_event_count[ETHOSU_PMU_EVENT_MAX];
    uint32_t pmu_cycle_counter_enable;
    uint64_t pmu_cycle_counter_count;
} ethosu_resmgr_inference_response_t;

typedef union ethosu_resmgr_inference_msg
{
    ethosu_resmgr_inference_request_t   tx; // Filled by client
    ethosu_resmgr_inference_response_t  rx; // Filled by device
} ethosu_resmgr_inference_msg_t;

typedef struct ethosu_resmgr_pmu_config
{
    uint32_t events[ETHOSU_PMU_EVENT_MAX];
    uint32_t cycle_count;
} ethosu_resmgr_pmu_config_t;

typedef struct ethosu_resmgr_pmu_counts
{
    uint32_t events[ETHOSU_PMU_EVENT_MAX];
    uint64_t cycle_count;
} ethosu_resmgr_pmu_counts_t;

typedef struct ethosu_resmgr_device_hw_id
{
    uint32_t version_status;
    uint32_t version_minor;
    uint32_t version_major;

    uint32_t product_major;

    uint32_t arch_patch_rev;
    uint32_t arch_minor_rev;
    uint32_t arch_major_rev;
} ethosu_resmgr_device_hw_id_t;

typedef struct ethosu_resmgr_device_hw_cfg
{
    uint32_t macs_per_cc;
    uint32_t cmd_stream_version;
    uint32_t custom_dma;
} ethosu_resmgr_device_hw_cfg_t;

typedef struct ethosu_api_device_capabilities
{
    ethosu_resmgr_device_hw_id_t hw_id;
    ethosu_resmgr_device_hw_cfg_t hw_cfg;

    uint32_t driver_patch_rev;
    uint32_t driver_minor_rev;
    uint32_t driver_major_rev;
} ethosu_api_device_capabilities_t;

enum ethosu_api_inference_type
{
    ETHOSU_API_INFERENCE_MODEL = 0,
    ETHOSU_API_INFERENCE_OP
};

typedef struct ethosu_api_result_status
{
    enum ethosu_resmgr_status status;
    ethosu_resmgr_pmu_config_t pmu_config;
    ethosu_resmgr_pmu_counts_t pmu_count;
} ethosu_api_result_status_t;

/**
 * DEVCTL API
 */

#define ETHOSU_DEVCTL_BASE              (0x01)
#define ETHOSU_DEVCTL_CMD(index)        (ETHOSU_DEVCTL_BASE + index)

#define ETHOSU_DEVCTL_PING              __DION(_DCMD_MISC, ETHOSU_DEVCTL_CMD(0x00))
#define ETHOSU_DEVCTL_VERSION_REQ       __DION(_DCMD_MISC, ETHOSU_DEVCTL_CMD(0x01))
#define ETHOSU_DEVCTL_CAPABILITIES_REQ  __DIOF(_DCMD_MISC, ETHOSU_DEVCTL_CMD(0x02), ethosu_api_device_capabilities_t)

// This one is implemented in the Linux module but never used in delegate so it is not supported for now
#define ETHOSU_DEVCTL_NETWORK_INFO      __DIOF(_DCMD_MISC, ETHOSU_DEVCTL_CMD(0x21), NULL)

// We dont need to use this one since we return all info in Invoke
#define ETHOSU_DEVCTL_INFERENCE_STATUS  __DIOTF(_DCMD_MISC, ETHOSU_DEVCTL_CMD(0x31), ethosu_api_result_status_t)

// This one is not implemented since it is not used by the delegate
#define ETHOSU_DEVCTL_INFERENCE_CANCEL  __DIOTF(_DCMD_MISC, ETHOSU_DEVCTL_CMD(0x32), NULL)
#define ETHOSU_DEVCTL_INFERENCE_INVOKE  __DIOTF(_DCMD_MISC, ETHOSU_DEVCTL_CMD(0x33), ethosu_resmgr_inference_msg_t)

/**
 * Override ioctl API
 */
enum ethosu_ioctl
{
    ETHOSU_IOCTL_PING             = ETHOSU_DEVCTL_PING,
    ETHOSU_IOCTL_VERSION_REQ      = ETHOSU_DEVCTL_VERSION_REQ,
    ETHOSU_IOCTL_CAPABILITIES_REQ = ETHOSU_DEVCTL_CAPABILITIES_REQ,
    ETHOSU_IOCTL_INFERENCE_INVOKE = ETHOSU_DEVCTL_INFERENCE_INVOKE
};

#endif