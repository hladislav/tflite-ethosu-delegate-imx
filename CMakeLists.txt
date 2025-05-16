#
# Copyright 2020-2022 NXP
# Copyright 2025 Ladislav Hano
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(tflite_ethosu_delegate)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH}
)

find_package(tensorflow REQUIRED)

list(APPEND ETHOSU_DELEGATES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_drv.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_delegate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_delegate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/external_delegate_adaptor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_delegate_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_supported_ops.cc
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  list(APPEND ETHOSU_DELEGATES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/qnx/src/ethosu_buffer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/qnx/src/ethosu_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/qnx/src/ethosu_inference.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/qnx/src/ethosu_network.cc
  )
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/qnx/include)
else()
  list(APPEND ETHOSU_DELEGATES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/linux/ethosu_linux_drv.cc
  )
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/linux)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "QNX")
  list(APPEND ETHOSU_DELEGATE_DEPENDENCIES "python3")
  include_directories("${QNX_INCLUDE_PATH}/python3.11")
  include_directories("${QNX_INCLUDE_PATH}/${arch}/python3.11")
  set(PYTHON_LIB_NAME "libpython3.11.so")
else()
  find_package (Python COMPONENTS Interpreter Development)
  get_filename_component(PYTHON_LIB_NAME ${Python_LIBRARIES} NAME)
  include_directories(${Python_INCLUDE_DIRS})
  list(APPEND ETHOSU_DELEGATE_DEPENDENCIES ${Python_LIBRARIES})
endif()

add_library(ethosu_delegate SHARED ${ETHOSU_DELEGATES_SRCS})
target_link_libraries(ethosu_delegate ${ETHOSU_DELEGATE_DEPENDENCIES})
target_compile_options(ethosu_delegate PUBLIC "-DPYTHON_LIB=${PYTHON_LIB_NAME}")
