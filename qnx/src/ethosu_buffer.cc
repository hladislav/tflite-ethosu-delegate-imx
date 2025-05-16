/*
 * Copyright 2025 Ladislav Hano
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <sys/mman.h>
#include <fcntl.h>

#include "ethosu.h"

using namespace EthosU;

namespace EthosU {

Buffer::Buffer(const Device &device, const size_t capacity) : 
    bufferCapacity(capacity), bufferOffset(0), bufferSize(capacity),
    pAddr_offset(0), fd(-1), dataPtr(nullptr)
{
    allocateBuffer(capacity);
}

Buffer::~Buffer() {
    if (dataPtr != nullptr) {
        emunmap(dataPtr, bufferCapacity);
    }

    if (fd != -1) {
        eclose(fd);
    }
}

ethosu_resmgr_buffer_t Buffer::construct_api_buffer() {
    ethosu_resmgr_buffer_t result = {
        // !!! -Wnarrowing trying to put size_t to uint32_t
        .paddr = pAddr_offset + bufferOffset,
        .size = bufferSize - bufferOffset
    };

    return result;
}

void Buffer::extend(const Device &device, const size_t capacity) {
    if (capacity < bufferCapacity) {
        return;
    }

    emunmap(dataPtr, bufferCapacity);
    dataPtr = nullptr;
    pAddr_offset = 0;

    eclose(fd);

    // ? want to propagate the exception further
    allocateBuffer(capacity);
}

void Buffer::allocateBuffer(size_t capacity) {
    fd = posix_typed_mem_open("/memory/below4G", O_RDWR, POSIX_TYPED_MEM_ALLOCATE_CONTIG);
    if (fd == -1) {
        throw "Failed to open typed memory!";
    }

    try {
        dataPtr = reinterpret_cast<char *>(emmap(nullptr, capacity, PROT_READ | PROT_WRITE | PROT_NOCACHE, MAP_SHARED, fd, 0));
    } catch (std::exception &e) {
        try {
            eclose(fd);
        } catch (...) { std::throw_with_nested(e); }
    }

    if (mem_offset(dataPtr, NOFD, capacity, &pAddr_offset, NULL) != 0) {
        throw "Failed to get physical memory offset";
    }

    bufferCapacity = capacity;
}

size_t Buffer::capacity() const {
    return bufferCapacity;
}

void Buffer::clear() {
    resize(0, 0);
}

char *Buffer::data() const {
    return dataPtr + offset();
}

void Buffer::resize(size_t size, size_t offset) {
    this->bufferOffset = offset;
    this->bufferSize = size;
}

size_t Buffer::offset() const {
    return this->bufferOffset;
}

size_t Buffer::size() const {
    return this->bufferSize;
}

int Buffer::getFd() {
    return fd;
}

}
