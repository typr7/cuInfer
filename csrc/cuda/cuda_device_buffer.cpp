#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>
#include <utility>

#include <cuda_runtime.h>

#include "cuda/cuda_device_buffer.h"
#include "cuda/cuda_utils.h"


namespace {

void allocate_device_memory(void*& ptr, std::size_t byte_size)
{
    CUDA_CHECK(cudaMalloc(&ptr, byte_size));
}

void deallocate_device_memory(void* ptr)
{
    CUDA_CHECK(cudaFree(ptr));
}

void copy_host_to_device(
    void* dst,
    const void* src,
    std::size_t byte_size,
    std::size_t dst_offset = 0
)
{
    CUDA_CHECK(cudaMemcpy(
        static_cast<std::byte*>(dst) + dst_offset,
        src,
        byte_size,
        cudaMemcpyHostToDevice
    ));
}

void copy_device_to_host(
    void* dst,
    const void* src,
    std::size_t byte_size,
    std::size_t src_offset = 0
)
{
    CUDA_CHECK(cudaMemcpy(
        dst,
        static_cast<const std::byte*>(src) + src_offset,
        byte_size,
        cudaMemcpyDeviceToHost
    ));
}

// async
void copy_host_to_device_async(
    void* dst,
    const void* src,
    std::size_t byte_size,
    cudaStream_t stream,
    std::size_t dst_offset = 0
) {
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<std::byte*>(dst) + dst_offset,
        src,
        byte_size,
        cudaMemcpyHostToDevice,
        stream
    ));
}

void copy_device_to_host_async(
    void* dst,
    const void* src,
    std::size_t byte_size,
    cudaStream_t stream,
    std::size_t src_offset = 0
) {
    CUDA_CHECK(cudaMemcpyAsync(
        dst,
        static_cast<const std::byte*>(src) + src_offset,
        byte_size,
        cudaMemcpyDeviceToHost,
        stream
    ));
}

} // namespace

CudaDeviceBuffer::CudaDeviceBuffer()
{
    CUDA_CHECK(cudaFree(nullptr));
}

CudaDeviceBuffer::CudaDeviceBuffer(std::size_t byte_size): byte_size(byte_size)
{
    CUDA_CHECK(cudaFree(nullptr));
    assert(byte_size != 0);
    allocate_device_memory(device_data, byte_size);
}

CudaDeviceBuffer::~CudaDeviceBuffer() noexcept
{
    try {
        destroy();
    } catch (const std::exception& e) {
        std::cerr << "CudaDeviceBuffer destructor caught exception: " << e.what() << '\n';
    }
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept:
    device_data(std::exchange(other.device_data, nullptr)),
    byte_size(std::exchange(other.byte_size, 0))
{
    try {
        CUDA_CHECK(cudaFree(nullptr));
    } catch (...) {
    }
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept
{
    if (this == &other) {
        return *this;
    }

    try {
        destroy();
    } catch (const std::exception& e) {
        std::cerr << "CudaDeviceBuffer move assignment caught exception: " << e.what() << '\n';
    }

    device_data = std::exchange(other.device_data, nullptr);
    byte_size = std::exchange(other.byte_size, 0);
    return *this;
}

std::size_t CudaDeviceBuffer::size() const noexcept
{
    return byte_size;
}

bool CudaDeviceBuffer::empty() const noexcept
{
    return device_data == nullptr;
}

void CudaDeviceBuffer::resize(std::size_t byte_size)
{
    assert(byte_size != 0);
    destroy();
    allocate_device_memory(device_data, byte_size);
    this->byte_size = byte_size;
}

void CudaDeviceBuffer::upload(const void* src, std::size_t byte_size)
{
    assert(device_data != nullptr);
    assert(src != nullptr);
    assert(byte_size != 0 && byte_size <= this->byte_size);
    copy_host_to_device(device_data, src, byte_size);
}

void CudaDeviceBuffer::upload_async(const void* src, std::size_t byte_size, cudaStream_t stream)
{
    assert(device_data != nullptr);
    assert(src != nullptr);
    assert(byte_size != 0 && byte_size <= this->byte_size);
    copy_host_to_device_async(device_data, src, byte_size, stream);
}

void CudaDeviceBuffer::upload_at(std::size_t dst_offset, const void* src, std::size_t byte_size)
{
    assert(device_data != nullptr);
    assert(src != nullptr);
    assert(byte_size != 0);
    assert(dst_offset + byte_size <= this->byte_size);
    copy_host_to_device(device_data, src, byte_size, dst_offset);
}

void CudaDeviceBuffer::upload_at_async(
    std::size_t dst_offset,
    const void* src,
    std::size_t byte_size,
    cudaStream_t stream
)
{
    assert(device_data != nullptr);
    assert(src != nullptr);
    assert(byte_size != 0);
    assert(dst_offset + byte_size <= this->byte_size);
    copy_host_to_device_async(device_data, src, byte_size, stream, dst_offset);
}

std::size_t CudaDeviceBuffer::download(void* dst, std::size_t byte_size) const
{
    assert(device_data != nullptr);
    assert(dst != nullptr);
    assert(byte_size != 0);

    const std::size_t download_size = std::min(byte_size, this->byte_size);
    copy_device_to_host(dst, device_data, download_size);
    return download_size;
}

std::size_t CudaDeviceBuffer::download_async(void* dst, std::size_t byte_size, cudaStream_t stream) const
{
    assert(device_data != nullptr);
    assert(dst != nullptr);
    assert(byte_size != 0);

    const std::size_t download_size = std::min(byte_size, this->byte_size);
    copy_device_to_host_async(dst, device_data, download_size, stream);
    return download_size;
}

void CudaDeviceBuffer::destroy()
{
    if (device_data != nullptr) {
        deallocate_device_memory(device_data);
        device_data = nullptr;
        byte_size = 0;
    }
}
