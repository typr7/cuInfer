#include "cuda/pinned_buffer.h"
#include "cuda/cuda_utils.h"

namespace cuinfer
{

PinnedBuffer::PinnedBuffer(std::size_t byte_size)
    : byte_size_(byte_size)
{
    CUDA_CHECK(cudaMallocHost(&data_, byte_size));
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      byte_size_(std::exchange(other.byte_size_, 0))
{
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept
{
    if (this != &other) {

        if (data_) {
            cudaFreeHost(data_);
        }

        data_ = std::exchange(other.data_, nullptr);
        byte_size_ = std::exchange(other.byte_size_, 0);
    }

    return *this;
}

PinnedBuffer::~PinnedBuffer() noexcept
{
    if (data_) {
        cudaFreeHost(data_);
        data_ = nullptr;
    }
}

void PinnedBuffer::resize(std::size_t byte_size)
{
    PinnedBuffer tmp(byte_size);
    std::swap(*this, tmp);
}

}
