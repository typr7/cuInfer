#pragma once

#include <cstddef>


namespace cllm
{

class PinnedBuffer
{
public:
    PinnedBuffer() = default;

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(std::size_t byte_size);

    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    ~PinnedBuffer() noexcept;

    void resize(std::size_t byte_size);

    void upload_at(std::size_t dst_offset, const void* src, std::size_t byte_size);

    template <typename T = void>
    T* data() const noexcept
    {
        return static_cast<T*>(data_);
    }

    std::size_t size() const noexcept
    {
        return byte_size_;
    }

private:
    void* data_ = nullptr;
    std::size_t byte_size_ = 0;
};

}