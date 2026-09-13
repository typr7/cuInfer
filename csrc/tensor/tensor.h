#pragma once

#include <array>
#include <stdexcept>

#include "tensor/data_type.h"


namespace cuinfer
{

template <int kDim>
struct Tensor
{
    static_assert(kDim > 0);

    void* device_ptr = nullptr;
    DataType dtype = DataType::kUnsupported;
    std::array<int, kDim> shape;
    std::array<int, kDim> stride;

    explicit operator bool() const noexcept { return device_ptr != nullptr; }

    template <typename T>
    T* data() const
    {
        constexpr DataType expected = dtype_of<T>();
        static_assert(expected != DataType::kUnsupported);
        
        if (dtype != expected) {
            throw std::runtime_error("unmatched dtype");
        }

        return static_cast<T*>(device_ptr);
    }
};

template <int kDim>
Tensor<kDim> make_tensor(
    void* device_ptr,
    DataType dtype,
    const std::array<int, kDim>& shape,
    const std::array<int, kDim>& stride
)
{
    return Tensor<kDim>{
        .device_ptr = device_ptr,
        .dtype = dtype,
        .shape = shape,
        .stride = stride
    };
}

template <int kDim>
Tensor<kDim> make_tensor(void* device_ptr, DataType dtype, const std::array<int, kDim>& shape)
{
    std::array<int, kDim> stride;
    int s = 1;
    for (int i = kDim - 1; i >= 0; i--) {
        stride[i] = s;
        s *= shape[i];
    }
    return make_tensor<kDim>(device_ptr, dtype, shape, stride);
}

template <int kDim>
using TensorRef = const Tensor<kDim>&;

}
