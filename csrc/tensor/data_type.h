#pragma once

#include <cstddef>
#include <string_view>
#include <type_traits>


struct __nv_bfloat16;

namespace cllm
{

enum class DataType
{
    kUnsupported,
    kBf16,
    kFp32,
};

inline constexpr std::size_t dtype_byte_size(DataType dtype) noexcept
{
    switch (dtype) {
        case DataType::kBf16:
            return 2;
        case DataType::kFp32:
            return 4;
        case DataType::kUnsupported:
        default:
            return 0;
    }
}

inline constexpr std::string_view dtype_string(DataType dtype) noexcept
{
    switch (dtype) {
        case DataType::kBf16:
            return "BF16";
        case DataType::kFp32:
            return "FP32";
        case DataType::kUnsupported:
        default:
            return "UNSUPPORTED";
    }
}

template <typename T>
consteval DataType dtype_of() noexcept
{
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return DataType::kBf16;
    } else if (std::is_same_v<T, float>) {
        return DataType::kFp32;
    } else {
        return DataType::kUnsupported;
    }
}

}
