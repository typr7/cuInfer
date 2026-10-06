#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>


struct __nv_bfloat16;
struct __half;

namespace cuinfer
{

enum class DataType
{
    kUnsupported,
    kBf16,
    kFp16,
    kFp32,
    kInt32,
    kInt8,
};

inline constexpr std::size_t dtype_byte_size(DataType dtype) noexcept
{
    switch (dtype) {
        case DataType::kBf16:
        case DataType::kFp16:
            return 2;
        case DataType::kFp32:
        case DataType::kInt32:
            return 4;
        case DataType::kInt8:
            return 1;
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
        case DataType::kFp16:
            return "F16";
        case DataType::kFp32:
            return "F32";
        case DataType::kInt32:
            return "I32";
        case DataType::kInt8:
            return "I8";
        case DataType::kUnsupported:
        default:
            return "UNSUPPORTED";
    }
}

template <typename T>
consteval DataType dtype_of() noexcept
{
    using U = std::remove_const_t<T>;
    if constexpr (std::is_same_v<U, __nv_bfloat16>) {
        return DataType::kBf16;
    } else if constexpr (std::is_same_v<U, __half>) {
        return DataType::kFp16;
    } else if constexpr (std::is_same_v<U, std::int32_t>) {
        return DataType::kInt32;
    } else if constexpr (std::is_same_v<U, std::int8_t>) {
        return DataType::kInt8;
    } else if constexpr (std::is_same_v<U, float>) {
        return DataType::kFp32;
    } else {
        return DataType::kUnsupported;
    }
}

}
