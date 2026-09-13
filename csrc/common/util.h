#pragma once

#include <cstddef>
#include <bit>


namespace cllm
{

template <std::size_t kAlignment>
constexpr std::size_t align_up(std::size_t offset)
{
    static_assert(std::has_single_bit(kAlignment));
    return (offset + kAlignment - 1) & ~(kAlignment - 1);
}

}
