#include <cassert>
#include <algorithm>
#include <format>

#include <cuda_bf16.h>

#include "ops/unified_kv_cache_update.h"
#include "ops/utils.h"
#include "cuda/cuda_utils.h"


namespace cllm::ops
{

constexpr uint32_t kMaxNumThreads = 256;

namespace
{

__global__ __launch_bounds__(kMaxNumThreads)
void bf16_unified_kv_cache_update_packedqkv(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ qkv,
    const int* __restrict__ slot_mapping,
    uint32_t q_size,
    uint32_t kv_size
)
{
    const int slot = slot_mapping[blockIdx.x];
    if (slot < 0) { // for CUDA Graph future
        return;
    }

    const uint32_t stride = q_size + 2 * kv_size;
    const uint64_t src_offset = static_cast<uint64_t>(blockIdx.x) * stride + q_size;
    const auto* k_u4 = reinterpret_cast<const uint4*>(qkv + src_offset);
    const auto* v_u4 = reinterpret_cast<const uint4*>(qkv + src_offset + kv_size);

    const uint64_t dst_offset = static_cast<uint64_t>(slot) * kv_size;
    auto* k_cache_u4 = reinterpret_cast<uint4*>(k_cache + dst_offset);
    auto* v_cache_u4 = reinterpret_cast<uint4*>(v_cache + dst_offset);

    const uint32_t kv_size_u4 = kv_size / kNumBf16sPerVector;
    for (uint32_t i = threadIdx.x; i < kv_size_u4; i += blockDim.x) {
        k_cache_u4[i] = k_u4[i];
        v_cache_u4[i] = v_u4[i];
    }
}

}

void unified_kv_cache_update(
    TensorRef<3> k_cache, // [num_slots, num_kv_heads, head_dim]
    TensorRef<3> v_cache, // [num_slots, num_kv_heads, head_dim]
    TensorRef<2> qkv, // [num_tokens, (num_q_heads + 2 * num_kv_heads) * head_dim]
    const int* slot_mapping,
    cudaStream_t stream
)
{
    assert(qkv && k_cache && v_cache && slot_mapping != nullptr);

    const uint32_t num_tokens = static_cast<uint32_t>(qkv.shape[0]);
    const uint32_t kv_size = static_cast<uint32_t>(k_cache.shape[1] * k_cache.shape[2]);
    const uint32_t q_size = static_cast<uint32_t>(qkv.shape[1]) - 2 * kv_size;
    if (kv_size % kNumBf16sPerVector != 0 || q_size % kNumBf16sPerVector != 0) {
        throw std::runtime_error(std::format(
            "unsupported: q_size={}, kv_size={}",
            q_size,
            kv_size
        ));
    }

    const uint32_t num_threads = std::clamp<uint32_t>(
        kv_size / kNumBf16sPerVector,
        1,
        kMaxNumThreads
    );
    bf16_unified_kv_cache_update_packedqkv<<<num_tokens, num_threads, 0, stream>>>(
        k_cache.data<__nv_bfloat16>(),
        v_cache.data<__nv_bfloat16>(),
        qkv.data<__nv_bfloat16>(),
        slot_mapping,
        q_size,
        kv_size
    );
    CUDA_CHECK(cudaGetLastError());
}

}
