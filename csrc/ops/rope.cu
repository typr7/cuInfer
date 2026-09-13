#include <cassert>
#include <cstdint>
#include <format>

#include "ops/rope.h"
#include "cuda/cuda_utils.h"
#include "ops/utils.h"


namespace cllm::ops
{

namespace
{

union Bit128
{
    uint4 vec;
    int64_t i64x2[2];
    nv_bfloat16 bf16x8[8];
};

__device__ __forceinline__
void swap(int64_t& a, int64_t& b)
{
    const auto tmp = a;
    a = b;
    b = tmp;
}

__device__ __forceinline__
void swap(uint32_t& a, uint32_t& b)
{
    const auto tmp = a;
    a = b;
    b = tmp;
}

template <uint32_t kHeadDim, uint32_t kNumHeads>
__global__ __launch_bounds__(128)
void bf16_rope_packedqkv(
    nv_bfloat16* __restrict__ qkv,
    const float* __restrict__ cos_sin,
    const int* __restrict__ positions,
    uint32_t stride
)
{
    constexpr uint32_t kNumThreadsPerBlock = 128;
    constexpr uint32_t kNumThreadsPerHead = kHeadDim / kNumBf16sPerVector;
    constexpr uint32_t kNumHeadsPerBlock = kNumThreadsPerBlock / kNumThreadsPerHead;
    constexpr uint32_t kHalfHeadThreads = kNumThreadsPerHead / 2;

    static_assert(kHeadDim == 64 || kHeadDim == 128);

    const uint32_t tid = threadIdx.x;
    const uint32_t head_idx = blockIdx.y * kNumHeadsPerBlock
        + tid / kNumThreadsPerHead;
    if (head_idx >= kNumHeads) {
        return;
    }

    const uint32_t head_lane = tid % kNumThreadsPerHead;
    const uint32_t group = head_lane / kHalfHeadThreads;
    const uint32_t group_lane = head_lane % kHalfHeadThreads;

    auto* head_u4 = reinterpret_cast<uint4*>(
        qkv
        + blockIdx.x * stride
        + head_idx * kHeadDim
    );

    const uint32_t pos = static_cast<uint32_t>(positions[blockIdx.x]);
    const auto* cos_sin_f4 = reinterpret_cast<const float4*>(
        cos_sin + pos * kHeadDim
    );
    const float4 cs[2] = {
        cos_sin_f4[4 * group_lane + group],
        cos_sin_f4[4 * group_lane + 2 + group]
    };

    BF16x8 head{.vec = head_u4[head_lane]};
    Bit128 b128{
        .vec = make_uint4(
            group == 0 ? head.vec.x : head.vec.y,
            group == 0 ? head.vec.z : head.vec.w,
            group == 0 ? head.vec.y : head.vec.x,
            group == 0 ? head.vec.w : head.vec.z
        )
    };
    b128.i64x2[1] = __shfl_xor_sync(0xffffffff, b128.i64x2[1], kHalfHeadThreads);

    const auto* cs_f = reinterpret_cast<const float*>(cs);
    float self_outputs[4];
    float peer_outputs[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        const float self_val = __bfloat162float(b128.bf16x8[i]);
        const float peer_val = __bfloat162float(b128.bf16x8[i + 4]);
        const float cos = cs_f[2 * i];
        const float sin = group == 0 ? cs_f[2 * i + 1] : -cs_f[2 * i + 1];
        self_outputs[i] = self_val * cos - peer_val * sin;
        peer_outputs[i] = peer_val * cos + self_val * sin;
    }

    b128.vec = make_uint4(
        pack_float2(self_outputs[0], self_outputs[1]),
        pack_float2(self_outputs[2], self_outputs[3]),
        pack_float2(peer_outputs[0], peer_outputs[1]),
        pack_float2(peer_outputs[2], peer_outputs[3])
    );
    b128.i64x2[1] = __shfl_xor_sync(0xffffffff, b128.i64x2[1], kHalfHeadThreads);

    if (group == 1) {
        swap(b128.i64x2[0], b128.i64x2[1]);
    }
    swap(b128.vec.y, b128.vec.z);
    head_u4[head_lane] = b128.vec;
}

}

void rope(
    TensorRef<2> qkv,
    TensorRef<3> rope_cache,
    const int* positions,
    int q_size,
    int k_size,
    int head_dim,
    cudaStream_t stream
)
{
    assert(qkv && rope_cache && positions != nullptr);

    const uint32_t num_tokens = static_cast<uint32_t>(qkv.shape[0]);
    const uint32_t stride = static_cast<uint32_t>(qkv.stride[0]);

    constexpr uint32_t kNumThreads = 128;
    if (q_size == 2048 && k_size == 1024 && head_dim == 128) {
        constexpr uint32_t kNumHeads = 24;
        constexpr uint32_t kNumHeadsPerBlock = 8;
        const dim3 grid_size(num_tokens, kNumHeads / kNumHeadsPerBlock);
        bf16_rope_packedqkv<128, kNumHeads><<<grid_size, kNumThreads, 0, stream>>>(
            static_cast<nv_bfloat16*>(qkv.device_ptr),
            static_cast<const float*>(rope_cache.device_ptr),
            positions,
            stride
        );
    } else if (q_size == 2048 && k_size == 512 && head_dim == 64) {
        constexpr uint32_t kNumHeads = 40;
        constexpr uint32_t kNumHeadsPerBlock = 16;
        constexpr uint32_t kNumBlocksPerToken =
            (kNumHeads + kNumHeadsPerBlock - 1) / kNumHeadsPerBlock;
        const dim3 grid_size(num_tokens, kNumBlocksPerToken);
        bf16_rope_packedqkv<64, kNumHeads><<<grid_size, kNumThreads, 0, stream>>>(
            static_cast<nv_bfloat16*>(qkv.device_ptr),
            static_cast<const float*>(rope_cache.device_ptr),
            positions,
            stride
        );
    } else {
        throw std::runtime_error(std::format(
            "unsupported size: q_size={}, k_size={}, head_dim={}",
            q_size,
            k_size,
            head_dim
        ));
    }
    CUDA_CHECK(cudaGetLastError());
}

}
