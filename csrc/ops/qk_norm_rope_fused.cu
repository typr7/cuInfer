#include <cassert>
#include <format>

#include "ops/qk_norm_rope_fused.h"
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

__global__ __launch_bounds__(128) // kHeadsPerBlock * kHeadDim / kNumBf16sPerVector = 8 * 128 / 8
void bf16_qk_norm_rope_fused_packedqkv_q2048k1024d128(
    nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ q_weights,
    const nv_bfloat16* __restrict__ k_weights,
    const float* __restrict__ cos_sin, // [max_position, rotary / 2, 2 (cos then sin)]
    const int* __restrict__ positions,
    uint32_t stride,
    float eps
)
{
    // qkv: [Q (2048) | K (1024) | V] -> [16 heads | 8 heads | V]
    constexpr uint32_t kNumThreadsPerBlock = 128;
    constexpr uint32_t kHeadDim = 128;
    constexpr uint32_t kNumHeadsPerBlock = 8; // 8 * 128 = 1024
    constexpr uint32_t kNumThreadsPerHead = kNumThreadsPerBlock / kNumHeadsPerBlock;

    const uint32_t tid = threadIdx.x;
    const uint32_t head_idx = tid / kNumThreadsPerHead;
    
    auto* head_u4 = reinterpret_cast<uint4*>(
        qkv
        + blockIdx.x * stride // qkv line
        + blockIdx.y * kNumHeadsPerBlock * kHeadDim // block
        + head_idx * kHeadDim // head
    );

    const auto* weights_u4 = reinterpret_cast<const uint4*>(
        blockIdx.y < 2 ? q_weights : k_weights
    );

    const auto pos = static_cast<uint32_t>(positions[blockIdx.x]);
    const auto* cos_sin_f4 = reinterpret_cast<const float4*>(cos_sin + pos * kHeadDim);

    const uint32_t head_lane = tid % kNumThreadsPerHead;
    const uint32_t group = head_lane >> 3;
    const uint32_t group_lane = head_lane & 0b111;
    
    BF16x8 head{.vec = head_u4[head_lane]};
    BF16x8 weights{.vec = weights_u4[head_lane]};
    const float4 cs[2] = {
        cos_sin_f4[4 * group_lane + group],
        cos_sin_f4[4 * group_lane + 2 + group]
    };

    float sum0 = 0.f;
    float sum1 = 0.f;
    #pragma unroll
    for (uint32_t i = 0; i < kNumBf16sPerVector; i += 2) {
        const float val0 = __bfloat162float(head.bf16x8[i]);
        const float val1 = __bfloat162float(head.bf16x8[i + 1]);
        sum0 = fmaf(val0, val0, sum0);
        sum1 = fmaf(val1, val1, sum1);
    }
    const float sum_val = warp_reduce_sum<kNumThreadsPerHead>(sum0 + sum1);
    
    float factor = head_lane == 0 ? rsqrtf(sum_val * (1.f / kHeadDim) + eps) : 0.f;
    factor = __shfl_sync(0xffffffff, factor, 0, kNumThreadsPerHead);

    #pragma unroll
    for (uint32_t i = 0; i < kNumBf16sPerVector; i++) {
        head.bf16x8[i] = __float2bfloat16(
            factor
            * __bfloat162float(weights.bf16x8[i])
            * __bfloat162float(head.bf16x8[i])
        );
    }
    
    // [self0, self1, peer0, peer1]
    Bit128 b128 {
        .vec = make_uint4(
            group == 0 ? head.vec.x : head.vec.y,
            group == 0 ? head.vec.z : head.vec.w,
            group == 0 ? head.vec.y : head.vec.x,
            group == 0 ? head.vec.w : head.vec.z
        )
    };
    b128.i64x2[1] = __shfl_xor_sync(0xffffffff, b128.i64x2[1], 8);

    const auto* cs_f = reinterpret_cast<const float*>(cs);

    float self_out[4];
    float peer_out[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        const float self_val = __bfloat162float(b128.bf16x8[i]);
        const float peer_val = __bfloat162float(b128.bf16x8[i + 4]);
        const float cos = cs_f[2 * i];
        const float sin = (group == 0 ? cs_f[2 * i + 1] : -cs_f[2 * i + 1]);
        self_out[i] = self_val * cos - peer_val * sin;
        peer_out[i] = peer_val * cos + self_val * sin;
    }
    b128.vec = make_uint4(
        pack_float2(self_out[0], self_out[1]),
        pack_float2(self_out[2], self_out[3]),
        pack_float2(peer_out[0], peer_out[1]),
        pack_float2(peer_out[2], peer_out[3])
    );
    b128.i64x2[1] = __shfl_xor_sync(0xffffffff, b128.i64x2[1], 8);


    if (group == 1) {
        swap(b128.i64x2[0], b128.i64x2[1]);
    }
    swap(b128.vec.y, b128.vec.z);
    head_u4[head_lane] = b128.vec;
}

}

void qk_norm_rope(
    TensorRef<2> qkv,
    TensorRef<1> q_weights,
    TensorRef<1> k_weights,
    TensorRef<3> rope_cache,
    const int* positions,
    int q_size,
    int k_size,
    int head_dim,
    float eps,
    cudaStream_t stream
)
{
    assert(qkv && q_weights && k_weights && rope_cache && positions != nullptr);

    const uint32_t num_tokens = static_cast<uint32_t>(qkv.shape[0]);
    const uint32_t stride = static_cast<uint32_t>(qkv.stride[0]);
    
    if (q_size == 2048 && k_size == 1024 && head_dim == 128) {
        constexpr uint32_t kNumThreads = 128;
        constexpr uint32_t kNumBlocksPerToken = 3;

        const dim3 grid_size(num_tokens, kNumBlocksPerToken);
        bf16_qk_norm_rope_fused_packedqkv_q2048k1024d128<<<grid_size, kNumThreads, 0, stream>>>(
            static_cast<nv_bfloat16*>(qkv.device_ptr),
            static_cast<const nv_bfloat16*>(q_weights.device_ptr),
            static_cast<const nv_bfloat16*>(k_weights.device_ptr),
            static_cast<const float*>(rope_cache.device_ptr),
            positions,
            static_cast<uint32_t>(stride),
            eps
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
