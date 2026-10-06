#include <cassert>
#include <format>

#include "ops/qk_norm_rope_fused.h"
#include "cuda/cuda_utils.h"
#include "ops/utils.h"


namespace cuinfer::ops
{

namespace
{

constexpr uint32_t kNumThreads = 128;
constexpr uint32_t kNumElemsPerBlock = 512;

union Bit128
{
    uint4 vec;
    int64_t i64x2[2];
    nv_bfloat16 bf16x8[8];
};

union Bit64
{
    uint2 vec;
    nv_bfloat16 bf16x4[4];
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

__global__ __launch_bounds__(kNumThreads)
void bf16_qk_norm_rope_fused_packedqkv_perheadd128(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_weights,
    const __nv_bfloat16* __restrict__ k_weights,
    const float* __restrict__ cos_sin,
    const int* __restrict__ positions,
    uint32_t stride,
    uint32_t q_size,
    float eps,
    const int* num_tokens
)
{
    if (num_tokens != nullptr && blockIdx.x >= *num_tokens) {
        return;
    }

    constexpr uint32_t kHeadDim = 128;
    constexpr uint32_t kNumElemsPerThread = kNumElemsPerBlock / kNumThreads;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % kNumThreadsPerWarp;

    auto* head_u2 = reinterpret_cast<uint2*>(
        qkv
        + blockIdx.x * stride
        + blockIdx.y * kNumElemsPerBlock
        + tid * kNumElemsPerThread
    );

    Bit64 head{
        .vec = *head_u2
    };

    Bit64 weights{
        .vec = as<const uint2>(
            blockIdx.y * kNumElemsPerBlock < q_size
            ? &q_weights[lane_id * kNumElemsPerThread]
            : &k_weights[lane_id * kNumElemsPerThread]
        )
    };

    float sum0 = 0.f;
    float sum1 = 0.f;
    #pragma unroll
    for (uint32_t i = 0; i < kNumElemsPerThread; i += 2) {
        const float val0 = __bfloat162float(head.bf16x4[i]);
        const float val1 = __bfloat162float(head.bf16x4[i + 1]);
        sum0 = fmaf(val0, val0, sum0);
        sum1 = fmaf(val1, val1, sum1);
    }

    sum0 = warp_reduce_sum(sum0 + sum1);

    float factor = 0.f;
    if (lane_id == 0) {
        factor = rsqrtf(sum0 * (1.f / kHeadDim) + eps);
    }

    factor = __shfl_sync(0xffffffff, factor, 0);

    #pragma unroll
    for (uint32_t i = 0; i < kNumElemsPerThread; i++) {
        head.bf16x4[i] = __float2bfloat16(
            factor * __bfloat162float(head.bf16x4[i]) * __bfloat162float(weights.bf16x4[i])
        );
    }

    const auto pos = static_cast<uint32_t>(positions[blockIdx.x]);
    const auto* cos_sin_f4 = reinterpret_cast<const float4*>(cos_sin + pos * kHeadDim);

    const uint32_t group = lane_id >> 4;
    const uint32_t group_lane = lane_id & 0b1111;

    const float4 cs[2] = {
        cos_sin_f4[2 * group_lane],
        cos_sin_f4[2 * group_lane + 1]
    };
    const auto* cs_f = reinterpret_cast<const float*>(cs);
    const Bit64 peer{
        .vec = make_uint2(
            __shfl_xor_sync(0xffffffff, head.vec.x, 16),
            __shfl_xor_sync(0xffffffff, head.vec.y, 16)
        )
    };

    #pragma unroll
    for (uint32_t i = 0; i < kNumElemsPerThread; i++) {
        const float self_val = __bfloat162float(head.bf16x4[i]);
        const float peer_val = __bfloat162float(peer.bf16x4[i]);
        const float cos = cs_f[2 * i];
        const float sin = group == 0 ? cs_f[2 * i + 1] : -cs_f[2 * i + 1];
        head.bf16x4[i] = __float2bfloat16(self_val * cos - peer_val * sin);
    }

    *head_u2 = head.vec;
}

__global__ __launch_bounds__(128) // kHeadsPerBlock * kHeadDim / kNumBf16sPerVector = 8 * 128 / 8
void bf16_qk_norm_rope_fused_packedqkv_q2048k1024d128(
    nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ q_weights,
    const nv_bfloat16* __restrict__ k_weights,
    const float* __restrict__ cos_sin, // [max_position, rotary / 2, 2 (cos then sin)]
    const int* __restrict__ positions,
    uint32_t stride,
    float eps,
    const int* num_tokens
)
{
    if (num_tokens != nullptr && blockIdx.x >= *num_tokens) {
        return;
    }

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
    cudaStream_t stream,
    const int* num_tokens_device
)
{
    assert(qkv && q_weights && k_weights && rope_cache && positions != nullptr);

    const uint32_t num_tokens = static_cast<uint32_t>(qkv.shape[0]);
    const uint32_t stride = static_cast<uint32_t>(qkv.stride[0]);
    
    if (q_size % kNumElemsPerBlock == 0 && k_size % kNumElemsPerBlock == 0 && head_dim == 128) {
        const dim3 grid(num_tokens, (q_size + k_size) / kNumElemsPerBlock);
        bf16_qk_norm_rope_fused_packedqkv_perheadd128<<<grid, kNumThreads, 0, stream>>>(
            qkv.data<__nv_bfloat16>(),
            q_weights.data<const __nv_bfloat16>(),
            k_weights.data<const __nv_bfloat16>(),
            rope_cache.data<const float>(),
            positions,
            stride,
            q_size,
            eps,
            num_tokens_device
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
