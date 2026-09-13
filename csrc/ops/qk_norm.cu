#include <cassert>
#include <format>

#include <cuda_bf16.h>

#include "ops/qk_norm.h"
#include "cuda/cuda_utils.h"
#include "ops/utils.h"


namespace cuinfer::ops
{

namespace
{

__global__ __launch_bounds__(128) // kHeadsPerBlock * kHeadDim / kNumBf16sPerVector = 8 * 128 / 8
void bf16_qk_norm_packedqkv_q2048k1024d128(
    nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ q_weights,
    const nv_bfloat16* __restrict__ k_weights,
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

    const uint32_t head_lane = tid % kNumThreadsPerHead;

    BF16x8 head;
    BF16x8 weights;
    head.vec = head_u4[head_lane];
    weights.vec = weights_u4[head_lane];

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
            * __bfloat162float(head.bf16x8[i])
            * __bfloat162float(weights.bf16x8[i])
        );
    }

    head_u4[head_lane] = head.vec;
}

}

void qk_norm(
    TensorRef<2> qkv,
    TensorRef<1> q_weights,
    TensorRef<1> k_weights,
    int q_size,
    int k_size,
    int head_dim,
    float eps,
    cudaStream_t stream
)
{
    assert(qkv);
    assert(q_weights);
    assert(k_weights);
    assert(q_weights.shape[0] == head_dim);
    assert(k_weights.shape[0] == head_dim);
    assert(qkv.dtype == q_weights.dtype);
    assert(qkv.dtype == k_weights.dtype);

    const uint32_t num_tokens = static_cast<uint32_t>(qkv.shape[0]);
    const uint32_t stride = static_cast<uint32_t>(qkv.stride[0]);

    if (q_size == 2048 && k_size == 1024 && head_dim == 128) {
        constexpr uint32_t kNumThreads = 128;
        constexpr uint32_t kNumBlocksPerToken = 3;

        const dim3 grid_size(num_tokens, kNumBlocksPerToken);
        bf16_qk_norm_packedqkv_q2048k1024d128<<<grid_size, kNumThreads, 0, stream>>>(
            static_cast<nv_bfloat16*>(qkv.device_ptr),
            static_cast<const nv_bfloat16*>(q_weights.device_ptr),
            static_cast<const nv_bfloat16*>(k_weights.device_ptr),
            static_cast<uint32_t>(stride),
            eps
        );
    } else {
        throw std::runtime_error(std::format(
            "unsupported qk norm size: q_size={}, k_size={}, head_dim={}",
            q_size,
            k_size,
            head_dim
        ));
    }
    CUDA_CHECK(cudaGetLastError());
}

}