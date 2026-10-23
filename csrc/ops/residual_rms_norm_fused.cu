#include <cassert>
#include <cstdint>
#include <format>
#include <algorithm>

#include "residual_rms_norm_fused.h"
#include "ops/utils.h"
#include "cuda/cuda_utils.h"


namespace cuinfer::ops
{

namespace
{

union Bit128
{
    uint4 vec;
    __nv_bfloat162 bf162x4[4];
    __nv_bfloat16 bf16x8[8];
};

// residual add: hidden + residual -> residual
// rms norm: residual -> hidden
template <uint32_t kHiddenSize, uint32_t kNumWarps>
__global__ __launch_bounds__(kNumWarps * kNumThreadsPerWarp)
void bf16_residual_rms_norm_fused(
    __nv_bfloat16* __restrict__ hidden,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weights,
    float eps,
    const int* num_tokens_device
)
{
    if (num_tokens_device != nullptr && blockIdx.x >= *num_tokens_device) {
        return;
    }

    constexpr uint32_t kHiddenSizeU4 = kHiddenSize / kNumBf16sPerVector;
    constexpr uint32_t kNumThreads = kNumWarps * kNumThreadsPerWarp;
    constexpr uint32_t kNumU4sPerThread = cdiv(kHiddenSizeU4, kNumThreads);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % kNumThreadsPerWarp;
    const uint32_t warp_id = tid / kNumThreadsPerWarp;

    __shared__ float warp_sum_vals[kNumWarps];

    uint4 reg[kNumU4sPerThread];

    auto* hidden_row_u4 = reinterpret_cast<uint4*>(hidden + blockIdx.x * kHiddenSize);
    auto* residual_row_u4 = reinterpret_cast<uint4*>(residual + blockIdx.x * kHiddenSize);

    float local_sum0 = 0.f;
    float local_sum1 = 0.f;

    for (uint32_t i = tid, cnt = 0; i < kHiddenSizeU4; i += kNumThreads, cnt++) {
        Bit128 r { .vec = hidden_row_u4[i] };

        Bit128 t {
            .vec = residual_row_u4[i]
        };

        r.bf162x4[0] = __hadd2(r.bf162x4[0], t.bf162x4[0]);
        r.bf162x4[1] = __hadd2(r.bf162x4[1], t.bf162x4[1]);
        r.bf162x4[2] = __hadd2(r.bf162x4[2], t.bf162x4[2]);
        r.bf162x4[3] = __hadd2(r.bf162x4[3], t.bf162x4[3]);

        residual_row_u4[i] = r.vec;
        reg[cnt] = r.vec;

        for (uint32_t k = 0; k < 4; k++) {
            const float2 val = __bfloat1622float2(r.bf162x4[k]);
            local_sum0 = fmaf(val.x, val.x, local_sum0);
            local_sum1 = fmaf(val.y, val.y, local_sum1);
        }
    }

    local_sum0 = warp_reduce_sum(local_sum0 + local_sum1);

    if (lane_id == 0) {
        warp_sum_vals[warp_id] = local_sum0;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum0 = lane_id < kNumWarps ? warp_sum_vals[lane_id] : 0.f;
        local_sum0 = warp_reduce_sum(local_sum0);
        if (lane_id == 0) {
            warp_sum_vals[0] = rsqrtf(fmaf(local_sum0, 1.f / kHiddenSize, eps));
        }
    }
    __syncthreads();

    const float factor = warp_sum_vals[0];

    for (uint32_t i = tid, cnt = 0; i < kHiddenSizeU4; i += kNumThreads, cnt++) {
        Bit128 weights {
            .vec = as<const uint4>(norm_weights + i * kNumBf16sPerVector)
        };
        Bit128 r { .vec = reg[cnt] };
        for (uint32_t k = 0; k < 4; k++) {
            const float2 f2 = __bfloat1622float2(__hmul2(r.bf162x4[k], weights.bf162x4[k]));
            r.bf16x8[2 * k] = __float2bfloat16(factor * f2.x);
            r.bf16x8[2 * k + 1] = __float2bfloat16(factor * f2.y);
        }
        hidden_row_u4[i] = r.vec;
    }
}

template <uint32_t kHiddenSize>
void launch_residual_rms_norm_fused(
    TensorRef<2> hidden,
    TensorRef<2> residual,
    TensorRef<1> norm_weights,
    float eps,
    cudaStream_t stream,
    const int* num_tokens_device
)
{
    static_assert(kHiddenSize % kNumBf16sPerVector == 0);

    constexpr uint32_t kNumThreads = std::clamp(kHiddenSize / kNumBf16sPerVector, 1u, 256u);
    constexpr uint32_t kNumWarps = cdiv(kNumThreads, kNumThreadsPerWarp);

    const uint32_t num_rows = hidden.shape[0];

    const dim3 grid_size(num_rows);
    const dim3 block_size(kNumThreads);

    constexpr auto kernel = bf16_residual_rms_norm_fused<kHiddenSize, kNumWarps>;
    kernel<<<grid_size, block_size, 0, stream>>>(
        hidden.data<__nv_bfloat16>(),
        residual.data<__nv_bfloat16>(),
        norm_weights.data<const __nv_bfloat16>(),
        eps,
        num_tokens_device
    );
    CUDA_CHECK(cudaGetLastError());
}

}

void residual_rms_norm_fused(
    TensorRef<2> hidden,
    TensorRef<2> residual,
    TensorRef<1> norm_weights,
    float eps,
    cudaStream_t stream,
    const int* num_tokens_device
)
{
    assert(hidden && residual && norm_weights);

    const int hidden_size = hidden.shape[1];
    switch (hidden_size) {
        case 1024:
            launch_residual_rms_norm_fused<1024>(
                hidden,
                residual,
                norm_weights,
                eps,
                stream,
                num_tokens_device
            );
            break;
        
        case 2048:
            launch_residual_rms_norm_fused<2048>(
                hidden,
                residual,
                norm_weights,
                eps,
                stream,
                num_tokens_device
            );
            break;
        
        default:
            throw std::runtime_error(std::format("unsupported hidden size: {}", hidden_size));
    }
}

}