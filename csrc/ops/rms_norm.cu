#include <cassert>
#include <format>

#include <cuda_bf16.h>

#include "ops/rms_norm.h"
#include "tensor/tensor.h"
#include "cuda/cuda_utils.h"
#include "ops/utils.h"


namespace cuinfer::ops
{

namespace
{

template <uint32_t kHiddenSize, uint32_t kNumWarps>
__global__ __launch_bounds__(kHiddenSize / kNumBf16sPerVector)
void bf16_rms_norm(
    const nv_bfloat16* input,
    const nv_bfloat16* weights,
    nv_bfloat16* output,
    float eps
)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid & (kNumThreadsPerWarp - 1);
    const uint32_t warp_id = tid / kNumThreadsPerWarp;
    const auto* input_u4 = reinterpret_cast<const uint4*>(input + blockIdx.x * kHiddenSize);
    const auto* weights_u4 = reinterpret_cast<const uint4*>(weights);
    auto* output_u4 = reinterpret_cast<uint4*>(output + blockIdx.x * kHiddenSize);

    extern __shared__ float smem[];
    alignas(16) nv_bfloat16 reg[kNumBf16sPerVector * 2];
    auto* reg_u4 = reinterpret_cast<uint4*>(reg);

    reg_u4[0] = input_u4[tid];
    reg_u4[1] = weights_u4[tid];
    
    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int i = 0; i < kNumBf16sPerVector; i += 2) {
        const float val0 = __bfloat162float(reg[i]);
        const float val1 = __bfloat162float(reg[i + 1]);
        sum0 = fmaf(val0, val0, sum0);
        sum1 = fmaf(val1, val1, sum1);
    }
    float sum_val = warp_reduce_sum(sum0 + sum1);

    if (lane_id == 0) {
        smem[warp_id] = sum_val;
    }
    __syncthreads();
    
    sum_val = lane_id < kNumWarps ? smem[lane_id] : 0.f;
    sum_val = warp_reduce_sum<kNumWarps>(sum_val);
    float factor = lane_id == 0 ? rsqrtf(sum_val * (1.f / kHiddenSize) + eps) : 0.f;
    factor = __shfl_sync(0xffffffff, factor, 0);

    #pragma unroll
    for (int i = 0; i < kNumBf16sPerVector; i++) {
        const float val
            = factor * __bfloat162float(reg[i]) * __bfloat162float(reg[i + kNumBf16sPerVector]);
        reg[i] = __float2bfloat16(val);
    }
    output_u4[tid] = reg_u4[0];
}

template <uint32_t kHiddenSize>
void launch_kernel(
    TensorRef<2> input,
    TensorRef<1> weights,
    TensorRef<2> output,
    float eps,
    cudaStream_t stream
)
{
    constexpr uint32_t kNumThreads = kHiddenSize / kNumBf16sPerVector;
    constexpr uint32_t kNumWarps = kNumThreads / kNumThreadsPerWarp;
    constexpr uint32_t kSmemByteSize = kNumWarps * sizeof(float);

    static_assert(kHiddenSize % kNumBf16sPerVector == 0);
    static_assert(kNumThreads % kNumThreadsPerWarp == 0);

    const uint32_t num_tokens = static_cast<uint32_t>(input.shape[0]);
    bf16_rms_norm<kHiddenSize, kNumWarps><<<num_tokens, kNumThreads, kSmemByteSize, stream>>>(
        static_cast<const nv_bfloat16*>(input.device_ptr),
        static_cast<const nv_bfloat16*>(weights.device_ptr),
        static_cast<nv_bfloat16*>(output.device_ptr),
        eps
    );
    CUDA_CHECK(cudaGetLastError());
}

}

void rms_norm(
    TensorRef<2> input,
    TensorRef<1> weights,
    TensorRef<2> output,
    float eps,
    cudaStream_t stream
)
{
    assert(input && weights && output);

    const int hidden_size = input.shape[1];
    switch (hidden_size) {
        case 1024: {
            launch_kernel<1024>(input, weights, output, eps, stream);
            break;
        }

        case 2048: {
            launch_kernel<2048>(input, weights, output, eps, stream);
            break;
        }

        default:
            throw std::runtime_error(std::format("unsupported hidden size: {}", hidden_size));
    }
}

}
