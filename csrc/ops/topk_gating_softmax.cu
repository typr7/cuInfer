#include <cassert>

#include <math_constants.h>

#include "ops/topk_gating_softmax.h"
#include "ops/utils.h"
#include "cuda/cuda_utils.h"


namespace cuinfer::ops
{

namespace
{

// for Qwen3-30B-A3B
constexpr uint32_t kNumExperts = 128;
constexpr uint32_t kTopK = 8;
constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kNumThreads = kWarpsPerBlock * kNumThreadsPerWarp;

template <typename T, bool kRenormalize>
__global__ __launch_bounds__(kNumThreads)
void topk_gating_softmax_kernel(
    const T* __restrict__ logits,
    float* __restrict__ weights,
    int* __restrict__ indices,
    uint32_t num_tokens,
    uint32_t logits_stride,
    uint32_t weights_stride,
    uint32_t indices_stride,
    const int* num_tokens_device
)
{
    const uint32_t row = blockIdx.x * kWarpsPerBlock + threadIdx.x / kNumThreadsPerWarp;
    if (row >= num_tokens || (num_tokens_device != nullptr && row >= *num_tokens_device)) {
        return;
    }

    const uint32_t lane = threadIdx.x % kNumThreadsPerWarp;
    constexpr uint32_t kExpertsPerLane = kNumExperts / kNumThreadsPerWarp;
    float values[kExpertsPerLane];
    int experts[kExpertsPerLane];
    #pragma unroll
    for (uint32_t i = 0; i < kExpertsPerLane; i++) {
        experts[i] = lane + i * kNumThreadsPerWarp;
        values[i] = static_cast<float>(logits[row * logits_stride + experts[i]]);
    }

    float max_logit = -CUDART_INF_F;
    float denominator = 0.f;
    if constexpr (!kRenormalize) {
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; i++) {
            max_logit = fmaxf(max_logit, values[i]);
        }
        max_logit = warp_butterfly_reduce_max(max_logit);
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; i++) {
            denominator += __expf(values[i] - max_logit);
        }
        denominator = warp_butterfly_reduce_sum(denominator);
    }

    float selected_logit = -CUDART_INF_F;
    int selected_expert = -1;
    #pragma unroll
    for (uint32_t k = 0; k < kTopK; k++) {
        float value = values[0];
        int expert = experts[0];
        #pragma unroll
        for (uint32_t i = 1; i < kExpertsPerLane; i++) {
            if (values[i] > value || (values[i] == value && experts[i] < expert)) {
                value = values[i];
                expert = experts[i];
            }
        }
        const float max_value = warp_butterfly_reduce_max(value);
        expert = __reduce_min_sync(0xffffffff, value == max_value ? expert : kNumExperts);
        if (lane == k) {
            selected_logit = max_value;
            selected_expert = expert;
        }
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; i++) {
            if (experts[i] == expert) {
                values[i] = -CUDART_INF_F;
                experts[i] = kNumExperts;
            }
        }
    }

    if constexpr (kRenormalize) {
        max_logit = __shfl_sync(0xffffffff, selected_logit, 0);
    }
    const float weight = lane < kTopK ? __expf(selected_logit - max_logit) : 0.f;
    if constexpr (kRenormalize) {
        denominator = warp_butterfly_reduce_sum<kTopK>(weight);
    }
    if (lane < kTopK) {
        weights[row * weights_stride + lane] = weight * __frcp_rn(denominator);
        indices[row * indices_stride + lane] = selected_expert;
    }
}

template <typename T>
void launch_topk_gating_softmax(TensorRef<2> logits, TensorRef<2> weights, TensorRef<2> indices,
                         bool renormalize, cudaStream_t stream, const int* num_tokens_device)
{
    const uint32_t num_tokens = logits.shape[0];
    const uint32_t num_blocks = (num_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const T* input = logits.data<T>();
    float* output = weights.data<float>();
    int* ids = indices.data<int>();
    if (renormalize) {
        topk_gating_softmax_kernel<T, true><<<num_blocks, kNumThreads, 0, stream>>>(
            input, output, ids, num_tokens, logits.stride[0], weights.stride[0],
            indices.stride[0], num_tokens_device
        );
    } else {
        topk_gating_softmax_kernel<T, false><<<num_blocks, kNumThreads, 0, stream>>>(
            input, output, ids, num_tokens, logits.stride[0], weights.stride[0],
            indices.stride[0], num_tokens_device
        );
    }
}

}

void topk_gating_softmax(
    TensorRef<2> logits,
    TensorRef<2> weights,
    TensorRef<2> indices,
    bool renormalize,
    cudaStream_t stream,
    const int* num_tokens_device
)
{
    assert(logits && weights && indices);

    switch (logits.dtype) {
        case DataType::kBf16:
            launch_topk_gating_softmax<nv_bfloat16>(
                logits, weights, indices, renormalize, stream, num_tokens_device
            );
            break;
        case DataType::kFp32:
            launch_topk_gating_softmax<float>(
                logits, weights, indices, renormalize, stream, num_tokens_device
            );
            break;
        default:
            throw std::runtime_error("unsupported topk softmax dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

}
