#include <cassert>
#include <algorithm>

#include "ops/residual_add.h"

#include "ops/utils.h"
#include "cuda/cuda_utils.h"


namespace cuinfer::ops
{

namespace
{

union BF162x4
{
    uint4 vec;
    nv_bfloat162 bf162x4[4];
};

// b <- a + b, a != b
template <uint32_t kHiddenSize>
requires (kHiddenSize == 1024 || kHiddenSize == 2048)
__global__ __launch_bounds__(kHiddenSize / kNumBf16sPerVector)
void bf16_elementwise_add(const nv_bfloat16* __restrict__ a, nv_bfloat16* __restrict__ b)
{
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * kHiddenSize + threadIdx.x * kNumBf16sPerVector;
    auto* pb = reinterpret_cast<uint4*>(b + idx);

    BF162x4 vec_a;
    BF162x4 vec_b;
    vec_a.vec = *reinterpret_cast<const uint4*>(a + idx);
    vec_b.vec = *pb;

    vec_b.bf162x4[0] = __hadd2(vec_a.bf162x4[0], vec_b.bf162x4[0]);
    vec_b.bf162x4[1] = __hadd2(vec_a.bf162x4[1], vec_b.bf162x4[1]);
    vec_b.bf162x4[2] = __hadd2(vec_a.bf162x4[2], vec_b.bf162x4[2]);
    vec_b.bf162x4[3] = __hadd2(vec_a.bf162x4[3], vec_b.bf162x4[3]);

    *pb = vec_b.vec;
}

template <uint32_t kNumThreads>
__global__ __launch_bounds__(kNumThreads)
void bf16_elementwise_add(const nv_bfloat16* __restrict__ a, nv_bfloat16* __restrict__ b, size_t n)
{
    const size_t num_vec = n / kNumBf16sPerVector;
    const auto* pa = reinterpret_cast<const uint4*>(a);
    auto* pb = reinterpret_cast<uint4*>(b);

    const size_t idx = static_cast<size_t>(blockIdx.x) * kNumThreads + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * kNumThreads;

    BF162x4 vec_a;
    BF162x4 vec_b;
    for (size_t i = idx; i < num_vec; i += stride) {
        vec_a.vec = pa[i];
        vec_b.vec = pb[i];
        vec_b.bf162x4[0] = __hadd2(vec_a.bf162x4[0], vec_b.bf162x4[0]);
        vec_b.bf162x4[1] = __hadd2(vec_a.bf162x4[1], vec_b.bf162x4[1]);
        vec_b.bf162x4[2] = __hadd2(vec_a.bf162x4[2], vec_b.bf162x4[2]);
        vec_b.bf162x4[3] = __hadd2(vec_a.bf162x4[3], vec_b.bf162x4[3]);
        pb[i] = vec_b.vec;
    }

    const size_t tail = n - num_vec * kNumBf16sPerVector;
    if (blockIdx.x == 0 && threadIdx.x < tail) {
        const size_t i = num_vec * kNumBf16sPerVector + threadIdx.x;
        b[i] += a[i];
    }
}

}

void residual_add(TensorRef<2> hidden, TensorRef<2> residual, cudaStream_t stream)
{
    assert(hidden && residual);

    const auto [num_tokens, hidden_size] = hidden.shape;
    switch (hidden_size) {
        case 1024: {
            constexpr uint32_t kNumThreads = 1024 / kNumBf16sPerVector;
            bf16_elementwise_add<1024><<<num_tokens, kNumThreads, 0, stream>>>(
                static_cast<const nv_bfloat16*>(hidden.device_ptr),
                static_cast<nv_bfloat16*>(residual.device_ptr)
            );
            break;
        }
        case 2048: {
            constexpr uint32_t kNumThreads = 2048 / kNumBf16sPerVector;
            bf16_elementwise_add<2048><<<num_tokens, kNumThreads, 0, stream>>>(
                static_cast<const nv_bfloat16*>(hidden.device_ptr),
                static_cast<nv_bfloat16*>(residual.device_ptr)
            );
            break;
        }
        default: {
            constexpr uint32_t kNumThreads = 256;
            const size_t num_elements = static_cast<size_t>(num_tokens) * hidden_size;
            const size_t num_vec = num_elements / kNumBf16sPerVector;
            const size_t num_blocks =
                std::clamp<size_t>((num_vec + kNumThreads - 1) / kNumThreads, 1, 4096);
            bf16_elementwise_add<kNumThreads><<<num_blocks, kNumThreads, 0, stream>>>(
                static_cast<const nv_bfloat16*>(hidden.device_ptr),
                static_cast<nv_bfloat16*>(residual.device_ptr),
                num_elements
            );
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

}
