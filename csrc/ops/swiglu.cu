#include <cassert>
#include <format>

#include "ops/swiglu.h"
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

// xy / (1 + e^(-x))
__device__ __forceinline__
nv_bfloat162 silu_and_mul_bf162(nv_bfloat162 x, nv_bfloat162 y)
{
    const float2 x_f2 = __bfloat1622float2(x);
    const float2 y_f2 = __bfloat1622float2(y);

    return __float22bfloat162_rn(make_float2(
        __fdividef(x_f2.x * y_f2.x, 1.f + __expf(-x_f2.x)),
        __fdividef(x_f2.y * y_f2.y, 1.f + __expf(-x_f2.y))
    ));
}

template <uint32_t kNumThreads>
__global__ __launch_bounds__(kNumThreads)
void bf16_swiglu_packedgu(nv_bfloat16* gate_up, uint32_t intermediate_size)
{
    const uint32_t stride = 2 * intermediate_size;
    const auto* gate_u4 = reinterpret_cast<const uint4*>(gate_up + blockIdx.x * stride);
    const auto* up_u4 =
        reinterpret_cast<const uint4*>(gate_up + intermediate_size + blockIdx.x * stride);
    auto* gated_u4 = reinterpret_cast<uint4*>(gate_up + blockIdx.x * stride);

    BF162x4 vec_x;
    BF162x4 vec_y;
    const uint32_t inter_vec = intermediate_size / kNumBf16sPerVector;
    for (uint32_t i = threadIdx.x; i < inter_vec; i += kNumThreads) {
        vec_x.vec = gate_u4[i];
        vec_y.vec = up_u4[i];
        vec_x.bf162x4[0] = silu_and_mul_bf162(vec_x.bf162x4[0], vec_y.bf162x4[0]);
        vec_x.bf162x4[1] = silu_and_mul_bf162(vec_x.bf162x4[1], vec_y.bf162x4[1]);
        vec_x.bf162x4[2] = silu_and_mul_bf162(vec_x.bf162x4[2], vec_y.bf162x4[2]);
        vec_x.bf162x4[3] = silu_and_mul_bf162(vec_x.bf162x4[3], vec_y.bf162x4[3]);
        gated_u4[i] = vec_x.vec;
    }
}

}

// gate_up: [gate | up]
void swiglu(TensorRef<2> gate_up, cudaStream_t stream)
{
    assert(gate_up);

    constexpr uint32_t kNumThreads = 128;

    const uint32_t num_tokens = gate_up.shape[0];
    const uint32_t inter_dim = gate_up.shape[1] / 2;

    if (inter_dim % kNumBf16sPerVector == 0) {
        bf16_swiglu_packedgu<kNumThreads><<<num_tokens, kNumThreads, 0, stream>>>(
            static_cast<nv_bfloat16*>(gate_up.device_ptr),
            inter_dim
        );
    } else {
        throw std::runtime_error(std::format("unsupported intermediate size: {}", inter_dim));
    }
    CUDA_CHECK(cudaGetLastError());
}

}
