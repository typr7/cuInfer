#include <cassert>

#include <cublas_v2.h>

#include "ops/projection.h"
#include "cuda/cuda_utils.h"


namespace cllm::ops
{

// [M, K], [N, K], [M, N]
void projection(
    TensorRef<2> input,   // [M, K]
    TensorRef<2> weights, // [N, K]
    TensorRef<2> output,  // [M, N]
    cublasHandle_t handle
)
{
    constexpr float kAlphaOne = 1.f;
    constexpr float kBetaZero = 0.f;

    assert(input && weights && output);
    assert(input.shape[0] == output.shape[0]);
    assert(input.shape[1] == weights.shape[1]);
    assert(weights.shape[0] == output.shape[1]);

    const int num_tokens = input.shape[0];
    const int output_size = weights.shape[0];
    const int input_size = input.shape[1];

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        output_size, num_tokens, input_size,
        &kAlphaOne,
        weights.device_ptr, CUDA_R_16BF, weights.stride[0],
        input.device_ptr, CUDA_R_16BF, input.stride[0],
        &kBetaZero,
        output.device_ptr, CUDA_R_16BF, output.stride[0],
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

}
