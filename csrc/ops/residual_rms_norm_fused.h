#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

// residual add: hidden + residual -> residual
// rms norm: residual -> hidden
void residual_rms_norm_fused(
    TensorRef<2> hidden,
    TensorRef<2> residual,
    TensorRef<1> norm_weights,
    float eps,
    cudaStream_t stream,
    const int* num_tokens_device = nullptr
);

}