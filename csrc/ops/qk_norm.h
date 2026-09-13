#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

void qk_norm(
    TensorRef<2> qkv,
    TensorRef<1> q_weights,
    TensorRef<1> k_weights,
    int q_size,
    int k_size,
    int head_dim,
    float eps,
    cudaStream_t stream
);

}