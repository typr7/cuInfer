#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cllm::ops
{

void qk_norm_rope(
    TensorRef<2> qkv,
    TensorRef<1> q_weights,
    TensorRef<1> k_weights,
    TensorRef<3> rope_cache,
    const int* positions,
    int q_size,
    int k_size,
    int head_dim,
    /* int rotary_dim = head_dim */
    float eps,
    cudaStream_t stream
);

}
