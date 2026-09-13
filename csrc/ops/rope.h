#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cllm::ops
{

void rope(
    TensorRef<2> qkv,
    TensorRef<3> rope_cache,
    const int* positions,
    int q_size,
    int k_size,
    int head_dim,
    /* int rotary_dim = head_dim */
    cudaStream_t stream
);

}
