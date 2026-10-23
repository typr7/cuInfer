#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

void topk_gating_softmax(
    TensorRef<2> logits,
    TensorRef<2> weights,
    TensorRef<2> indices,
    bool renormalize,
    cudaStream_t stream,
    const int* num_tokens_device = nullptr
);

}
