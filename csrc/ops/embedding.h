#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

void embedding(
    const int* token_ids,
    TensorRef<2> embedding_table,
    TensorRef<2> output,
    cudaStream_t stream
);

}
