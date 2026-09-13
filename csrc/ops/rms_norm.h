#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

void rms_norm(
    TensorRef<2> input,
    TensorRef<1> weights,
    TensorRef<2> output,
    float eps,
    cudaStream_t stream
);

}
