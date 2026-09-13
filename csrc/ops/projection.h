#pragma once

#include <cublas_v2.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

void projection(
    TensorRef<2> input,
    TensorRef<2> weights,
    TensorRef<2> output,
    cublasHandle_t handle
);

}
