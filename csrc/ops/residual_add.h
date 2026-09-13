#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cllm::ops
{

void residual_add(TensorRef<2> hidden, TensorRef<2> residual, cudaStream_t stream);

}
