#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cllm::ops
{

void swiglu(TensorRef<2> gate_up, cudaStream_t stream);

}
