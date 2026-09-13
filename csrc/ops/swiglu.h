#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

void swiglu(TensorRef<2> gate_up, cudaStream_t stream);

}
