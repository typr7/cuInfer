#pragma once

#include "cuda/cuda_device_buffer.h"
#include "model/model_config.h"
#include "tensor/tensor.h"


namespace cllm
{

struct RopeCache
{
    CudaDeviceBuffer data;

    // [max_position, rotary_dim / 2, 2 (cos then sin)]
    Tensor<3> view;

    static RopeCache create(const ModelConfig& config);
};

}