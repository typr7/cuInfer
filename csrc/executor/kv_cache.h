#pragma once

#include "model/model_config.h"
#include "cuda/cuda_device_buffer.h"
#include "tensor/data_type.h"
#include "tensor/tensor.h"


namespace cuinfer
{

struct KVCacheView
{
    void* data = nullptr;
    DataType dtype = DataType::kUnsupported;

    int num_layers = 0;
    int num_blocks = 0;
    int block_size = 0;
    int num_kv_heads = 0;
    int head_dim = 0;

    Tensor<3> k(int layer) const noexcept { return kv(layer, 0); }
    Tensor<3> v(int layer) const noexcept { return kv(layer, 1); }

private:
    Tensor<3> kv(int layer, int which) const noexcept;
};

struct KVCache
{
    CudaDeviceBuffer data;

    KVCacheView view;

    static KVCache create(const ModelConfig& config, int num_blocks, int block_size);

    static std::size_t block_bytes(const ModelConfig& config, int block_size) noexcept;
};


}
