#pragma once

#include <vector>
#include <filesystem>

#include "cuda/cuda_device_buffer.h"
#include "tensor/tensor.h"
#include "model/model_config.h"


namespace cuinfer
{

struct LayerWeights
{
    Tensor<1> input_layernorm;
    Tensor<2> qkv_proj;
    Tensor<1> q_norm;
    Tensor<1> k_norm;
    Tensor<2> o_proj;
    Tensor<1> post_attn_layernorm;
    Tensor<2> gate_up_proj;
    Tensor<2> down_proj;
};

struct ModelWeights
{
    Tensor<2> embed_tokens;
    Tensor<1> norm;
    // for tie_word_embeddings == true, it points same buffer as embed_tokens
    Tensor<2> lm_head;
    std::vector<LayerWeights> layers;
    CudaDeviceBuffer data;

    static ModelWeights load_from_safetensors(
        const std::filesystem::path& path,
        const ModelConfig& config
    );
};

}
