#pragma once

#include <vector>
#include <optional>
#include <filesystem>

#include "tensor/data_type.h"


namespace cllm
{

enum class ModelArch
{
    kUnsupported,
    kLlama,  // Llama-3.2-1B
    kQwen3,  // Qwen3-0.6B
};

// Llama 3 rescales the low-frequency RoPE bands so the model generalizes past
// its original training length. Qwen3 does not use it.
struct RopeScaling
{
    float factor;
    float low_freq_factor;
    float high_freq_factor;
    int original_max_position_embeddings;
};

struct ModelConfig
{
    ModelArch arch;
    DataType dtype = DataType::kBf16;

    int max_model_len;
    int vocab_size;
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_kv_heads;

    // Read this from the Hugging Face config instead of deriving it from
    // hidden_size / num_attention_heads: Qwen3-0.6B has hidden_size 1024 and
    // 16 attention heads, but head_dim 128 rather than 64.
    int head_dim;

    float rms_norm_eps;
    float rope_theta;

    // Qwen3 applies an RMSNorm to Q and K before RoPE; Llama does not.
    bool has_qk_norm;

    // Both target models reuse the embedding matrix as the LM head.
    bool tie_word_embeddings;

    std::vector<int> eos_token_ids;

    std::optional<RopeScaling> rope_scaling;

    static ModelConfig load(const std::filesystem::path& path);
};

}
