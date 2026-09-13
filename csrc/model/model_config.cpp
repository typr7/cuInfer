#include <fstream>
#include <format>

#include <nlohmann/json.hpp>

#include "model/model_config.h"


namespace cuinfer
{

// function `from_json` is for parsing json to ModelConfig,
// code such as `json.get<RopeScaling>()` will need this.
void from_json(const nlohmann::json& j, RopeScaling& r)
{
    r.factor = j.at("factor").get<float>();
    r.low_freq_factor = j.at("low_freq_factor").get<float>();
    r.high_freq_factor = j.at("high_freq_factor").get<float>();
    r.original_max_position_embeddings = j.at("original_max_position_embeddings").get<int>();
}

void from_json(const nlohmann::json& j, ModelConfig& cfg)
{
    const auto arch = j.at("architectures").at(0).get<std::string>();
    if (arch == "Qwen3ForCausalLM") {
        cfg.arch = ModelArch::kQwen3;
        cfg.has_qk_norm = true;
    } else if (arch == "LlamaForCausalLM") {
        cfg.arch = ModelArch::kLlama;
        cfg.has_qk_norm = false;
    } else {
        cfg.arch = ModelArch::kUnsupported;
    }

    const auto dtype = j.at("torch_dtype").get<std::string>();
    if (dtype == "bfloat16") {
        cfg.dtype = DataType::kBf16;
    } else {
        cfg.dtype = DataType::kUnsupported;
    }

    cfg.max_model_len = j.at("max_position_embeddings").get<int>();
    cfg.vocab_size = j.at("vocab_size").get<int>();
    cfg.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    cfg.hidden_size = j.at("hidden_size").get<int>();
    cfg.intermediate_size = j.at("intermediate_size").get<int>();
    cfg.num_attention_heads = j.at("num_attention_heads").get<int>();
    cfg.num_kv_heads = j.at("num_key_value_heads").get<int>();
    cfg.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    cfg.rope_theta = j.at("rope_theta").get<float>();
    cfg.tie_word_embeddings = j.at("tie_word_embeddings").get<bool>();

    // head_dim field may not be set in config.json
    cfg.head_dim = j.contains("head_dim")
        ? j.at("head_dim").get<int>()
        : cfg.hidden_size / cfg.num_attention_heads;
    
    // eos_token_id
    const auto eos = j.at("eos_token_id");
    cfg.eos_token_ids = eos.is_array()
        ? eos.get<std::vector<int>>()
        : std::vector{eos.get<int>()};

    // rope_scaling
    const auto iter = j.find("rope_scaling");
    cfg.rope_scaling = (iter == j.end() || iter->is_null())
        ? std::nullopt
        : std::make_optional(iter->get<RopeScaling>());
}

ModelConfig ModelConfig::load(const std::filesystem::path& model_dir)
{
    using nlohmann::json;

    const auto config_path = model_dir / "config.json";
    const auto generation_config_path = model_dir / "generation_config.json";

    ModelConfig model_config;
    {
        std::ifstream file(config_path);

        if (!file) {
            throw std::runtime_error(std::format("cannot open `{}`", config_path.string()));
        }

        try {
            const auto j = json::parse(file);
            model_config = j.get<ModelConfig>();
        } catch (const json::exception& e) {
            throw std::runtime_error(std::format(
                "invalid model config `{}`: {}", config_path.string(),e.what()
            ));
        }

        if (model_config.arch == ModelArch::kUnsupported) {
            throw std::runtime_error(std::format(
                "unsupported model architecture in config `{}`", config_path.string()
            ));
        } else if (model_config.dtype == DataType::kUnsupported) {
            throw std::runtime_error(std::format(
                "unsupported data type in config `{}`", config_path.string()
            ));
        }
    }

    std::ifstream file(generation_config_path);
    if (file) {
        try {
            const auto j = json::parse(file);
            const auto eos_iter = j.find("eos_token_id");
            if (eos_iter != j.end() && !eos_iter->is_null()) {
                model_config.eos_token_ids = eos_iter->is_array()
                    ? eos_iter->get<std::vector<int>>()
                    : std::vector{eos_iter->get<int>()};
            }
        } catch (const json::exception& e) {
            throw std::runtime_error(std::format(
                "invalid generation config `{}`: {}", generation_config_path.string(), e.what()
            ));
        }
    }

    return model_config;
}
    
}
