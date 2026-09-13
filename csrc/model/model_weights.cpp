#include <algorithm>
#include <numeric>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cassert>

#include <nlohmann/json.hpp>

#include "model/model_weights.h"
#include "common/util.h"


namespace cllm
{

namespace
{

constexpr std::size_t kMaxHeaderSize = 100'000'000;
constexpr std::size_t kWeightAlignment = 256;
constexpr std::size_t kUploadChunkSize = 16ULL * 1024 * 1024; // 16MiB

struct CopyPlan
{
    std::string tensor_name;
    std::size_t src_offset;
    std::size_t dst_offset;
    std::size_t byte_size;
};

std::pair<std::size_t, std::size_t> check_tensor_spec(
    const nlohmann::json& header,
    std::string_view tensor_name,
    DataType expected_dtype,
    std::vector<int> expected_shape
)
{
    using nlohmann::json;

    auto shape2str = [](const std::vector<int>& shape) {
        std::string str = "[";
        for (int i = 0; i < shape.size(); i++) {
            if (i != 0) {
                str += ", ";
            }
            str += std::to_string(shape[i]);
        }
        return str + ']';
    };

    const json& tensor_spec = header.at(tensor_name);
        
    const auto tensor_dtype = tensor_spec.at("dtype").get_ref<const json::string_t&>();
    if (tensor_dtype != dtype_string(expected_dtype)) {
        throw std::runtime_error(std::format(
            "mismatched data type between `{}` in safetensors header ({}) and config.json ({})",
            tensor_name, tensor_dtype, dtype_string(expected_dtype)
        ));
    }

    const auto tensor_shape = tensor_spec.at("shape").get<std::vector<int>>();
    if (tensor_shape != expected_shape) {
        throw std::runtime_error(std::format(
            "mismatched shape between `{}` in safetensors header ({}) and config.json ({})",
            tensor_name, shape2str(tensor_shape), shape2str(expected_shape)
        ));
    }

    const auto data_offsets = tensor_spec.at("data_offsets").get<std::vector<std::size_t>>();
    assert(expected_shape.size() != 0);
    std::size_t tensor_size = std::accumulate(
        expected_shape.begin(),
        expected_shape.end(),
        dtype_byte_size(expected_dtype),
        std::multiplies<std::size_t>{}
    );
    if (
        data_offsets.size() != 2
        || data_offsets[1] < data_offsets[0]
        || (data_offsets[1] - data_offsets[0]) != tensor_size
    ) {
        throw std::runtime_error(std::format(
            "invalid data_offsets for `{}` in safetensors header", tensor_name
        ));
    }

    return {data_offsets[0], tensor_size};
}

std::vector<CopyPlan> build_plan(
    const nlohmann::json& header,
    const ModelConfig& config
)
{
    using nlohmann::json;

    const int vocab_size = config.vocab_size;
    const int hidden_size = config.hidden_size;
    const int intermediate_size = config.intermediate_size;
    const int head_dim = config.head_dim;
    const int q_size = config.num_attention_heads * head_dim;
    const int kv_size = config.num_kv_heads * head_dim;

    std::vector<CopyPlan> plan;

    std::size_t dst_offset = 0;
    
    auto check_then_add = [&](std::string name, std::vector<int> shape, bool align) {
        auto [src_offset, byte_size] = check_tensor_spec(
            header, name, config.dtype, std::move(shape)
        );

        if (align) {
            // this is for cuda kernel performance
            dst_offset = align_up<kWeightAlignment>(dst_offset);
        }

        plan.push_back(CopyPlan{
            .tensor_name = std::move(name),
            .src_offset = src_offset,
            .dst_offset = dst_offset,
            .byte_size = byte_size
        });

        dst_offset += byte_size;
    };

    check_then_add("model.embed_tokens.weight", {vocab_size, hidden_size}, true);

    for (int i = 0; i < config.num_hidden_layers; i++) {
        const std::string layer = std::format("model.layers.{}", i);

        check_then_add(layer + ".input_layernorm.weight", {hidden_size}, true);

        check_then_add(layer + ".self_attn.q_proj.weight", {q_size, hidden_size}, true);
        check_then_add(layer + ".self_attn.k_proj.weight", {kv_size, hidden_size}, false);
        check_then_add(layer + ".self_attn.v_proj.weight", {kv_size, hidden_size}, false);

        if (config.has_qk_norm) {
            check_then_add(layer + ".self_attn.q_norm.weight", {head_dim}, true);
            check_then_add(layer + ".self_attn.k_norm.weight", {head_dim}, true);
        }

        check_then_add(layer + ".self_attn.o_proj.weight", {hidden_size, q_size}, true);
        check_then_add(layer + ".post_attention_layernorm.weight", {hidden_size}, true);

        check_then_add(layer + ".mlp.gate_proj.weight", {intermediate_size, hidden_size}, true);
        check_then_add(layer + ".mlp.up_proj.weight", {intermediate_size, hidden_size}, false);
        check_then_add(layer + ".mlp.down_proj.weight", {hidden_size, intermediate_size}, true);
    }

    check_then_add("model.norm.weight", {hidden_size}, true);
    if (!config.tie_word_embeddings) {
        check_then_add("lm_head.weight", {vocab_size, hidden_size}, true);
    }

    std::sort(plan.begin(), plan.end(), [](const CopyPlan& a, const CopyPlan& b) {
        return a.src_offset < b.src_offset;
    });

    return plan;
}

void upload(
    std::ifstream& src_file,
    std::size_t data_offset,
    const std::vector<CopyPlan>& plans,
    CudaDeviceBuffer& dst_buffer
)
{
    std::size_t max_plan_copy = 0;
    for (const auto& p: plans) {
        max_plan_copy = std::max(max_plan_copy, p.byte_size);
    }

    std::vector<std::byte> buffer(std::min(max_plan_copy, kUploadChunkSize));
    for (const auto& p: plans) {
        std::size_t file_offset = data_offset + p.src_offset;
        src_file.seekg(file_offset, std::ios::beg);
        if (!src_file) {
            throw std::runtime_error(std::format("failed to seek to offset {}", file_offset));
        }

        std::size_t copied = 0;
        while (copied < p.byte_size) {
            std::size_t chunk_size = std::min(buffer.size(), p.byte_size - copied);
            if (!src_file.read(reinterpret_cast<char*>(buffer.data()), chunk_size)) {
                throw std::runtime_error(std::format("failed to read tensor `{}`", p.tensor_name));
            }
            dst_buffer.upload_at(p.dst_offset + copied, buffer.data(), chunk_size);

            copied += chunk_size;
        }
    }
}

void build_view(
    ModelWeights& weights,
    const std::vector<CopyPlan>& plans,
    const ModelConfig& config
)
{
    const int q_size = config.num_attention_heads * config.head_dim;
    const int kv_size = config.num_kv_heads * config.head_dim;

    auto find_copy = [&plans](std::string_view tensor_name) -> const CopyPlan& {
        const auto iter = std::find_if(
            plans.begin(), plans.end(),
            [tensor_name](const CopyPlan& p) {
                return p.tensor_name == tensor_name;
            }
        );
        assert(iter != plans.end());
        return *iter;
    };

    auto make_view_1d = [&](std::string_view tensor_name, int size) {
        const CopyPlan& copy = find_copy(tensor_name);
        return make_tensor<1>(
            weights.data.data<std::byte>() + copy.dst_offset,
            config.dtype,
            {size}
        );
    };

    auto make_view_2d = [&](std::string_view tensor_name, int rows, int columns) {
        const CopyPlan& copy = find_copy(tensor_name);
        return make_tensor<2>(
            weights.data.data<std::byte>() + copy.dst_offset,
            config.dtype,
            {rows, columns}
        );
    };

    weights.embed_tokens = make_view_2d(
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size
    );

    weights.norm = make_view_1d("model.norm.weight", config.hidden_size);

    weights.lm_head = config.tie_word_embeddings
        ? weights.embed_tokens
        : make_view_2d("lm_head.weight", config.vocab_size, config.hidden_size);

    weights.layers.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        const std::string layer = std::format("model.layers.{}", i);
        LayerWeights& dst = weights.layers[i];

        dst.input_layernorm = make_view_1d(layer + ".input_layernorm.weight", config.hidden_size);

        dst.qkv_proj = make_view_2d(
            layer + ".self_attn.q_proj.weight",
            q_size + 2 * kv_size,
            config.hidden_size
        );

        if (config.has_qk_norm) {
            dst.q_norm = make_view_1d(layer + ".self_attn.q_norm.weight", config.head_dim);
            dst.k_norm = make_view_1d(layer + ".self_attn.k_norm.weight", config.head_dim);
        }

        dst.o_proj = make_view_2d(layer + ".self_attn.o_proj.weight", config.hidden_size, q_size);

        dst.post_attn_layernorm = make_view_1d(
            layer + ".post_attention_layernorm.weight",
            config.hidden_size
        );

        dst.gate_up_proj = make_view_2d(
            layer + ".mlp.gate_proj.weight",
            2 * config.intermediate_size,
            config.hidden_size
        );

        dst.down_proj = make_view_2d(
            layer + ".mlp.down_proj.weight",
            config.hidden_size,
            config.intermediate_size
        );
    }
}

}

ModelWeights ModelWeights::load_from_safetensors(
    const std::filesystem::path& path,
    const ModelConfig& config
)
{
    try {
        // ┌───────────────────────┐
        // │ Header Size (8 bytes) │
        // ├───────────────────────┤
        // │ JSON Header           │
        // ├───────────────────────┤
        // │ Byte Buffer           │
        // └───────────────────────┘

        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open file");
        }
        
        std::size_t file_size = static_cast<std::size_t>(std::filesystem::file_size(path));

        // header size
        std::uint64_t header_size;
        if (
            !file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size))
            || header_size > kMaxHeaderSize
        ) {
            throw std::runtime_error("invalid header size");
        }

        // header
        std::vector<std::uint8_t> header_buffer(header_size);
        if (!file.read(reinterpret_cast<char*>(header_buffer.data()), header_buffer.size())) {
            throw std::runtime_error("failed to read header");
        }

        // parse header and build copy plan
        std::vector<CopyPlan> plans;
        try {
            const auto header = nlohmann::json::parse(header_buffer.begin(), header_buffer.end());
            plans = build_plan(header, config);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::format("invalid header: {}", e.what()));
        }

        // copy data from safetensors to device memory
        std::size_t expected_min_byte_buffer_size = 0;
        std::size_t device_buffer_size = 0;
        for (const auto& p: plans) {
            expected_min_byte_buffer_size = std::max(
                expected_min_byte_buffer_size, p.src_offset + p.byte_size
            );
            device_buffer_size = std::max(device_buffer_size, p.dst_offset + p.byte_size);
        }
        std::size_t byte_buffer_offset = sizeof(std::uint64_t) + header_size;
        // two if(!file.read(...)) ensure that file_size >= byte_buffer_offset
        std::size_t byte_buffer_size = file_size - byte_buffer_offset;
        if (expected_min_byte_buffer_size > byte_buffer_size) {
            throw std::runtime_error(std::format(
                "byte buffer size ({}) is smaller than expected ({}) ",
                byte_buffer_size, expected_min_byte_buffer_size
            ));
        }

        ModelWeights weights;
        weights.data.resize(device_buffer_size);
        upload(file, byte_buffer_offset, plans, weights.data);
        build_view(weights, plans, config);

        return weights;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::format(
            "failed to load model weights from safetensors `{}`: {}", path.string(), e.what()
        ));
    }
}

}
