#include <format>

#include "model/causal_lm.h"
#include "common/util.h"
#include "ops/embedding.h"
#include "ops/rms_norm.h"
#include "ops/residual_add.h"


namespace cuinfer
{

CausalLM::CausalLM(const ModelConfig& config, const std::filesystem::path& model_dir,
                   const CudaContext& context)
    : config_(config),
      rope_(RopeCache::create(config))
{
    const Checkpoint checkpoint(model_dir);
    const auto& embed = checkpoint.get("model.embed_tokens.weight", config.dtype,
                                       {config.vocab_size, config.hidden_size});
    embed_data_.resize(static_cast<std::size_t>(config.vocab_size) * config.hidden_size
                       * dtype_byte_size(config.dtype));
    upload_slice(embed, 0, 0, config.vocab_size, embed_data_.data());
    embed_tokens_ = make_tensor<2>(embed_data_.data(), config.dtype,
                                   {config.vocab_size, config.hidden_size});

    const std::size_t norm_bytes = config.hidden_size * dtype_byte_size(config.dtype);
    norm_data_.resize(norm_bytes);
    upload_slice(checkpoint.get("model.norm.weight", config.dtype, {config.hidden_size}),
                 0, 0, config.hidden_size, norm_data_.data());
    norm_ = make_tensor<1>(norm_data_.data(), config.dtype, {config.hidden_size});
    lm_head_ = config.tie_word_embeddings
        ? Linear::tied(embed_tokens_)
        : Linear::load(checkpoint, {{"lm_head", config.vocab_size}}, config.hidden_size,
                       Parallel::kReplicated);

    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        const std::string prefix = std::format("model.layers.{}", i);
        DecoderLayer layer;
        const std::size_t post_offset = align_up<256>(norm_bytes);
        layer.norm_data.resize(post_offset + norm_bytes);
        auto* base = layer.norm_data.data<std::byte>();
        upload_slice(checkpoint.get(prefix + ".input_layernorm.weight", config.dtype,
                                    {config.hidden_size}), 0, 0, config.hidden_size, base);
        upload_slice(checkpoint.get(prefix + ".post_attention_layernorm.weight", config.dtype,
                                    {config.hidden_size}), 0, 0, config.hidden_size, base + post_offset);
        layer.input_layernorm = make_tensor<1>(base, config.dtype, {config.hidden_size});
        layer.post_attention_layernorm = make_tensor<1>(base + post_offset, config.dtype,
                                                       {config.hidden_size});
        layer.attn = Attention::load(checkpoint, prefix + ".self_attn", i, config, context, rope_);
        // TODO: SparseMoE
        layer.ffn = std::make_unique<DenseMLP>(DenseMLP::load(checkpoint, prefix + ".mlp", config));
        layers_.push_back(std::move(layer));
    }
}

void CausalLM::forward(
    const CudaContext& context,
    const ForwardBatch& batch,
    const KVCacheView& kv_cache,
    const WorkspaceView& workspace
) const
{
    ops::embedding(
        batch.token_ids,
        embed_tokens_,
        workspace.residual,
        context.stream(),
        batch.num_tokens_device
    );

    for (int layer = 0; layer < config_.num_hidden_layers; layer++) {
        pre_attention(context, batch, kv_cache, workspace, layer);
        forward_attention(context, batch, kv_cache, workspace, layer);
        post_attention(context, batch, workspace, layer);
    }
}

void CausalLM::compute_logits(
    const CudaContext& context,
    const ForwardBatch& batch,
    const WorkspaceView& workspace
) const
{
    if (batch.num_sampling_reqs == 0) {
        return;
    }

    ops::embedding(
        batch.logits_indices,
        workspace.residual,
        workspace.sampling_hidden,
        context.stream()
    );

    ops::rms_norm(
        workspace.sampling_hidden,
        norm_,
        workspace.sampling_hidden,
        config_.rms_norm_eps,
        context.stream()
    );

    lm_head_.forward(workspace.sampling_hidden, workspace.logits, context, workspace);
}

void CausalLM::forward_graph_segment(
    const CudaContext& context,
    const ForwardBatch& batch,
    const KVCacheView& kv_cache,
    const WorkspaceView& workspace,
    int segment
) const
{
    if (segment == 0) {
        ops::embedding(
            batch.token_ids, embed_tokens_, workspace.residual,
            context.stream(), batch.num_tokens_device
        );
    } else {
        post_attention(context, batch, workspace, segment - 1);
    }
    if (segment < config_.num_hidden_layers) {
        pre_attention(context, batch, kv_cache, workspace, segment);
    }
}

void CausalLM::pre_attention(
    const CudaContext& context,
    const ForwardBatch& batch,
    const KVCacheView& kv_cache,
    const WorkspaceView& workspace,
    int layer
) const
{
    const DecoderLayer& decoder = layers_.at(layer);
    ops::rms_norm(
        workspace.residual,
        decoder.input_layernorm,
        workspace.hidden,
        config_.rms_norm_eps,
        context.stream(),
        batch.num_tokens_device
    );
    decoder.attn.prepare(context, batch, kv_cache, workspace);
}

void CausalLM::forward_attention(
    const CudaContext& context,
    const ForwardBatch& batch,
    const KVCacheView& kv_cache,
    const WorkspaceView& workspace,
    int layer
) const
{
    layers_.at(layer).attn.attend(context, batch, kv_cache, workspace);
}

void CausalLM::post_attention(
    const CudaContext& context,
    const ForwardBatch& batch,
    const WorkspaceView& workspace,
    int layer
) const
{
    const DecoderLayer& decoder = layers_.at(layer);
    decoder.attn.project(context, batch, workspace);
    ops::residual_add(workspace.hidden, workspace.residual, context.stream(), batch.num_tokens_device);
    ops::rms_norm(
        workspace.residual,
        decoder.post_attention_layernorm,
        workspace.hidden,
        config_.rms_norm_eps,
        context.stream(),
        batch.num_tokens_device
    );
    decoder.ffn->forward(context, batch, workspace);
    ops::residual_add(workspace.hidden, workspace.residual, context.stream(), batch.num_tokens_device);
}

}
