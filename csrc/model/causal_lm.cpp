#include <cmath>

#include "model/causal_lm.h"
#include "ops/embedding.h"
#include "ops/paged_attention.h"
#include "ops/projection.h"
#include "ops/rms_norm.h"
#include "ops/qk_norm_rope_fused.h"
#include "ops/rope.h"
#include "ops/residual_add.h"
#include "ops/swiglu.h"
#include "ops/unified_kv_cache_update.h"


namespace cllm
{

CausalLM::CausalLM(const ModelConfig& config, ModelWeights&& weights)
    : config_(config),
      weights_(std::move(weights)),
      rope_(RopeCache::create(config)),
      q_size_(config.num_attention_heads * config.head_dim),
      kv_size_(config.num_kv_heads * config.head_dim)
{
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
        weights_.embed_tokens,
        workspace.residual,
        context.stream()
    );

    for (int layer = 0; layer < config_.num_hidden_layers; layer++) {
        decoder_layer(context, batch, kv_cache, workspace, layer);
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
        weights_.norm,
        workspace.sampling_hidden,
        config_.rms_norm_eps,
        context.stream()
    );

    ops::projection(
        workspace.sampling_hidden,
        weights_.lm_head,
        workspace.logits,
        context.cublas()
    );
}

void CausalLM::decoder_layer(
    const CudaContext& context,
    const ForwardBatch& batch,
    const KVCacheView& kv_cache,
    const WorkspaceView& workspace,
    int layer
) const
{
    const LayerWeights& layer_weights = weights_.layers.at(layer);
    ops::rms_norm(
        workspace.residual,
        layer_weights.input_layernorm,
        workspace.hidden,
        config_.rms_norm_eps,
        context.stream()
    );
    ops::projection(workspace.hidden, layer_weights.qkv_proj, workspace.qkv, context.cublas());

    // logical reshape: [T, Q/K] -> [T, H_Q/K, D]

    if (config_.has_qk_norm) {
        ops::qk_norm_rope(
            workspace.qkv,
            layer_weights.q_norm,
            layer_weights.k_norm,
            rope_.view,
            batch.positions,
            q_size_,
            kv_size_,
            config_.head_dim,
            config_.rms_norm_eps,
            context.stream()
        );
    } else {
        ops::rope(
            workspace.qkv,
            rope_.view,
            batch.positions,
            q_size_,
            kv_size_,
            config_.head_dim,
            context.stream()
        );
    }

    ops::unified_kv_cache_update(
        kv_cache.k(layer),
        kv_cache.v(layer),
        workspace.qkv,
        batch.slot_mapping,
        context.stream()
    );

    ops::paged_attention(
        workspace.qkv,
        kv_cache.k(layer),
        kv_cache.v(layer),
        workspace.attn_out,
        batch,
        kv_cache.block_size,
        context.stream()
    );

    ops::projection(workspace.attn_out, layer_weights.o_proj, workspace.hidden, context.cublas());

    ops::residual_add(workspace.hidden, workspace.residual, context.stream());

    ops::rms_norm(
        workspace.residual,
        layer_weights.post_attn_layernorm,
        workspace.hidden,
        config_.rms_norm_eps,
        context.stream()
    );

    ops::projection(
        workspace.hidden,
        layer_weights.gate_up_proj,
        workspace.gate_up,
        context.cublas()
    );

    ops::swiglu(workspace.gate_up, context.stream()); // -> gated

    ops::projection(workspace.gated, layer_weights.down_proj, workspace.hidden, context.cublas());

    ops::residual_add(workspace.hidden, workspace.residual, context.stream());
}

}
