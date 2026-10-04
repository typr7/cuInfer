#include "model/layers/attention.h"
#include "common/util.h"
#include "ops/paged_attention.h"
#include "ops/qk_norm_rope_fused.h"
#include "ops/rope.h"
#include "ops/unified_kv_cache_update.h"


namespace cuinfer
{

Attention Attention::load(const Checkpoint& checkpoint, const std::string& prefix, int layer,
                          const ModelConfig& config, const CudaContext& context, const RopeCache& rope)
{
    Attention attention;
    attention.layer_ = layer;
    attention.q_size_ = config.num_attention_heads * config.head_dim;
    attention.kv_size_ = config.num_kv_heads * config.head_dim;
    attention.head_dim_ = config.head_dim;
    attention.rms_norm_eps_ = config.rms_norm_eps;
    attention.rope_ = rope.view;
    attention.qkv_proj_ = Linear::load(checkpoint, {
        {prefix + ".q_proj", attention.q_size_},
        {prefix + ".k_proj", attention.kv_size_},
        {prefix + ".v_proj", attention.kv_size_}
    }, config.hidden_size, Parallel::kColumn);
    attention.o_proj_ = Linear::load(checkpoint, {{prefix + ".o_proj", config.hidden_size}},
                                    attention.q_size_, Parallel::kRow);
    if (config.has_qk_norm) {
        const std::size_t norm_bytes = config.head_dim * dtype_byte_size(config.dtype);
        const std::size_t k_offset = align_up<256>(norm_bytes);
        attention.norm_data_.resize(k_offset + norm_bytes);
        auto* base = attention.norm_data_.data<std::byte>();
        upload_slice(checkpoint.get(prefix + ".q_norm.weight", config.dtype, {config.head_dim}),
                     0, 0, config.head_dim, base);
        upload_slice(checkpoint.get(prefix + ".k_norm.weight", config.dtype, {config.head_dim}),
                     0, 0, config.head_dim, base + k_offset);
        attention.q_norm_ = make_tensor<1>(base, config.dtype, {config.head_dim});
        attention.k_norm_ = make_tensor<1>(base + k_offset, config.dtype, {config.head_dim});
    }
    return attention;
}

void Attention::prepare(const CudaContext& context, const ForwardBatch& batch,
                        const KVCacheView& kv_cache, const WorkspaceView& workspace) const
{
    qkv_proj_.forward(workspace.hidden, workspace.qkv, context, workspace);

    // logical reshape: [T, Q/K] -> [T, H_Q/K, D]

    if (q_norm_) {
        ops::qk_norm_rope(
            workspace.qkv,
            q_norm_,
            k_norm_,
            rope_,
            batch.positions,
            q_size_,
            kv_size_,
            head_dim_,
            rms_norm_eps_,
            context.stream(),
            batch.num_tokens_device
        );
    } else {
        ops::rope(
            workspace.qkv,
            rope_,
            batch.positions,
            q_size_,
            kv_size_,
            head_dim_,
            context.stream(),
            batch.num_tokens_device
        );
    }

    ops::unified_kv_cache_update(
        kv_cache.k(layer_),
        kv_cache.v(layer_),
        workspace.qkv,
        batch.slot_mapping,
        context.stream()
    );
}

void Attention::attend(const CudaContext& context, const ForwardBatch& batch,
                       const KVCacheView& kv_cache, const WorkspaceView& workspace) const
{
    ops::paged_attention(
        workspace.qkv,
        kv_cache.k(layer_),
        kv_cache.v(layer_),
        workspace.attn_out,
        batch,
        kv_cache.block_size,
        context.stream()
    );
}

void Attention::project(const CudaContext& context, const ForwardBatch& batch,
                        const WorkspaceView& workspace) const
{
    o_proj_.forward(workspace.attn_out, workspace.hidden, context, workspace);
}

}
