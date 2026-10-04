#pragma once

#include <memory>
#include <vector>

#include "model/model_config.h"
#include "model/layers/attention.h"
#include "model/layers/mlp.h"
#include "cuda/cuda_context.h"
#include "executor/forward_batch.h"
#include "executor/kv_cache.h"
#include "executor/workspace.h"
#include "model/rope_cache.h"


namespace cuinfer
{

struct DecoderLayer
{
    Tensor<1> input_layernorm;
    Attention attn;
    Tensor<1> post_attention_layernorm;
    std::unique_ptr<FeedForward> ffn;
    CudaDeviceBuffer norm_data;
};

class CausalLM
{
public:
    CausalLM(const ModelConfig& config, const std::filesystem::path& model_dir,
             const CudaContext& context);
    ~CausalLM() noexcept = default;

    void forward(
        const CudaContext& context,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        const WorkspaceView& workspace
    ) const;

    void compute_logits(
        const CudaContext& context,
        const ForwardBatch& batch,
        const WorkspaceView& workspace
    ) const;

    void forward_graph_segment(
        const CudaContext& context,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        const WorkspaceView& workspace,
        int segment
    ) const;

    void forward_attention(
        const CudaContext& context,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        const WorkspaceView& workspace,
        int layer
    ) const;

private:
    void pre_attention(
        const CudaContext& context,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        const WorkspaceView& workspace,
        int layer
    ) const;

    void post_attention(
        const CudaContext& context,
        const ForwardBatch& batch,
        const WorkspaceView& workspace,
        int layer
    ) const;

private:
    ModelConfig config_;
    Tensor<2> embed_tokens_;
    Tensor<1> norm_;
    CudaDeviceBuffer embed_data_;
    CudaDeviceBuffer norm_data_;
    Linear lm_head_;
    RopeCache rope_;
    std::vector<DecoderLayer> layers_;
};

}
