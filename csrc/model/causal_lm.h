#pragma once

#include "model/model_config.h"
#include "model/model_weights.h"
#include "cuda/cuda_context.h"
#include "executor/forward_batch.h"
#include "executor/kv_cache.h"
#include "executor/workspace.h"
#include "model/rope_cache.h"


namespace cuinfer
{

class CausalLM
{
public:
    CausalLM(const ModelConfig& config, ModelWeights&& weights);
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
    ModelWeights weights_;
    RopeCache rope_;

    int q_size_;
    int kv_size_;
};

}
