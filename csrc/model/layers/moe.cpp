#include <format>

#include "model/layers/moe.h"
#include "ops/topk_gating_softmax.h"


namespace cuinfer
{

SparseMoE SparseMoE::load(const Checkpoint& checkpoint, const std::string& prefix,
                          const ModelConfig& config)
{
    const MoEConfig& moe_config = config.moe.value();
    SparseMoE moe;
    moe.gate_ = Linear::load(checkpoint, {{prefix + ".gate", moe_config.num_experts}},
                             config.hidden_size, Parallel::kReplicated);
    moe.top_k_ = moe_config.top_k;
    moe.norm_topk_prob_ = moe_config.norm_topk_prob;
    // TODO: Select this rank's expert range when EP is implemented.
    moe.expert_begin_ = 0;
    moe.expert_end_ = moe_config.num_experts;

    std::vector<std::vector<LinearSource>> w13_sources;
    std::vector<std::vector<LinearSource>> w2_sources;
    w13_sources.reserve(moe.expert_end_ - moe.expert_begin_);
    w2_sources.reserve(moe.expert_end_ - moe.expert_begin_);
    for (int expert = moe.expert_begin_; expert < moe.expert_end_; expert++) {
        const std::string expert_prefix = std::format("{}.experts.{}", prefix, expert);
        w13_sources.push_back({
            {expert_prefix + ".gate_proj", moe_config.intermediate_size},
            {expert_prefix + ".up_proj", moe_config.intermediate_size}
        });
        w2_sources.push_back({{expert_prefix + ".down_proj", config.hidden_size}});
    }
    moe.experts_w13_ = GroupedLinear::load(checkpoint, w13_sources, config.hidden_size);
    moe.experts_w2_ = GroupedLinear::load(checkpoint, w2_sources, moe_config.intermediate_size);
    return moe;
}

void SparseMoE::forward(const CudaContext& context, const ForwardBatch& batch,
                        const WorkspaceView& workspace) const
{
    // TODO: qwen3-30b-a3b

    // router
    gate_.forward(workspace.hidden, workspace.router_logits, context, workspace);

    ops::topk_gating_softmax(
        workspace.router_logits,
        workspace.topk_weights,
        workspace.topk_ids,
        norm_topk_prob_,
        context.stream(),
        batch.num_tokens_device
    );

    
}

}
