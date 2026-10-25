#pragma once

#include "model/layers/mlp.h"


namespace cuinfer
{

class SparseMoE final : public FeedForward
{
public:
    static SparseMoE load(const Checkpoint&, const std::string& prefix, const ModelConfig&);
    void forward(const CudaContext&, const ForwardBatch&, const WorkspaceView&) const override;

private:
    Linear gate_;
    GroupedLinear experts_w13_;
    GroupedLinear experts_w2_;
    int top_k_;
    bool norm_topk_prob_;
    int expert_begin_;
    int expert_end_;
};

}
