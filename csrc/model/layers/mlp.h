#pragma once

#include "model/layers/linear.h"
#include "executor/forward_batch.h"


namespace cuinfer
{

class FeedForward
{
public:
    virtual ~FeedForward() = default;
    // workspace.hidden (normalized) -> workspace.hidden
    virtual void forward(const CudaContext&, const ForwardBatch&, const WorkspaceView&) const = 0;
};

class DenseMLP final : public FeedForward
{
public:
    static DenseMLP load(const Checkpoint&, const std::string& prefix, const ModelConfig&);
    void forward(const CudaContext&, const ForwardBatch&, const WorkspaceView&) const override;

private:
    Linear gate_up_proj_;
    Linear down_proj_;
};

}
