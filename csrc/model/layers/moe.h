#pragma once

#include "model/layers/mlp.h"


namespace cuinfer
{

class SparseMoE final : public FeedForward
{
public:
    void forward(const CudaContext&, const ForwardBatch&, const WorkspaceView&) const override;
};

}
