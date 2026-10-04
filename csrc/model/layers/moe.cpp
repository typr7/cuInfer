#include <stdexcept>

#include "model/layers/moe.h"


namespace cuinfer
{

void SparseMoE::forward(const CudaContext& context, const ForwardBatch& batch,
                        const WorkspaceView& workspace) const
{
    // TODO: qwen3-30b-a3b
    throw std::runtime_error("SparseMoE is not implemented");
}

}
