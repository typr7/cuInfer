#include "model/layers/mlp.h"
#include "ops/swiglu.h"


namespace cuinfer
{

DenseMLP DenseMLP::load(const Checkpoint& checkpoint, const std::string& prefix,
                        const ModelConfig& config)
{
    DenseMLP mlp;
    mlp.gate_up_proj_ = Linear::load(checkpoint, {
        {prefix + ".gate_proj", config.intermediate_size},
        {prefix + ".up_proj", config.intermediate_size}
    }, config.hidden_size, Parallel::kColumn);
    mlp.down_proj_ = Linear::load(checkpoint, {{prefix + ".down_proj", config.hidden_size}},
                                  config.intermediate_size, Parallel::kRow);
    return mlp;
}

void DenseMLP::forward(const CudaContext& context, const ForwardBatch& batch,
                       const WorkspaceView& workspace) const
{
    gate_up_proj_.forward(workspace.hidden, workspace.gate_up, context, workspace);
    ops::swiglu(workspace.gate_up, context.stream(), batch.num_tokens_device);
    down_proj_.forward(workspace.gated, workspace.hidden, context, workspace);
}

}
