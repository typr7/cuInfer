#pragma once

#include <string>
#include <vector>

#include "model/checkpoint.h"
#include "cuda/cuda_context.h"
#include "executor/workspace.h"


namespace cuinfer
{

enum class WeightFormat { kBf16, kW4A16, kW8A8 };
enum class Parallel { kReplicated, kColumn, kRow };

struct LinearSource
{
    std::string prefix;
    int out_features;
};

class Linear
{
public:
    static Linear load(const Checkpoint&, const std::vector<LinearSource>& sources,
                       int in_features, Parallel parallel);
    static Linear tied(TensorRef<2> weight);

    void forward(TensorRef<2> in, TensorRef<2> out,
                 const CudaContext&, const WorkspaceView&) const;

    int in_features() const { return weight_.shape[1]; }
    int out_features() const { return weight_.shape[0]; }

private:
    WeightFormat format_ = WeightFormat::kBf16;
    Parallel parallel_ = Parallel::kReplicated;
    Tensor<2> weight_;
    CudaDeviceBuffer data_;
};

class GroupedLinear
{
public:
    static GroupedLinear load(const Checkpoint&,
                             const std::vector<std::vector<LinearSource>>& experts,
                             int in_features);
    void forward(TensorRef<2> in, TensorRef<2> out, const int* sorted_token_ids,
                 const int* expert_ids, const CudaContext&, const WorkspaceView&) const;

private:
    Tensor<3> weight_; // [E_local, N, K]
    CudaDeviceBuffer data_;
};

}
