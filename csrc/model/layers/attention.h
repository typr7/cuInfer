#pragma once

#include "model/layers/linear.h"
#include "model/rope_cache.h"
#include "executor/forward_batch.h"
#include "executor/kv_cache.h"


namespace cuinfer
{

class Attention
{
public:
    static Attention load(const Checkpoint&, const std::string& prefix, int layer,
                          const ModelConfig&, const CudaContext&, const RopeCache&);

    void prepare(const CudaContext&, const ForwardBatch&, const KVCacheView&, const WorkspaceView&) const;
    void attend(const CudaContext&, const ForwardBatch&, const KVCacheView&, const WorkspaceView&) const;
    void project(const CudaContext&, const ForwardBatch&, const WorkspaceView&) const;

private:
    Linear qkv_proj_;
    Linear o_proj_;
    Tensor<1> q_norm_;
    Tensor<1> k_norm_;
    CudaDeviceBuffer norm_data_;
    Tensor<3> rope_;
    int layer_;
    int q_size_;
    int kv_size_;
    int head_dim_;
    float rms_norm_eps_;
};

}
