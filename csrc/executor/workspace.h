#pragma once

#include "tensor/tensor.h"
#include "cuda/cuda_device_buffer.h"
#include "model/model_config.h"


namespace cuinfer
{

struct WorkspaceView
{
    Tensor<2> hidden; // [T, H] 1, 4, 7
    Tensor<2> residual; // [T, H] 0, 5, 8
    Tensor<2> qkv; // [T, H_q + 2 * H_kv] 2
    Tensor<2> attn_out; // [T, H_q] 3
    Tensor<2> gate_up; // [T, 2 * I] 6
    Tensor<2> gated;
    Tensor<2> sampling_hidden; // [S, H]
    Tensor<2> logits; // [S, vocab_len]
};

struct Workspace
{
    static Workspace create(
        const ModelConfig& config,
        int max_tokens,
        int max_sampling_reqs
    );

    WorkspaceView view(int num_tokens, int num_sampling_reqs) const;

private:
    Tensor<2> hidden_;
    Tensor<2> residual_;
    Tensor<2> qkv_;
    Tensor<2> attn_out_;
    Tensor<2> gate_up_;
    Tensor<2> gated_;
    Tensor<2> sampling_hidden_;
    Tensor<2> logits_;

    CudaDeviceBuffer data_;
};

}
