#pragma once

#include "tensor/tensor.h"
#include "cuda/cuda_device_buffer.h"
#include "cuda/cuda_context.h"
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

    // MoE
    Tensor<2> router_logits; // [T, E]
    Tensor<2> topk_weights; // [T, K], FP32
    Tensor<2> topk_ids; // [T, K], INT32
    Tensor<1> expert_counts; // [E], INT32
    Tensor<2> moe_gate_up; // [T * K, 2 * I_moe]
    Tensor<2> moe_gated; // [T * K, I_moe], stride 2 * I_moe

    // TODO: quant
};

struct Workspace
{
    static Workspace create(
        const ModelConfig& config,
        int max_tokens,
        int max_sampling_reqs
    );

    WorkspaceView view(int num_tokens, int num_sampling_reqs) const;
    void zero(const CudaContext& context);

private:
    Tensor<2> hidden_;
    Tensor<2> residual_;
    Tensor<2> qkv_;
    Tensor<2> attn_out_;
    Tensor<2> gate_up_;
    Tensor<2> gated_;
    Tensor<2> sampling_hidden_;
    Tensor<2> logits_;

    // MoE
    Tensor<2> router_logits_;
    Tensor<2> topk_weights_;
    Tensor<2> topk_ids_;
    Tensor<1> expert_counts_;
    Tensor<2> moe_gate_up_;
    Tensor<2> moe_gated_;

    CudaDeviceBuffer data_;
};

}
