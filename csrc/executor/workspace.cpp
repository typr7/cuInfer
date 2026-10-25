#include <algorithm>
#include <cassert>

#include "executor/workspace.h"
#include "tensor/data_type.h"
#include "common/util.h"
#include "cuda/cuda_utils.h"


namespace cuinfer
{

namespace
{

constexpr std::size_t kAlignment = 256;

}

Workspace Workspace::create(
    const ModelConfig& config,
    int max_tokens,
    int max_sampling_reqs
)
{
    assert(max_tokens > 0);
    assert(max_sampling_reqs > 0);

    const std::size_t hidden_size = config.hidden_size;
    const std::size_t q_size
        = static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
    const std::size_t kv_size = static_cast<std::size_t>(config.num_kv_heads) * config.head_dim;
    const std::size_t intermediate = config.intermediate_size;
    const std::size_t vocab_size = config.vocab_size;

    auto reserve_tensor_bytes = [](std::size_t& offset, std::size_t size, DataType dtype) {
        offset = align_up<kAlignment>(offset);
        offset += size * dtype_byte_size(dtype);
    };

    std::size_t attn_workspace_size = 0;
    reserve_tensor_bytes(attn_workspace_size, max_tokens * hidden_size, config.dtype); // hidden
    reserve_tensor_bytes(attn_workspace_size, max_tokens * hidden_size, config.dtype); // residual
    reserve_tensor_bytes(attn_workspace_size, max_tokens * (q_size + 2 * kv_size), config.dtype); // qkv
    reserve_tensor_bytes(attn_workspace_size, max_tokens * q_size, config.dtype); // attn_out

    std::size_t mlp_workspace_size = 0;
    reserve_tensor_bytes(mlp_workspace_size, max_tokens * hidden_size, config.dtype); // hidden
    reserve_tensor_bytes(mlp_workspace_size, max_tokens * hidden_size, config.dtype); // residual
    reserve_tensor_bytes(mlp_workspace_size, max_tokens * 2 * intermediate, config.dtype); // gate_up

    std::size_t moe_workspace_size = 0;
    if (config.moe) {
        const auto& moe = *config.moe;
        reserve_tensor_bytes(moe_workspace_size, max_tokens * hidden_size, config.dtype); // hidden
        reserve_tensor_bytes(moe_workspace_size, max_tokens * hidden_size, config.dtype); // residual
        reserve_tensor_bytes(moe_workspace_size, max_tokens * moe.num_experts, config.dtype); // router_logits
        reserve_tensor_bytes(moe_workspace_size, max_tokens * moe.top_k, DataType::kFp32); // topk_weights
        reserve_tensor_bytes(moe_workspace_size, max_tokens * moe.top_k, DataType::kInt32); // topk_ids
        reserve_tensor_bytes(moe_workspace_size, moe.num_experts, DataType::kInt32); // expert_counts
        reserve_tensor_bytes(
            moe_workspace_size,
            max_tokens * moe.top_k * 2 * moe.intermediate_size,
            config.dtype
        ); // moe_gate_up
    }

    std::size_t logits_workspace_size = 0;
    reserve_tensor_bytes(logits_workspace_size, max_sampling_reqs * hidden_size, config.dtype); // sampling_hidden
    reserve_tensor_bytes(logits_workspace_size, max_sampling_reqs * vocab_size, config.dtype); // logits

    std::size_t required_byte_size = std::max({
        attn_workspace_size,
        mlp_workspace_size,
        moe_workspace_size,
        logits_workspace_size
    });

    CudaDeviceBuffer data(required_byte_size);
    auto* base = data.data<std::byte>();

    auto place_view_2d = [base](std::size_t& offset, std::size_t rows, std::size_t cols, DataType dtype) {
        offset = align_up<kAlignment>(offset);
        auto* device_ptr = base + offset;
        offset += rows * cols * dtype_byte_size(dtype);
        return make_tensor<2>(device_ptr, dtype, {static_cast<int>(rows), static_cast<int>(cols)});
    };

    auto place_view_1d = [base](std::size_t& offset, std::size_t dim, DataType dtype) {
        offset = align_up<kAlignment>(offset);
        auto* device_ptr = base + offset;
        offset += dim * dtype_byte_size(dtype);
        return make_tensor<1>(device_ptr, dtype, {static_cast<int>(dim)});
    };

    attn_workspace_size = 0;
    logits_workspace_size = 0;
    
    Workspace workspace {};
    workspace.hidden_ = place_view_2d(attn_workspace_size, max_tokens, hidden_size, config.dtype);
    workspace.residual_ = place_view_2d(attn_workspace_size, max_tokens, hidden_size, config.dtype);

    mlp_workspace_size = attn_workspace_size;
    moe_workspace_size = attn_workspace_size;
    // attn
    workspace.qkv_ = place_view_2d(attn_workspace_size, max_tokens, q_size + 2 * kv_size, config.dtype);
    workspace.attn_out_ = place_view_2d(attn_workspace_size, max_tokens, q_size, config.dtype);
    // mlp
    workspace.gate_up_ = place_view_2d(mlp_workspace_size, max_tokens, 2 * intermediate, config.dtype);
    workspace.gated_ = make_tensor<2>(
        workspace.gate_up_.device_ptr,
        workspace.gate_up_.dtype,
        {max_tokens, workspace.gate_up_.shape[1] / 2},
        {workspace.gate_up_.stride[0], 1}
    );
    // moe
    if (config.moe) {
        const auto& moe = *config.moe;
        workspace.router_logits_ = place_view_2d(
            moe_workspace_size,
            max_tokens,
            moe.num_experts,
            config.dtype
        );
        workspace.topk_weights_ = place_view_2d(
            moe_workspace_size,
            max_tokens,
            moe.top_k,
            DataType::kFp32
        );
        workspace.topk_ids_ = place_view_2d(
            moe_workspace_size,
            max_tokens,
            moe.top_k,
            DataType::kInt32
        );
        workspace.expert_counts_ = place_view_1d(
            moe_workspace_size,
            moe.num_experts,
            DataType::kInt32
        );
        workspace.moe_gate_up_ = place_view_2d(
            moe_workspace_size,
            max_tokens * moe.top_k,
            2 * moe.intermediate_size,
            config.dtype
        );
        workspace.moe_gated_ = make_tensor<2>(
            workspace.moe_gate_up_.device_ptr,
            workspace.moe_gate_up_.dtype,
            {max_tokens * moe.top_k, moe.intermediate_size},
            {workspace.moe_gate_up_.stride[0], 1}
        );
    }
    // compute logits
    workspace.sampling_hidden_ = place_view_2d(
        logits_workspace_size,
        max_sampling_reqs,
        hidden_size,
        config.dtype
    );
    workspace.logits_ = place_view_2d(
        logits_workspace_size,
        max_sampling_reqs,
        vocab_size,
        config.dtype
    );

    workspace.data_ = std::move(data);

    return workspace;
}

void Workspace::zero(const CudaContext& context)
{
    CUDA_CHECK(cudaMemsetAsync(
        data_.data(), 0, data_.size(), context.stream()
    ));
}

WorkspaceView Workspace::view(
    int num_tokens,
    int num_sampling_reqs
) const
{
    assert(num_tokens > 0 && num_tokens <= hidden_.shape[0]);
    assert(num_sampling_reqs >= 0 && num_sampling_reqs <= logits_.shape[0]);

    auto with_rows = [](Tensor<2> tensor, int rows) {
        if (tensor) {
            tensor.shape[0] = rows;
        }
        return tensor;
    };

    const int num_routed_rows = num_tokens * topk_ids_.shape[1];

    return WorkspaceView{
        .hidden = with_rows(hidden_, num_tokens),
        .residual = with_rows(residual_, num_tokens),
        .qkv = with_rows(qkv_, num_tokens),
        .attn_out = with_rows(attn_out_, num_tokens),
        .gate_up = with_rows(gate_up_, num_tokens),
        .gated = with_rows(gated_, num_tokens),
        .sampling_hidden = with_rows(sampling_hidden_, num_sampling_reqs),
        .logits = with_rows(logits_, num_sampling_reqs),
        .router_logits = with_rows(router_logits_, num_tokens),
        .topk_weights = with_rows(topk_weights_, num_tokens),
        .topk_ids = with_rows(topk_ids_, num_tokens),
        .expert_counts = expert_counts_,
        .moe_gate_up = with_rows(moe_gate_up_, num_routed_rows),
        .moe_gated = with_rows(moe_gated_, num_routed_rows)
    };
}

}
