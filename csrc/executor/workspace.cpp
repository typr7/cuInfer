#include <algorithm>
#include <cassert>

#include "executor/workspace.h"
#include "tensor/data_type.h"
#include "common/util.h"


namespace cllm
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
    const std::size_t dtype_size = dtype_byte_size(config.dtype);

    auto reserve_tensor_bytes = [dtype_size](std::size_t& offset, std::size_t size) {
        offset = align_up<kAlignment>(offset);
        offset += size * dtype_size;
    };

    std::size_t attn_workspace_size = 0;
    reserve_tensor_bytes(attn_workspace_size, max_tokens * hidden_size); // hidden
    reserve_tensor_bytes(attn_workspace_size, max_tokens * hidden_size); // residual
    reserve_tensor_bytes(attn_workspace_size, max_tokens * (q_size + 2 * kv_size)); // qkv
    reserve_tensor_bytes(attn_workspace_size, max_tokens * q_size); // attn_out

    std::size_t ffn_workspace_size = 0;
    reserve_tensor_bytes(ffn_workspace_size, max_tokens * hidden_size); // hidden
    reserve_tensor_bytes(ffn_workspace_size, max_tokens * hidden_size); // residual
    reserve_tensor_bytes(ffn_workspace_size, max_tokens * 2 * intermediate); // gate_up

    std::size_t logits_workspace_size = 0;
    reserve_tensor_bytes(
        logits_workspace_size,
        max_sampling_reqs * hidden_size
    ); // sampling_hidden
    reserve_tensor_bytes(logits_workspace_size, max_sampling_reqs * vocab_size); // logits

    std::size_t required_byte_size = std::max({
        attn_workspace_size,
        ffn_workspace_size,
        logits_workspace_size
    });

    CudaDeviceBuffer data(required_byte_size);
    auto* base = data.data<std::byte>();

    auto place_view = [&](std::size_t& offset, std::size_t rows, std::size_t cols) {
        offset = align_up<kAlignment>(offset);
        auto* device_ptr = base + offset;
        offset += rows * cols * dtype_size;
        return make_tensor<2>(
            device_ptr,
            config.dtype,
            {static_cast<int>(rows), static_cast<int>(cols)}
        );
    };

    attn_workspace_size = 0;
    logits_workspace_size = 0;
    
    Workspace workspace {};
    workspace.hidden_ = place_view(attn_workspace_size, max_tokens, hidden_size);
    workspace.residual_ = place_view(attn_workspace_size, max_tokens, hidden_size);

    ffn_workspace_size = attn_workspace_size;
    // attn
    workspace.qkv_ = place_view(attn_workspace_size, max_tokens, q_size + 2 * kv_size);
    workspace.attn_out_ = place_view(attn_workspace_size, max_tokens, q_size);
    // ffn
    workspace.gate_up_ = place_view(ffn_workspace_size, max_tokens, 2 * intermediate);
    workspace.gated_ = make_tensor<2>(
        workspace.gate_up_.device_ptr,
        workspace.gate_up_.dtype,
        {max_tokens, workspace.gate_up_.shape[1] / 2},
        {workspace.gate_up_.stride[0], 1}
    );
    // compute logits
    workspace.sampling_hidden_ = place_view(
        logits_workspace_size,
        max_sampling_reqs,
        hidden_size
    );
    workspace.logits_ = place_view(logits_workspace_size, max_sampling_reqs, vocab_size);

    workspace.data_ = std::move(data);

    return workspace;
}

WorkspaceView Workspace::view(
    int num_tokens,
    int num_sampling_reqs
) const
{
    assert(num_tokens > 0 && num_tokens <= hidden_.shape[0]);
    assert(num_sampling_reqs >= 0 && num_sampling_reqs <= logits_.shape[0]);

    auto with_rows = [](Tensor<2> tensor, int rows) {
        tensor.shape[0] = rows;
        return tensor;
    };

    return WorkspaceView{
        .hidden = with_rows(hidden_, num_tokens),
        .residual = with_rows(residual_, num_tokens),
        .qkv = with_rows(qkv_, num_tokens),
        .attn_out = with_rows(attn_out_, num_tokens),
        .gate_up = with_rows(gate_up_, num_tokens),
        .gated = with_rows(gated_, num_tokens),
        .sampling_hidden = with_rows(sampling_hidden_, num_sampling_reqs),
        .logits = with_rows(logits_, num_sampling_reqs)
    };
}

}
