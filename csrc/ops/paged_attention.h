#pragma once

#include <cuda_runtime.h>

#include "executor/forward_batch.h"
#include "tensor/tensor.h"


namespace cllm::ops
{

// BF16, head_dim 64/128, power-of-two pages. Call after updating the KV cache.
// Query tokens are the final query_len positions of each request's seq_len.
void paged_attention(
    TensorRef<2> qkv,     // [num_tokens, (num_q_heads + 2 * num_kv_heads) * head_dim]
    TensorRef<3> k_cache, // [num_blocks * block_size, num_kv_heads, head_dim]
    TensorRef<3> v_cache,
    TensorRef<2> output,  // [num_tokens, num_q_heads * head_dim]
    const ForwardBatch& batch,
    int block_size,
    cudaStream_t stream
);

}
