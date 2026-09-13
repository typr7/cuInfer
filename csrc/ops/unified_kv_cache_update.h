#pragma once

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cuinfer::ops
{

void unified_kv_cache_update(
    TensorRef<3> k_cache, // [num_slots, num_kv_heads, head_dim]
    TensorRef<3> v_cache, // [num_slots, num_kv_heads, head_dim]
    TensorRef<2> qkv, // [num_tokens, (num_q_heads + 2 * num_kv_heads) * head_dim]
    const int* slot_mapping,
    cudaStream_t stream
);

}
