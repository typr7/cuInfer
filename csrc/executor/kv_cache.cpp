#include <cassert>

#include "executor/kv_cache.h"


namespace cuinfer
{

// kv cache pool shape: [num_hidden_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
Tensor<3> KVCacheView::kv(int layer, int which) const noexcept
{
    assert(data != nullptr);
    assert(0 <= layer && layer < num_layers);
    assert(which == 0 || which == 1);

    const int num_slots = num_blocks * block_size;
    const std::size_t stride = static_cast<std::size_t>(num_slots) * num_kv_heads * head_dim;
    const std::size_t offset = (static_cast<std::size_t>(layer) * 2 + which) * stride;

    return make_tensor<3>(
        static_cast<std::byte*>(data) + offset * dtype_byte_size(dtype),
        dtype,
        {num_slots, num_kv_heads, head_dim}
    );
}

KVCache KVCache::create(const ModelConfig &config, int num_blocks, int block_size)
{
    const std::size_t kv_cache_bytes = num_blocks * block_bytes(config, block_size);
    CudaDeviceBuffer data(kv_cache_bytes);
    void* device_ptr = data.data();
    return KVCache{
        .data = std::move(data),
        .view = {
            .data = device_ptr,
            .dtype = config.dtype,
            .num_layers = config.num_hidden_layers,
            .num_blocks = num_blocks,
            .block_size = block_size,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim
        }
    };
}

std::size_t KVCache::block_bytes(const ModelConfig &config, int block_size) noexcept
{
    return static_cast<std::size_t>(2)
        * config.num_hidden_layers
        * block_size
        * config.num_kv_heads
        * config.head_dim
        * dtype_byte_size(config.dtype);
}


}