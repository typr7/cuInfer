#include <cassert>
#include <format>

#include <cuda_bf16.h>

#include "ops/embedding.h"
#include "tensor/tensor.h"
#include "ops/utils.h"
#include "cuda/cuda_utils.h"


namespace cllm::ops
{

namespace
{

template <uint32_t kEmbeddingDim>
__global__ __launch_bounds__(kEmbeddingDim / kNumBf16sPerVector)
void bf16_embedding(
    const int* __restrict__ token_ids,
    const nv_bfloat16* __restrict__ embedding_table,
    nv_bfloat16* __restrict__ output
)
{
    const auto token_id = static_cast<uint32_t>(token_ids[blockIdx.x]);
    const auto* input_u4
        = reinterpret_cast<const uint4*>(embedding_table + token_id * kEmbeddingDim);
    auto* output_u4 = reinterpret_cast<uint4*>(output + blockIdx.x * kEmbeddingDim);
    output_u4[threadIdx.x] = input_u4[threadIdx.x];
}

template <uint32_t kEmbeddingDim>
void launch_embedding(
    const int* token_ids,
    TensorRef<2> embedding_table,
    TensorRef<2> output,
    cudaStream_t stream
)
{
    constexpr uint32_t kNumThreads = kEmbeddingDim / kNumBf16sPerVector;

    static_assert(kEmbeddingDim % kNumBf16sPerVector == 0);

    const uint32_t num_tokens = static_cast<uint32_t>(output.shape[0]);
    bf16_embedding<kEmbeddingDim><<<num_tokens, kNumThreads, 0, stream>>>(
        token_ids,
        static_cast<const nv_bfloat16*>(embedding_table.device_ptr),
        static_cast<nv_bfloat16*>(output.device_ptr)
    );
    CUDA_CHECK(cudaGetLastError());
}

}

void embedding(
    const int* token_ids,
    TensorRef<2> embedding_table,
    TensorRef<2> output,
    cudaStream_t stream
)
{
    assert(token_ids != nullptr && embedding_table && output);

    const int embedding_dim = embedding_table.shape[1];
    switch (embedding_dim) {
        case 1024: {
            launch_embedding<1024>(token_ids, embedding_table, output, stream);
            break;
        }
        case 2048: {
            launch_embedding<2048>(token_ids, embedding_table, output, stream);
            break;
        }
        default:
            throw std::runtime_error(std::format(
                "unsupported embedding dimension: {}",
                embedding_dim
            ));
    }
}

}
