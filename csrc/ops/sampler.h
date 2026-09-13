#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "tensor/tensor.h"


namespace cllm
{

struct SampleParams;

namespace ops
{

std::size_t sampler_workspace_size(int num_reqs, int vocab_size);

void sample(
    TensorRef<2> logits,
    const SampleParams* params,
    int* output,
    void* workspace,
    std::uint64_t seed,
    std::uint64_t offset,
    cudaStream_t stream
);

}

}