#pragma once

#include <cstdint>

#include "cuda/cuda_context.h"
#include "cuda/cuda_device_buffer.h"
#include "executor/forward_batch.h"
#include "tensor/tensor.h"


namespace cuinfer
{

class Sampler
{
public:
    Sampler(int max_num_seqs, int vocab_size);

    void sample(
        const CudaContext& context,
        TensorRef<2> logits,
        const ForwardBatch& batch
    );

private:
    CudaDeviceBuffer workspace_;
    std::uint64_t seed_;
    std::uint64_t offset_ = 0;
};

}
