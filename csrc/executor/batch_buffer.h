#pragma once

#include "engine/config.h"
#include "model/model_config.h"
#include "cuda/pinned_buffer.h"
#include "cuda/cuda_device_buffer.h"
#include "executor/forward_batch.h"
#include "executor/model_input.h"
#include "cuda/cuda_context.h"


namespace cllm
{

class BatchBuffer
{
public:
    static BatchBuffer create(const Config& config, const ModelConfig& model_config);

    ForwardBatch upload(const ModelInput& input, const CudaContext& context);
    void download_sampled_token_ids(int num_sampled_tokens, const CudaContext& context);
    const int* sampled_token_ids() const noexcept
    {
        return pinned_.data<int>();
    }

private:
    CudaDeviceBuffer device_;
    PinnedBuffer pinned_;
};

}
