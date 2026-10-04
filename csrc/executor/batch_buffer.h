#pragma once

#include <cstddef>
#include <vector>

#include "engine/config.h"
#include "model/model_config.h"
#include "cuda/pinned_buffer.h"
#include "cuda/cuda_device_buffer.h"
#include "executor/forward_batch.h"
#include "scheduler/request.h"
#include "cuda/cuda_context.h"


namespace cuinfer
{

class BatchBuffer
{
public:
    static BatchBuffer create(const Config& config, const ModelConfig& model_config);

    ForwardBatch upload(
        const std::vector<ScheduledRequest>& scheduled,
        const CudaContext& context,
        int padded_num_tokens = 0
    );
    void download_sampled_token_ids(int num_sampled_tokens, const CudaContext& context);
    const int* sampled_token_ids() const noexcept
    {
        return pinned_.data<int>();
    }

private:
    struct Offsets
    {
        std::size_t token_ids;
        std::size_t positions;
        std::size_t slot_mapping;
        std::size_t query_start_loc;
        std::size_t seq_lens;
        std::size_t block_table;
        std::size_t logits_indices;
        std::size_t sample_params;
        std::size_t num_tokens;
    };

    Offsets offsets_;
    int block_size_;
    int block_table_stride_;
    CudaDeviceBuffer device_;
    PinnedBuffer pinned_;
};

}
