#pragma once

#include <vector>

#include "scheduler/request.h"


namespace cuinfer
{

struct ModelInput
{
    std::vector<int> token_ids;
    std::vector<int> positions;
    std::vector<int> slot_mapping;
    std::vector<int> query_start_loc;
    std::vector<int> seq_lens;
    std::vector<int> block_table;
    std::vector<int> logits_indices;
    std::vector<SampleParams> sample_params;

    // will not be passed to device
    std::vector<int> sampling_request_indices;

    int block_table_stride = 0;

    int num_tokens() const noexcept
    {
        return static_cast<int>(token_ids.size());
    }

    int num_reqs() const noexcept
    {
        return static_cast<int>(seq_lens.size());
    }
};

// Flattens a scheduled batch into the layout above. Pure host-side logic, kept
// out of ModelRunner so it can be tested without a GPU.
ModelInput prepare_model_input(
    const std::vector<ScheduledRequest>& scheduled,
    int block_size
);

}
