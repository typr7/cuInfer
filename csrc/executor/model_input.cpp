#include <algorithm>
#include <cassert>

#include "executor/model_input.h"


namespace cllm
{

ModelInput prepare_model_input(
    const std::vector<ScheduledRequest>& scheduled,
    int block_size
)
{
    assert(block_size > 0);

    ModelInput input;

    int num_tokens = 0;
    int max_num_blocks = 0;
    for (const ScheduledRequest& request : scheduled) {
        num_tokens += static_cast<int>(request.tokens_to_compute.size());
        max_num_blocks = std::max(
            max_num_blocks, static_cast<int>(request.allocated_blocks.size())
        );
    }

    const int num_reqs = static_cast<int>(scheduled.size());

    input.token_ids.reserve(num_tokens);
    input.positions.reserve(num_tokens);
    input.slot_mapping.reserve(num_tokens);
    input.query_start_loc.reserve(num_reqs + 1);
    input.seq_lens.reserve(num_reqs);
    input.block_table.assign(
        static_cast<std::size_t>(num_reqs) * max_num_blocks, 0
    );
    input.block_table_stride = max_num_blocks;

    input.query_start_loc.push_back(0);

    for (int i = 0; i < num_reqs; i++) {
        const ScheduledRequest& request = scheduled[i];
        const int num_query_tokens = static_cast<int>(request.tokens_to_compute.size());
        assert(num_query_tokens > 0);

        input.token_ids.insert(
            input.token_ids.end(),
            request.tokens_to_compute.begin(),
            request.tokens_to_compute.end()
        );

        for (int j = 0; j < num_query_tokens; j++) {
            const int position = request.position_offset + j;
            const int block = request.allocated_blocks[position / block_size];

            input.positions.push_back(position);
            input.slot_mapping.push_back(block * block_size + position % block_size);
        }

        input.query_start_loc.push_back(static_cast<int>(input.token_ids.size()));
        input.seq_lens.push_back(request.position_offset + num_query_tokens);

        std::copy(
            request.allocated_blocks.begin(),
            request.allocated_blocks.end(),
            input.block_table.begin() + static_cast<std::size_t>(i) * max_num_blocks
        );

        if (request.needs_sampling) {
            input.logits_indices.push_back(static_cast<int>(input.token_ids.size()) - 1);
            input.sample_params.push_back(request.sample_params);
            input.sampling_request_indices.push_back(i);
        }
    }

    return input;
}

}
