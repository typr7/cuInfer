#pragma once

#include <string>
#include <vector>

#include "scheduler/sample_params.h"


namespace cllm
{

enum class RequestStatus
{
    kWaiting,
    kRunning,
    kPreempted,
};

struct Request
{
    std::string id;

    RequestStatus status;

    int num_prompt_tokens;

    // for preempted request that has decoded token,
    // num_prefill_tokens = num_prompt_tokens + num_decoded_tokens_before_preempted
    int num_prefill_tokens;

    // the number of tokens that have been computed, have kv caches stored
    int num_computed_tokens;

    int max_output_tokens;

    // prompt tokens followed by every sampled token
    std::vector<int> token_ids;

    SampleParams sample_params;
};

struct ScheduledRequest
{
    std::string request_id;

    bool is_prefill;

    // False for every chunk of a chunked prefill except the last one. Only the
    // final chunk produces a token, so only its logits are worth computing.
    bool needs_sampling;

    int position_offset;

    std::vector<int> tokens_to_compute;
    std::vector<int> allocated_blocks;

    SampleParams sample_params;
};

struct SampledToken
{
    std::string request_id;
    int token_id;
    bool eos_token;
};

}
