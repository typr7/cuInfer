#pragma once


namespace cllm
{

struct SampleParams;

struct ForwardBatch
{
    const int* token_ids;        // [num_tokens]
    const int* positions;        // [num_tokens]
    const int* slot_mapping;     // [num_tokens]
    const int* query_start_loc;  // [num_reqs + 1]
    const int* seq_lens;         // [num_reqs]
    const int* block_table;      // [num_reqs, block_table_stride]
    const int* logits_indices;   // [num_sampling_reqs]
    const SampleParams* sample_params; // [num_sampling_reqs]
    int* sampled_token_ids;      // [num_sampling_reqs], aliases token_ids after embedding

    int block_table_stride;
    int num_tokens;
    int num_reqs;
    int num_sampling_reqs;
    int max_query_len;
    int max_seq_len;
};

}