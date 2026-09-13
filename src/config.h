#pragma once

#include <string>


namespace cllm
{

struct Config
{
    // path to local Hugging Face repo, which contains model.safetensors and config.json
    std::string model_path;
    float gpu_memory_utilization = 0.8;
    int block_size = 16;

    // Token budget for one scheduler step. Also the batch size the model runner
    // profiles at, since no step can ever exceed it.
    int max_num_scheduled_tokens = 8192;

    // Upper bound on requests running concurrently. Bounds the per-step batch
    // metadata the model runner preallocates.
    int max_num_seqs = 256;
};

}
