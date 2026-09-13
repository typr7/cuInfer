#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "engine/config.h"
#include "executor/model_input.h"
#include "scheduler/request.h"
#include "model/model_config.h"


namespace cllm
{

class ModelRunner
{
public:
    explicit ModelRunner(const Config& cfg);
    ~ModelRunner() noexcept;

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    std::size_t profile_available_kv_cache_memory(float gpu_memory_utilization);
    void allocate_kv_cache(int num_blocks);

    std::size_t kv_cache_block_bytes() const noexcept;
    ModelConfig model_config() const noexcept;

    void run_model(const std::vector<ScheduledRequest>& scheduled);
    std::vector<SampledToken> finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
