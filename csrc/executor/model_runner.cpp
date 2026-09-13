#include <algorithm>
#include <cassert>

#include "executor/model_runner.h"
#include "cuda/cuda_context.h"
#include "cuda/cuda_utils.h"
#include "model/model_config.h"
#include "executor/forward_batch.h"
#include "executor/workspace.h"
#include "executor/batch_buffer.h"
#include "model/causal_lm.h"
#include "executor/sampler.h"


namespace cllm
{

struct ModelRunner::Impl
{
    explicit Impl(const Config& config)
        : config(config),
          model_config(ModelConfig::load(config.model_path)),
          model(model_config, ModelWeights::load_from_safetensors(
              std::filesystem::path(config.model_path) / "model.safetensors",
              model_config
          )),
          sampler(config.max_num_seqs, model_config.vocab_size),
          workspace(Workspace::create(
              model_config,
              config.max_num_scheduled_tokens,
              config.max_num_seqs
          )),
          batch_buffer(BatchBuffer::create(config, model_config))
        {
        }

    void allocate_kv_cache(int num_blocks)
    {
        kv_cache = KVCache::create(model_config, num_blocks, config.block_size);
    }

    void run_model(const std::vector<ScheduledRequest>& scheduled)
    {
        const ModelInput input = prepare_model_input(scheduled, config.block_size);
        const ForwardBatch batch = batch_buffer.upload(input, context);
        const WorkspaceView workspace_view = workspace.view(batch.num_tokens, batch.num_sampling_reqs);

        model.forward(context, batch, kv_cache.view, workspace_view);
        model.compute_logits(context, batch, workspace_view);
        sampler.sample(context, workspace_view.logits, batch);
        batch_buffer.download_sampled_token_ids(batch.num_sampling_reqs, context);

        inflight.clear();
        inflight.reserve(input.sampling_request_indices.size());
        for (int idx: input.sampling_request_indices) {
            inflight.push_back({
                .request_id = scheduled[idx].request_id,
                .token_id = -1,
                .eos_token = false
            });
        }
    }

    std::vector<SampledToken> finish()
    {
        context.synchronize();

        const int* sampled_token_ids = batch_buffer.sampled_token_ids();
        for (int i = 0; i < inflight.size(); i++) {
            SampledToken& cur = inflight[i];
            int token_id = sampled_token_ids[i];
            cur.token_id = token_id;
            cur.eos_token = is_eos_token(token_id);
        }

        return std::exchange(inflight, {});
    }

    bool is_eos_token(int token_id) const noexcept
    {
        return std::any_of(
            model_config.eos_token_ids.begin(),
            model_config.eos_token_ids.end(),
            [token_id](int eos) {
                return token_id == eos;
            }
        );
    }

    Config config;

    ModelConfig model_config;

    CudaContext context;

    CausalLM model;

    Sampler sampler;

    // device buffer for activation
    Workspace workspace;

    // Per-step H2D inputs and sampled token D2H output.
    BatchBuffer batch_buffer;

    // allocated by allocate_kv_cache()
    KVCache kv_cache;

    std::vector<SampledToken> inflight;
};


// ModelRunner
ModelRunner::ModelRunner(const Config& cfg)
    : impl_(std::make_unique<Impl>(cfg))
{
}

ModelRunner::~ModelRunner() noexcept = default;

std::size_t ModelRunner::profile_available_kv_cache_memory(float gpu_memory_utilization)
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    const std::size_t used_bytes = total_bytes - free_bytes;
    const auto budget = static_cast<std::size_t>(total_bytes * gpu_memory_utilization);
    return budget > used_bytes ? budget - used_bytes : 0;
}

void ModelRunner::allocate_kv_cache(int num_blocks)
{
    assert(num_blocks > 0);
    impl_->allocate_kv_cache(num_blocks);
}

std::size_t ModelRunner::kv_cache_block_bytes() const noexcept
{
    return KVCache::block_bytes(impl_->model_config, impl_->config.block_size);
}

ModelConfig ModelRunner::model_config() const noexcept
{
    return impl_->model_config;
}

void ModelRunner::run_model(const std::vector<ScheduledRequest>& scheduled)
{
    if (scheduled.empty()) {
        return;
    }

    impl_->run_model(scheduled);
}

std::vector<SampledToken> ModelRunner::finish()
{
    return impl_->finish();
}

}
