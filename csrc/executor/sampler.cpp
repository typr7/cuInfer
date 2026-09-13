#include <random>

#include "executor/sampler.h"
#include "ops/sampler.h"


namespace cuinfer
{

Sampler::Sampler(int max_num_seqs, int vocab_size)
    : workspace_(ops::sampler_workspace_size(max_num_seqs, vocab_size)),
      seed_(std::random_device{}())
{
}

void Sampler::sample(
    const CudaContext& context,
    TensorRef<2> logits,
    const ForwardBatch& batch
)
{
    const int num_sampling_reqs = batch.num_sampling_reqs;
    if (num_sampling_reqs == 0) {
        return;
    }

    ops::sample(
        logits,
        batch.sample_params,
        batch.sampled_token_ids,
        workspace_.data(),
        seed_,
        offset_,
        context.stream()
    );
    offset_ += num_sampling_reqs;
}

}
