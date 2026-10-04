#pragma once

#include <vector>

#include "model/causal_lm.h"


namespace cuinfer
{

class CudaGraphRunner
{
public:
    CudaGraphRunner(int max_num_tokens, int num_layers);
    ~CudaGraphRunner() noexcept;

    CudaGraphRunner(const CudaGraphRunner&) = delete;
    CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

    int bucket_size(int num_tokens) const;

    void capture(
        const CudaContext& context,
        const CausalLM& model,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        Workspace& workspace
    );

    void forward(
        const CudaContext& context,
        const CausalLM& model,
        const ForwardBatch& batch,
        const KVCacheView& kv_cache,
        const Workspace& workspace
    ) const;

private:
    struct Bucket
    {
        int num_tokens;
        std::vector<cudaGraphExec_t> segments;
    };

    void clear() noexcept;

    int num_layers_;
    bool warmed_up_ = false;
    std::vector<Bucket> buckets_;
};

}
