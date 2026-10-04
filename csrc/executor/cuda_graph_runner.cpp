#include <algorithm>
#include <cassert>

#include "executor/cuda_graph_runner.h"
#include "cuda/cuda_utils.h"


namespace cuinfer
{

CudaGraphRunner::CudaGraphRunner(int max_num_tokens, int num_layers)
    : num_layers_(num_layers)
{
    for (int size = 1; size < max_num_tokens;) {
        buckets_.push_back({size, std::vector<cudaGraphExec_t>(num_layers + 1, nullptr)});
        if (size > max_num_tokens / 2) {
            break;
        }
        size *= 2;
    }
    buckets_.push_back({max_num_tokens, std::vector<cudaGraphExec_t>(num_layers + 1, nullptr)});
}

CudaGraphRunner::~CudaGraphRunner() noexcept
{
    clear();
}

void CudaGraphRunner::clear() noexcept
{
    for (Bucket& bucket : buckets_) {
        for (cudaGraphExec_t& segment : bucket.segments) {
            if (segment != nullptr) {
                cudaGraphExecDestroy(segment);
                segment = nullptr;
            }
        }
    }
}

int CudaGraphRunner::bucket_size(int num_tokens) const
{
    const auto bucket = std::lower_bound(
        buckets_.begin(), buckets_.end(), num_tokens,
        [](const Bucket& bucket, int tokens) { return bucket.num_tokens < tokens; }
    );
    assert(bucket != buckets_.end());
    return bucket->num_tokens;
}

void CudaGraphRunner::capture(
    const CudaContext& context,
    const CausalLM& model,
    const ForwardBatch& batch,
    const KVCacheView& kv_cache,
    Workspace& workspace
)
{
    assert(batch.num_tokens == 0 && batch.num_tokens_device != nullptr);
    context.synchronize();
    clear();
    workspace.zero(context);

    for (Bucket& bucket : buckets_) {
        const WorkspaceView view = workspace.view(bucket.num_tokens, 0);
        if (!warmed_up_) {
            for (int segment = 0; segment <= num_layers_; segment++) {
                model.forward_graph_segment(context, batch, kv_cache, view, segment);
            }
            context.synchronize();
        }

        for (int segment = 0; segment <= num_layers_; segment++) {
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(context.stream(), cudaStreamCaptureModeThreadLocal));
            try {
                model.forward_graph_segment(context, batch, kv_cache, view, segment);
                CUDA_CHECK(cudaStreamEndCapture(context.stream(), &graph));
            } catch (...) {
                cudaStreamEndCapture(context.stream(), &graph);
                if (graph != nullptr) {
                    cudaGraphDestroy(graph);
                }
                throw;
            }
            const cudaError_t status = cudaGraphInstantiate(&bucket.segments[segment], graph, 0);
            cudaGraphDestroy(graph);
            CUDA_CHECK(status);
            CUDA_CHECK(cudaGraphUpload(bucket.segments[segment], context.stream()));
        }
    }
    context.synchronize();
    warmed_up_ = true;
}

void CudaGraphRunner::forward(
    const CudaContext& context,
    const CausalLM& model,
    const ForwardBatch& batch,
    const KVCacheView& kv_cache,
    const Workspace& workspace
) const
{
    const auto bucket = std::lower_bound(
        buckets_.begin(), buckets_.end(), batch.num_tokens,
        [](const Bucket& bucket, int tokens) { return bucket.num_tokens < tokens; }
    );
    assert(bucket != buckets_.end());
    const WorkspaceView view = workspace.view(batch.num_tokens, batch.num_sampling_reqs);
    CUDA_CHECK(cudaGraphLaunch(bucket->segments[0], context.stream()));
    for (int layer = 0; layer < num_layers_; layer++) {
        model.forward_attention(context, batch, kv_cache, view, layer);
        CUDA_CHECK(cudaGraphLaunch(bucket->segments[layer + 1], context.stream()));
    }
}

}
