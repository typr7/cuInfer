#pragma once

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/config.h"
#include "scheduler/kv_cache_manager.h"
#include "scheduler/request.h"
#include "engine/protocol.h"
#include "model/model_config.h"


namespace cuinfer
{

using RequestList = std::list<Request>;
using RequestIterator = RequestList::iterator;

// enable: contiunous batching, chunked prefill, kv cache blocks manage, FCFS, preemptive scheduling
// disable: spec decoding, prefix caching, priority scheduling, async scheduling
class Scheduler
{
public:
    Scheduler(
        const Config& cfg,
        const ModelConfig& model_config,
        const KVCacheConfig& kv_cache_config
    );
    ~Scheduler() noexcept = default;

    void add_request(Request request);

    void abort_requests(const std::vector<std::string>& request_ids);

    bool has_requests() const noexcept;

    std::vector<ScheduledRequest> schedule();

    EngineCoreOutputs update(const std::vector<SampledToken>& sampled);

private:
    void remove_request(RequestIterator request_iter);

    RequestList& queue_of(RequestStatus status) noexcept;

    FinishReason finish_reason(const Request& request, bool eos_token) const noexcept;

private:
    RequestList running_;
    RequestList waiting_;
    RequestList preempted_;

    std::unordered_map<std::string, RequestIterator> id_to_request_;

    KVCacheManager kv_cache_manager_;

    int max_num_seqs_;
    int max_num_scheduled_tokens_;
    int max_model_len_;
};

}
