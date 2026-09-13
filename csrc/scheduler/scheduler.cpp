#include <cassert>
#include <algorithm>
#include <iterator>

#include "scheduler/scheduler.h"


namespace cuinfer
{

Scheduler::Scheduler(
    const Config& config,
    const ModelConfig& model_config,
    const KVCacheConfig& kv_cache_config
)
    : kv_cache_manager_(kv_cache_config),
      max_num_seqs_(config.max_num_seqs),
      max_num_scheduled_tokens_(config.max_num_scheduled_tokens),
      max_model_len_(model_config.max_model_len)
{
}

void Scheduler::add_request(Request request)
{
    int num_tokens = static_cast<int>(request.token_ids.size());
    request.status = RequestStatus::kWaiting;
    request.num_prompt_tokens = num_tokens;
    request.num_prefill_tokens = num_tokens;
    request.num_computed_tokens = 0;
    auto iter = waiting_.insert(waiting_.end(), std::move(request));
    id_to_request_.emplace(iter->id, iter);
}

void Scheduler::abort_requests(const std::vector<std::string>& request_ids)
{
    for (const auto& id: request_ids) {
        auto found = id_to_request_.find(id);
        if (found != id_to_request_.end()) {
            remove_request(found->second);
        }
    }
}

bool Scheduler::has_requests() const noexcept
{
    return !id_to_request_.empty();
}

std::vector<ScheduledRequest> Scheduler::schedule()
{
    std::vector<ScheduledRequest> scheduled;

    int token_budget = max_num_scheduled_tokens_;

    // pass 1: running
    std::size_t num_preempted_before = preempted_.size();
    auto cur_iter = running_.begin();
    while (cur_iter != running_.end() && token_budget > 0 && scheduled.size() < max_num_seqs_) {
        int num_tokens = static_cast<int>(cur_iter->token_ids.size());
        int num_scheduled_tokens = num_tokens - cur_iter->num_computed_tokens;
        num_scheduled_tokens = std::min(num_scheduled_tokens, token_budget);
        assert(num_scheduled_tokens > 0);

        // try assgin kv cache blocks to new tokens
        bool allocated = false;
        while (true) {
            if (kv_cache_manager_.allocate_slots(*cur_iter, num_scheduled_tokens)) {
                allocated = true;
                break;
            }

            auto preempted_iter = std::prev(running_.end());

            kv_cache_manager_.release_blocks(preempted_iter->id);
            preempted_iter->num_computed_tokens = 0;
            preempted_iter->status = RequestStatus::kPreempted;

            preempted_.splice(preempted_.begin(), running_, preempted_iter);

            if (preempted_iter->id == cur_iter->id) {
                break;
            }
        }

        if (!allocated) {
            break;
        }

        token_budget -= num_scheduled_tokens;

        const auto compute_begin = cur_iter->token_ids.begin() + cur_iter->num_computed_tokens;
        const int num_computed_after = cur_iter->num_computed_tokens + num_scheduled_tokens;
        scheduled.push_back({
            .request_id = cur_iter->id,
            .is_prefill = cur_iter->num_computed_tokens < cur_iter->num_prefill_tokens,
            .needs_sampling = num_computed_after >= cur_iter->num_prefill_tokens,
            .position_offset = cur_iter->num_computed_tokens,
            .tokens_to_compute = std::vector<int>(
                compute_begin,
                compute_begin + num_scheduled_tokens
            ),
            .allocated_blocks = kv_cache_manager_.get_allocated_blocks(cur_iter->id),
            .sample_params = cur_iter->sample_params
        });
        cur_iter->num_computed_tokens = num_computed_after;

        cur_iter++;
    }

    if (preempted_.size() == num_preempted_before) {
        // pass 2: preempted and waiting
        while (
            (!preempted_.empty() || !waiting_.empty())
            && token_budget > 0
            && scheduled.size() < max_num_seqs_
        ) {
            auto& request_queue = preempted_.empty() ? waiting_ : preempted_;
            auto cur_iter = request_queue.begin();

            int num_tokens = static_cast<int>(cur_iter->token_ids.size());
            int num_scheduled_tokens = std::min(num_tokens, token_budget);
            assert(num_scheduled_tokens > 0);

            if (!kv_cache_manager_.allocate_slots(*cur_iter, num_scheduled_tokens)) {
                break;
            }

            cur_iter->num_prefill_tokens = static_cast<int>(cur_iter->token_ids.size());
            token_budget -= num_scheduled_tokens;

            scheduled.push_back({
                .request_id = cur_iter->id,
                .is_prefill = true,
                .needs_sampling = num_scheduled_tokens >= cur_iter->num_prefill_tokens,
                .position_offset = 0,
                .tokens_to_compute = std::vector<int>(
                    cur_iter->token_ids.begin(),
                    cur_iter->token_ids.begin() + num_scheduled_tokens
                ),
                .allocated_blocks = kv_cache_manager_.get_allocated_blocks(cur_iter->id),
                .sample_params = cur_iter->sample_params
            });
            cur_iter->num_computed_tokens = num_scheduled_tokens;
            cur_iter->status = RequestStatus::kRunning;

            running_.splice(running_.end(), request_queue, cur_iter);
        }
    }

    return scheduled;
}

EngineCoreOutputs Scheduler::update(const std::vector<SampledToken>& sampled)
{
    EngineCoreOutputs outputs;
    outputs.reserve(sampled.size());

    for (const auto& s: sampled) {
        auto req = id_to_request_.at(s.request_id);
        req->token_ids.push_back(s.token_id);

        FinishReason reason = finish_reason(*req, s.eos_token);
        outputs.push_back({
            .request_id = s.request_id,
            .new_token_ids = {s.token_id},
            .finish_reason = reason
        });

        if (reason != FinishReason::kRunning) {
            remove_request(req);
        }
    }

    return outputs;
}

void Scheduler::remove_request(RequestIterator request_iter)
{
    // Only running requests hold kv cache blocks; preemption already released
    // them and waiting requests never had any.
    if (request_iter->status == RequestStatus::kRunning) {
        kv_cache_manager_.release_blocks(request_iter->id);
    }
    id_to_request_.erase(request_iter->id);
    queue_of(request_iter->status).erase(request_iter);
}

RequestList& Scheduler::queue_of(RequestStatus status) noexcept
{
    switch (status) {
        case RequestStatus::kRunning:
            return running_;
        case RequestStatus::kPreempted:
            return preempted_;
        case RequestStatus::kWaiting:
            break;
    }
    return waiting_;
}

FinishReason Scheduler::finish_reason(const Request& request, bool eos_token) const noexcept
{
    if (eos_token) {
        return FinishReason::kStop;
    }

    int num_tokens = static_cast<int>(request.token_ids.size());
    int num_output_tokens = num_tokens - request.num_prompt_tokens;
    if (num_output_tokens >= request.max_output_tokens || num_tokens >= max_model_len_) {
        return FinishReason::kLength;
    }

    return FinishReason::kRunning;
}

}
