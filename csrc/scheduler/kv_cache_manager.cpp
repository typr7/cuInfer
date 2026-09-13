#include <numeric>
#include <cassert>

#include "scheduler/kv_cache_manager.h"


namespace cuinfer
{

KVCacheManager::KVCacheManager(const KVCacheConfig& cfg)
    : num_block_slots_(cfg.num_block_slots),
      num_blocks_(cfg.num_blocks),
      free_blocks_(cfg.num_blocks)
{
    std::iota(free_blocks_.begin(), free_blocks_.end(), 0);
}

bool KVCacheManager::allocate_slots(
    const Request& request, int num_scheduled_tokens
)
{
    assert(num_scheduled_tokens > 0);

    int num_allocated_blocks = 0;
    auto allocated_iter = req_to_allocated_blocks_.find(request.id);
    if (allocated_iter != req_to_allocated_blocks_.end()) {
        num_allocated_blocks = static_cast<int>(allocated_iter->second.size());
    }

    int num_required_slots = request.num_computed_tokens + num_scheduled_tokens;
    int num_required_blocks = (
        (num_required_slots + num_block_slots_ - 1) / num_block_slots_
    );

    assert(num_required_blocks >= num_allocated_blocks);
    if (num_required_blocks == num_allocated_blocks) {
        return true;
    }

    int num_blocks_to_allocate = num_required_blocks - num_allocated_blocks;
    if (num_blocks_to_allocate > static_cast<int>(free_blocks_.size())) {
        return false;
    }

    std::vector<int>& allocated_blocks = req_to_allocated_blocks_[request.id];
    auto first_block = free_blocks_.end() - num_blocks_to_allocate;

    allocated_blocks.reserve(allocated_blocks.size() + num_blocks_to_allocate);
    allocated_blocks.insert(
        allocated_blocks.end(), first_block, free_blocks_.end()
    );
    free_blocks_.erase(first_block, free_blocks_.end());

    return true;
}

void KVCacheManager::release_blocks(const std::string& request_id)
{
    std::vector<int>& assigned_blocks = req_to_allocated_blocks_.at(request_id);

    free_blocks_.insert(
        free_blocks_.end(),
        assigned_blocks.begin(),
        assigned_blocks.end()
    );

    req_to_allocated_blocks_.erase(request_id);
}

std::vector<int> KVCacheManager::get_allocated_blocks(const std::string& request_id) const
{
    return req_to_allocated_blocks_.at(request_id);
}

}
