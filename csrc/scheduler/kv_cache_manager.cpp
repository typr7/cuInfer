#include <algorithm>
#include <cassert>
#include <cstdint>

#include "scheduler/kv_cache_manager.h"


namespace cuinfer
{

namespace
{

constexpr std::size_t kHashSeed = 0x9e3779b97f4a7c15ULL;

// splitmix64's finalizer, which spreads every token over the whole hash.
std::size_t mix(std::size_t value) noexcept
{
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::size_t block_hash(std::size_t parent, const int* tokens, int count) noexcept
{
    std::size_t hash = parent;
    for (int i = 0; i < count; i++) {
        hash = mix(hash ^ static_cast<std::uint32_t>(tokens[i]));
    }
    return hash;
}

}

KVCacheManager::KVCacheManager(const KVCacheConfig& cfg)
    : num_block_slots_(cfg.num_block_slots),
      enable_prefix_caching_(cfg.enable_prefix_caching),
      blocks_(cfg.num_blocks)
{
    for (int block_id = 0; block_id < cfg.num_blocks; block_id++) {
        push_free(block_id);
    }
}

void KVCacheManager::push_free(int block_id) noexcept
{
    Block& block = blocks_[block_id];
    assert(block.ref_count == 0);

    block.prev_free = free_tail_;
    block.next_free = -1;
    if (free_tail_ == -1) {
        free_head_ = block_id;
    } else {
        blocks_[free_tail_].next_free = block_id;
    }
    free_tail_ = block_id;
    num_free_blocks_++;
}

int KVCacheManager::pop_free() noexcept
{
    assert(free_head_ != -1);

    const int block_id = free_head_;
    Block& block = blocks_[block_id];
    free_head_ = block.next_free;
    if (free_head_ == -1) {
        free_tail_ = -1;
    } else {
        blocks_[free_head_].prev_free = -1;
    }
    block.prev_free = -1;
    block.next_free = -1;
    num_free_blocks_--;

    return block_id;
}

void KVCacheManager::remove_free(int block_id) noexcept
{
    Block& block = blocks_[block_id];
    assert(block.ref_count == 0);

    if (block.prev_free == -1) {
        free_head_ = block.next_free;
    } else {
        blocks_[block.prev_free].next_free = block.next_free;
    }
    if (block.next_free == -1) {
        free_tail_ = block.prev_free;
    } else {
        blocks_[block.next_free].prev_free = block.prev_free;
    }
    block.prev_free = -1;
    block.next_free = -1;
    num_free_blocks_--;
}

// The new tokens overwrite the block, so the content it was cached under is gone.
int KVCacheManager::take_block() noexcept
{
    const int block_id = pop_free();
    Block& block = blocks_[block_id];

    if (block.cached) {
        hash_to_block_.erase(block.hash);
        block.cached = false;
    }
    block.hash_valid = false;
    block.ref_count = 1;

    return block_id;
}

void KVCacheManager::hold_block(int block_id) noexcept
{
    Block& block = blocks_[block_id];
    if (block.ref_count == 0) {
        remove_free(block_id);
    }
    block.ref_count++;
}

void KVCacheManager::cache_full_blocks(
    const Request& request, const std::vector<int>& allocated_blocks
)
{
    if (!enable_prefix_caching_) {
        return;
    }

    const int num_full_blocks = std::min(
        static_cast<int>(allocated_blocks.size()),
        request.num_computed_tokens / num_block_slots_
    );

    int first = 0;
    while (first < num_full_blocks && blocks_[allocated_blocks[first]].hash_valid) {
        first++;
    }
    if (first == num_full_blocks) {
        return;
    }

    std::size_t hash = first == 0 ? kHashSeed : blocks_[allocated_blocks[first - 1]].hash;
    const int* tokens = request.token_ids.data();
    for (int i = first; i < num_full_blocks; i++) {
        hash = block_hash(
            hash,
            tokens + static_cast<std::ptrdiff_t>(i) * num_block_slots_,
            num_block_slots_
        );
        insert_hash(hash, allocated_blocks[i]);
    }
}

void KVCacheManager::insert_hash(std::size_t hash, int block_id)
{
    Block& block = blocks_[block_id];
    block.hash = hash;
    block.hash_valid = true;

    auto found = hash_to_block_.find(hash);
    if (found == hash_to_block_.end()) {
        hash_to_block_.emplace(hash, block_id);
        block.cached = true;
        return;
    }

    if (blocks_[found->second].ref_count == 0) {
        blocks_[found->second].cached = false;
        found->second = block_id;
        block.cached = true;
    }
}

int KVCacheManager::reuse_prefix(const Request& request)
{
    if (!enable_prefix_caching_) {
        return 0;
    }

    const std::vector<int>& token_ids = request.token_ids;

    const int num_shared_blocks = (
        (static_cast<int>(token_ids.size()) - 1) / num_block_slots_
    );

    std::vector<int> matched_blocks;
    std::size_t hash = kHashSeed;
    for (int i = 0; i < num_shared_blocks; i++) {
        hash = block_hash(
            hash,
            token_ids.data() + static_cast<std::ptrdiff_t>(i) * num_block_slots_,
            num_block_slots_
        );

        auto found = hash_to_block_.find(hash);
        if (found == hash_to_block_.end()) {
            break;
        }
        matched_blocks.push_back(found->second);
    }

    if (matched_blocks.empty()) {
        return 0;
    }

    std::vector<int>& allocated_blocks = req_to_allocated_blocks_[request.id];
    for (int block_id: matched_blocks) {
        hold_block(block_id);
        allocated_blocks.push_back(block_id);
    }

    return static_cast<int>(matched_blocks.size()) * num_block_slots_;
}

void KVCacheManager::record_prefix_reuse(int num_reused_tokens)
{
    if (!enable_prefix_caching_) {
        return;
    }

    stats_.num_queries++;
    if (num_reused_tokens > 0) {
        stats_.num_hits++;
        stats_.num_reused_tokens += num_reused_tokens;
    }
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
    const int num_blocks_to_allocate = num_required_blocks - num_allocated_blocks;
    if (num_blocks_to_allocate > num_free_blocks_) {
        return false;
    }

    std::vector<int>& allocated_blocks = req_to_allocated_blocks_[request.id];
    for (int i = 0; i < num_blocks_to_allocate; i++) {
        allocated_blocks.push_back(take_block());
    }

    cache_full_blocks(request, allocated_blocks);

    return true;
}

void KVCacheManager::release_blocks(const Request& request)
{
    auto allocated_iter = req_to_allocated_blocks_.find(request.id);
    if (allocated_iter == req_to_allocated_blocks_.end()) {
        return;
    }

    // The request is about to lose its blocks, so this is the last chance to
    // store the full ones it computed.
    cache_full_blocks(request, allocated_iter->second);

    for (int block_id: allocated_iter->second) {
        Block& block = blocks_[block_id];
        assert(block.ref_count > 0);

        block.ref_count--;
        if (block.ref_count == 0) {
            push_free(block_id);
        }
    }

    req_to_allocated_blocks_.erase(allocated_iter);
}

std::vector<int> KVCacheManager::get_allocated_blocks(const std::string& request_id) const
{
    return req_to_allocated_blocks_.at(request_id);
}

PrefixCacheStats KVCacheManager::prefix_cache_stats() const
{
    PrefixCacheStats stats = stats_;
    stats.num_cached_blocks = static_cast<int>(std::count_if(
        blocks_.begin(),
        blocks_.end(),
        [](const Block& block) {
            return block.cached;
        }
    ));
    stats.num_blocks = static_cast<int>(blocks_.size());

    return stats;
}

}
