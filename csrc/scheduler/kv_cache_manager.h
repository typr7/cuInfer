#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "scheduler/request.h"


namespace cuinfer
{

struct KVCacheConfig
{
    int num_block_slots;
    int num_blocks;
    bool enable_prefix_caching;
};

// Counters behind the engine's prefix cache log line.
struct PrefixCacheStats
{
    int num_queries = 0;        // requests scheduled after a cache lookup
    int num_hits = 0;           // of those, requests that reused at least one block
    int num_reused_tokens = 0;  // prompt tokens served from the cache
    int num_cached_blocks = 0;  // blocks currently stored under a hash
    int num_blocks = 0;
};

class KVCacheManager
{
public:
    explicit KVCacheManager(const KVCacheConfig& cfg);
    ~KVCacheManager() noexcept = default;

    int reuse_prefix(const Request& request);

    // counts a request that looked up the cache and was then scheduled: it
    // reused num_reused_tokens of its prompt, or none of it.
    void record_prefix_reuse(int num_reused_tokens);

    bool allocate_slots(const Request& request, int num_scheduled_tokens);

    // returns the request's blocks and publishes the ones it filled since the
    // last call, so they can be reused.
    void release_blocks(const Request& request);

    std::vector<int> get_allocated_blocks(const std::string& request_id) const;

    PrefixCacheStats prefix_cache_stats() const;

private:
    struct Block
    {
        int ref_count = 0;

        int prev_free = -1;
        int next_free = -1;

        // valid once the block is full and its kv cache has been written.
        std::size_t hash = 0;
        bool hash_valid = false;

        bool cached = false;
    };

    void push_free(int block_id) noexcept;
    int pop_free() noexcept;
    void remove_free(int block_id) noexcept;

    int take_block() noexcept;

    void hold_block(int block_id) noexcept;

    void cache_full_blocks(const Request& request, const std::vector<int>& allocated_blocks);
    void insert_hash(std::size_t hash, int block_id);

private:
    int num_block_slots_;

    bool enable_prefix_caching_;

    std::vector<Block> blocks_;

    int free_head_ = -1;
    int free_tail_ = -1;
    int num_free_blocks_ = 0;

    std::unordered_map<std::size_t, int> hash_to_block_;

    std::unordered_map<std::string, std::vector<int>> req_to_allocated_blocks_;

    PrefixCacheStats stats_;
};

}
