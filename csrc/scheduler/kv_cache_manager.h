#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "scheduler/request.h"


namespace cllm
{

struct KVCacheConfig
{
    int num_block_slots;
    int num_blocks;
};

class KVCacheManager
{
public:
    KVCacheManager(const KVCacheConfig& cfg);
    ~KVCacheManager() noexcept = default;

    bool allocate_slots(const Request& request, int num_scheduled_tokens);

    void release_blocks(const std::string& request_id);

    std::vector<int> get_allocated_blocks(const std::string& request_id) const;

private:
    int num_block_slots_;

    int num_blocks_;

    std::vector<int> free_blocks_;

    std::unordered_map<std::string, std::vector<int>> req_to_allocated_blocks_;
};

}