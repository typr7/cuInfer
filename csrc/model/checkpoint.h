#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensor/data_type.h"


namespace cuinfer
{

struct HostTensor
{
    const std::byte* data;
    DataType dtype;
    std::vector<int> shape;
};

class Checkpoint
{
public:
    explicit Checkpoint(const std::filesystem::path& model_dir);
    ~Checkpoint() noexcept;

    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    bool contains(std::string_view name) const;
    const HostTensor& get(std::string_view name) const;
    const HostTensor& get(std::string_view name, DataType dtype, std::vector<int> shape) const;

private:
    void map_shard(const std::filesystem::path& path);

    std::vector<std::pair<void*, std::size_t>> mappings_;
    std::unordered_map<std::string, HostTensor> tensors_;
};

void upload_slice(const HostTensor& src, int dim, int begin, int count, void* dst);

}
