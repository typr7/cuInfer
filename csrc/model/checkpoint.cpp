#include <cassert>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <format>
#include <set>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

#include "model/checkpoint.h"
#include "cuda/cuda_utils.h"


namespace cuinfer
{

namespace
{

constexpr std::size_t kMaxHeaderSize = 100'000'000;

DataType parse_dtype(std::string_view value)
{
    for (DataType dtype : {DataType::kBf16, DataType::kFp16, DataType::kFp32,
                           DataType::kInt32, DataType::kInt8}) {
        if (value == dtype_string(dtype)) {
            return dtype;
        }
    }
    throw std::runtime_error(std::format("unsupported safetensors dtype `{}`", value));
}

}

Checkpoint::Checkpoint(const std::filesystem::path& model_dir)
{
    try {
        const auto index_path = model_dir / "model.safetensors.index.json";
        if (std::filesystem::exists(index_path)) {
            std::ifstream file(index_path);
            const auto index = nlohmann::json::parse(file);
            const auto& weight_map = index.at("weight_map");
            std::set<std::string> shards;
            for (const auto& shard : weight_map) {
                shards.insert(shard.get<std::string>());
            }
            for (const auto& shard : shards) {
                map_shard(model_dir / shard);
            }
            for (const auto& [name, shard] : weight_map.items()) {
                get(name);
            }
        } else {
            map_shard(model_dir / "model.safetensors");
        }
    } catch (const std::exception& error) {
        for (const auto& [data, size] : mappings_) {
            munmap(data, size);
        }
        throw std::runtime_error(std::format(
            "failed to load checkpoint `{}`: {}", model_dir.string(), error.what()
        ));
    }
}

Checkpoint::~Checkpoint() noexcept
{
    for (const auto& [data, size] : mappings_) {
        munmap(data, size);
    }
}

void Checkpoint::map_shard(const std::filesystem::path& path)
{
    const std::size_t size = std::filesystem::file_size(path);
    if (size < sizeof(std::uint64_t)) {
        throw std::runtime_error(std::format("invalid safetensors file `{}`", path.string()));
    }
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error(std::format("cannot open `{}`", path.string()));
    }
    void* mapping = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(std::format("cannot mmap `{}`", path.string()));
    }
    mappings_.emplace_back(mapping, size);
    const auto* bytes = static_cast<const std::byte*>(mapping);
    std::uint64_t header_size;
    std::memcpy(&header_size, bytes, sizeof(header_size));
    if (header_size > kMaxHeaderSize || header_size > size - sizeof(header_size)) {
        throw std::runtime_error(std::format("invalid header size in `{}`", path.string()));
    }
    const auto* header_begin = reinterpret_cast<const char*>(bytes + sizeof(header_size));
    const auto header = nlohmann::json::parse(header_begin, header_begin + header_size);
    const std::size_t data_offset = sizeof(header_size) + header_size;
    for (const auto& [name, spec] : header.items()) {
        if (name == "__metadata__") {
            continue;
        }
        const DataType dtype = parse_dtype(spec.at("dtype").get<std::string>());
        auto shape = spec.at("shape").get<std::vector<int>>();
        std::size_t tensor_size = dtype_byte_size(dtype);
        for (int dim : shape) {
            if (dim < 0 || (dim != 0 && tensor_size > (size - data_offset) / dim)) {
                throw std::runtime_error(std::format("invalid shape for `{}`", name));
            }
            tensor_size *= dim;
        }
        const auto offsets = spec.at("data_offsets").get<std::vector<std::size_t>>();
        if (offsets.size() != 2 || offsets[1] < offsets[0]
            || offsets[1] > size - data_offset || offsets[1] - offsets[0] != tensor_size) {
            throw std::runtime_error(std::format("invalid data_offsets for `{}`", name));
        }
        if (!tensors_.emplace(name, HostTensor{
                bytes + data_offset + offsets[0], dtype, std::move(shape)
            }).second) {
            throw std::runtime_error(std::format("duplicate tensor `{}`", name));
        }
    }
}

bool Checkpoint::contains(std::string_view name) const
{
    return tensors_.contains(std::string(name));
}

const HostTensor& Checkpoint::get(std::string_view name) const
{
    const auto iter = tensors_.find(std::string(name));
    if (iter == tensors_.end()) {
        throw std::runtime_error(std::format("missing checkpoint tensor `{}`", name));
    }
    return iter->second;
}

const HostTensor& Checkpoint::get(std::string_view name, DataType dtype, std::vector<int> shape) const
{
    const HostTensor& tensor = get(name);
    if (tensor.dtype != dtype) {
        throw std::runtime_error(std::format(
            "mismatched dtype for `{}`: expected {}, got {}",
            name, dtype_string(dtype), dtype_string(tensor.dtype)
        ));
    }
    if (tensor.shape != shape) {
        throw std::runtime_error(std::format(
            "mismatched shape for `{}`: expected {}, got {}",
            name, nlohmann::json(shape).dump(), nlohmann::json(tensor.shape).dump()
        ));
    }
    return tensor;
}

void upload_slice(const HostTensor& src, int dim, int begin, int count, void* dst)
{
    assert(dim == 0 || (dim == 1 && src.shape.size() == 2));
    assert(begin >= 0 && count > 0 && begin + count <= src.shape[dim]);
    const std::size_t element_size = dtype_byte_size(src.dtype);
    if (dim == 0) {
        std::size_t row_bytes = element_size;
        for (std::size_t i = 1; i < src.shape.size(); i++) {
            row_bytes *= src.shape[i];
        }
        CUDA_CHECK(cudaMemcpy(dst, src.data + begin * row_bytes,
                              count * row_bytes, cudaMemcpyHostToDevice));
    } else {
        const std::size_t src_pitch = src.shape[1] * element_size;
        const std::size_t dst_pitch = count * element_size;
        CUDA_CHECK(cudaMemcpy2D(dst, dst_pitch, src.data + begin * element_size,
                                src_pitch, dst_pitch, src.shape[0], cudaMemcpyHostToDevice));
    }
}

}
