#include <algorithm>
#include <cassert>

#include "executor/batch_buffer.h"
#include "common/util.h"
#include "cuda/cuda_utils.h"


namespace cuinfer
{

namespace
{

constexpr std::size_t kAlignment = 32;

template <typename T>
std::size_t reserve(std::size_t& buffer_size, std::size_t size)
{
    const std::size_t aligned = align_up<kAlignment>(buffer_size);
    buffer_size = aligned + size * sizeof(T);
    return aligned;
}

}

BatchBuffer BatchBuffer::create(const Config &config, const ModelConfig &model_config)
{
    // Reserve fixed field offsets using the maximum input dimensions.
    const int max_tokens = config.max_num_scheduled_tokens;
    const int max_seqs = config.max_num_seqs;
    const int block_size = config.block_size;
    const int max_model_len = model_config.max_model_len;
    const int max_block_table_stride = (max_model_len + block_size - 1) / block_size;

    std::size_t buffer_size = 0;
    BatchBuffer buffer;
    buffer.offsets_ = {
        .token_ids = reserve<int>(buffer_size, max_tokens),
        .positions = reserve<int>(buffer_size, max_tokens),
        .slot_mapping = reserve<int>(buffer_size, max_tokens),
        .query_start_loc = reserve<int>(buffer_size, max_seqs + 1),
        .seq_lens = reserve<int>(buffer_size, max_seqs),
        .block_table = reserve<int>(buffer_size,
            static_cast<std::size_t>(max_seqs) * max_block_table_stride),
        .logits_indices = reserve<int>(buffer_size, max_seqs),
        .sample_params = reserve<SampleParams>(buffer_size, max_seqs),
        .num_tokens = reserve<int>(buffer_size, 1)
    };
    buffer.block_size_ = block_size;
    buffer.block_table_stride_ = max_block_table_stride;
    buffer.device_.resize(buffer_size);
    buffer.pinned_.resize(buffer_size);
    return buffer;
}

ForwardBatch BatchBuffer::upload(
    const std::vector<ScheduledRequest>& scheduled,
    const CudaContext& context,
    int padded_num_tokens
)
{
    auto* base = pinned_.data<std::byte>();
    auto* token_ids = reinterpret_cast<int*>(base + offsets_.token_ids);
    auto* positions = reinterpret_cast<int*>(base + offsets_.positions);
    auto* slot_mapping = reinterpret_cast<int*>(base + offsets_.slot_mapping);
    auto* query_start_loc = reinterpret_cast<int*>(base + offsets_.query_start_loc);
    auto* seq_lens = reinterpret_cast<int*>(base + offsets_.seq_lens);
    auto* block_table = reinterpret_cast<int*>(base + offsets_.block_table);
    auto* logits_indices = reinterpret_cast<int*>(base + offsets_.logits_indices);
    auto* sample_params = reinterpret_cast<SampleParams*>(base + offsets_.sample_params);

    int host_block_table_stride = 0;
    for (const ScheduledRequest& request : scheduled) {
        host_block_table_stride = std::max(
            host_block_table_stride, static_cast<int>(request.allocated_blocks.size())
        );
    }

    const int num_reqs = static_cast<int>(scheduled.size());
    int num_tokens = 0;
    int num_sampling_reqs = 0;
    int max_query_len = 0;
    int max_seq_len = 0;
    query_start_loc[0] = 0;
    for (int i = 0; i < num_reqs; i++) {
        const ScheduledRequest& request = scheduled[i];
        const int num_query_tokens = static_cast<int>(request.tokens_to_compute.size());
        assert(num_query_tokens > 0);

        std::copy(request.tokens_to_compute.begin(), request.tokens_to_compute.end(),
            token_ids + num_tokens);
        for (int j = 0; j < num_query_tokens; j++) {
            const int position = request.position_offset + j;
            const int block = request.allocated_blocks[position / block_size_];
            positions[num_tokens + j] = position;
            slot_mapping[num_tokens + j] = block * block_size_ + position % block_size_;
        }

        num_tokens += num_query_tokens;
        query_start_loc[i + 1] = num_tokens;
        seq_lens[i] = request.position_offset + num_query_tokens;
        max_query_len = std::max(max_query_len, num_query_tokens);
        max_seq_len = std::max(max_seq_len, seq_lens[i]);

        int* row = block_table + static_cast<std::size_t>(i) * host_block_table_stride;
        int* row_end = std::copy(request.allocated_blocks.begin(), request.allocated_blocks.end(), row);
        std::fill(row_end, row + host_block_table_stride, 0);

        if (request.needs_sampling) {
            logits_indices[num_sampling_reqs] = num_tokens - 1;
            sample_params[num_sampling_reqs] = request.sample_params;
            num_sampling_reqs++;
        }
    }

    if (padded_num_tokens == 0) {
        padded_num_tokens = num_tokens;
    }
    assert(padded_num_tokens >= num_tokens);
    std::fill(token_ids + num_tokens, token_ids + padded_num_tokens, 0);
    std::fill(positions + num_tokens, positions + padded_num_tokens, 0);
    std::fill(slot_mapping + num_tokens, slot_mapping + padded_num_tokens, -1);
    auto* token_count = reinterpret_cast<int*>(base + offsets_.num_tokens);
    *token_count = num_tokens;

    auto upload = [&]<typename T>(const T* data, std::size_t size, std::size_t offset) -> const T* {
        if (size > 0) {
            device_.upload_at_async(offset, data, size * sizeof(T), context.stream());
        }
        return reinterpret_cast<const T*>(device_.data<std::byte>() + offset);
    };

    ForwardBatch batch{
        .token_ids = upload(token_ids, padded_num_tokens, offsets_.token_ids),
        .positions = upload(positions, padded_num_tokens, offsets_.positions),
        .slot_mapping = upload(slot_mapping, padded_num_tokens, offsets_.slot_mapping),
        .query_start_loc = upload(query_start_loc, num_reqs + 1, offsets_.query_start_loc),
        .seq_lens = upload(seq_lens, num_reqs, offsets_.seq_lens),
        .block_table = reinterpret_cast<const int*>(device_.data<std::byte>() + offsets_.block_table),
        .logits_indices = upload(logits_indices, num_sampling_reqs, offsets_.logits_indices),
        .sample_params = upload(sample_params, num_sampling_reqs, offsets_.sample_params),
        .sampled_token_ids = device_.data<int>(),
        .num_tokens_device = upload(token_count, 1, offsets_.num_tokens),

        .block_table_stride = block_table_stride_,
        .num_tokens = num_tokens,
        .num_reqs = num_reqs,
        .num_sampling_reqs = num_sampling_reqs,
        .max_query_len = max_query_len,
        .max_seq_len = max_seq_len
    };

    if (num_reqs > 0) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            device_.data<std::byte>() + offsets_.block_table,
            static_cast<std::size_t>(block_table_stride_) * sizeof(int),
            block_table,
            static_cast<std::size_t>(host_block_table_stride) * sizeof(int),
            static_cast<std::size_t>(host_block_table_stride) * sizeof(int),
            num_reqs,
            cudaMemcpyHostToDevice,
            context.stream()
        ));
    }

    return batch;
}

void BatchBuffer::download_sampled_token_ids(
    int num_sampled_tokens,
    const CudaContext& context
)
{
    if (num_sampled_tokens == 0) {
        return;
    }

    device_.download_async(
        pinned_.data(),
        static_cast<std::size_t>(num_sampled_tokens) * sizeof(int),
        context.stream()
    );
}

}
