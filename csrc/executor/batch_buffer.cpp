#include <algorithm>

#include "executor/batch_buffer.h"
#include "common/util.h"


namespace cuinfer
{

namespace
{

constexpr std::size_t kAlignment = 32;

template <typename T>
void reserve(std::size_t& buffer_size, std::size_t size)
{
    const std::size_t aligned = align_up<kAlignment>(buffer_size);
    buffer_size = aligned + size * sizeof(T);
}

}

BatchBuffer BatchBuffer::create(const Config &config, const ModelConfig &model_config)
{
    // construct a max input to determine the buffer size
    const int max_tokens = config.max_num_scheduled_tokens;
    const int max_seqs = config.max_num_seqs;
    const int block_size = config.block_size;
    const int max_model_len = model_config.max_model_len;
    const int max_block_table_stride = (max_model_len + block_size - 1) / block_size;

    std::size_t buffer_size = 0;
    reserve<int>(buffer_size, max_tokens);
    reserve<int>(buffer_size, max_tokens);
    reserve<int>(buffer_size, max_tokens);
    reserve<int>(buffer_size, max_seqs + 1);
    reserve<int>(buffer_size, max_seqs);
    reserve<int>(buffer_size, static_cast<std::size_t>(max_seqs) * max_block_table_stride);
    reserve<int>(buffer_size, max_seqs);
    reserve<SampleParams>(buffer_size, max_seqs);

    BatchBuffer buffer;
    buffer.device_.resize(buffer_size);
    buffer.pinned_.resize(buffer_size);
    return buffer;
}

ForwardBatch BatchBuffer::upload(const ModelInput& input, const CudaContext& context)
{
    std::size_t upload_size = 0;
    auto upload_to_pinned = [&]<typename T>(const std::vector<T>& in) -> const T* {
        if (in.empty()) {
            return nullptr;
        }

        upload_size = align_up<kAlignment>(upload_size);

        std::size_t byte_size = in.size() * sizeof(T);
        pinned_.upload_at(upload_size, in.data(), byte_size);

        const T* device_ptr = reinterpret_cast<T*>(device_.data<std::byte>() + upload_size);
        upload_size += byte_size;
        return device_ptr;
    };

    ForwardBatch batch{
        .token_ids = upload_to_pinned(input.token_ids),
        .positions = upload_to_pinned(input.positions),
        .slot_mapping = upload_to_pinned(input.slot_mapping),
        .query_start_loc = upload_to_pinned(input.query_start_loc),
        .seq_lens = upload_to_pinned(input.seq_lens),
        .block_table = upload_to_pinned(input.block_table),
        .logits_indices = upload_to_pinned(input.logits_indices),
        .sample_params = upload_to_pinned(input.sample_params),
        .sampled_token_ids = device_.data<int>(),

        .block_table_stride = input.block_table_stride,
        .num_tokens = input.num_tokens(),
        .num_reqs = input.num_reqs(),
        .num_sampling_reqs = static_cast<int>(input.logits_indices.size()),
        .max_query_len = 0,
        .max_seq_len = *std::max_element(input.seq_lens.begin(), input.seq_lens.end())
    };

    for (int i = 0; i < input.num_reqs(); i++) {
        batch.max_query_len = std::max(
            batch.max_query_len,
            input.query_start_loc[i + 1] - input.query_start_loc[i]
        );
    }

    device_.upload_async(pinned_.data(), upload_size, context.stream());

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
