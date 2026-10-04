#include <cassert>
#include <format>
#include <stdexcept>

#include "model/layers/linear.h"
#include "ops/projection.h"


namespace cuinfer
{

namespace
{

WeightFormat weight_format(const Checkpoint& checkpoint, const std::string& prefix)
{
    if (checkpoint.contains(prefix + ".qweight")) {
        return WeightFormat::kW4A16;
    }
    if (checkpoint.get(prefix + ".weight").dtype == DataType::kInt8) {
        return WeightFormat::kW8A8;
    }
    return WeightFormat::kBf16;
}

int output_size(const std::vector<LinearSource>& sources)
{
    int size = 0;
    for (const auto& source : sources) {
        size += source.out_features;
    }
    return size;
}

void upload_weights(const Checkpoint& checkpoint, const std::vector<LinearSource>& sources,
                    int in_features, std::byte* dst)
{
    for (const auto& source : sources) {
        const WeightFormat format = weight_format(checkpoint, source.prefix);
        if (format != WeightFormat::kBf16) {
            // TODO
            throw std::runtime_error(std::format(
                "quantized Linear `{}` is not implemented", source.prefix
            ));
        }
        const auto& weight = checkpoint.get(source.prefix + ".weight", DataType::kBf16,
                                            {source.out_features, in_features});
        upload_slice(weight, 0, 0, source.out_features, dst);
        dst += source.out_features * static_cast<std::size_t>(in_features)
            * dtype_byte_size(DataType::kBf16);
    }
}

}

Linear Linear::load(const Checkpoint& checkpoint, const std::vector<LinearSource>& sources,
                    int in_features, Parallel parallel)
{
    Linear linear;
    linear.parallel_ = parallel;
    // TODO: TP rank slicing before concatenation. Local sizes equal full sizes at tp=1.
    const int out_features = output_size(sources);
    linear.data_.resize(static_cast<std::size_t>(out_features) * in_features
                        * dtype_byte_size(DataType::kBf16));
    upload_weights(checkpoint, sources, in_features, linear.data_.data<std::byte>());
    linear.weight_ = make_tensor<2>(linear.data_.data(), DataType::kBf16,
                                   {out_features, in_features});
    return linear;
}

Linear Linear::tied(TensorRef<2> weight)
{
    Linear linear;
    linear.weight_ = weight;
    return linear;
}

void Linear::forward(TensorRef<2> in, TensorRef<2> out,
                     const CudaContext& context, const WorkspaceView& workspace) const
{
    switch (format_) {
        case WeightFormat::kBf16:
            ops::projection(in, weight_, out, context.cublas());
            break;
        case WeightFormat::kW4A16:
            // TODO: W4A16 GEMM.
            throw std::runtime_error("W4A16 GEMM is not implemented");
        case WeightFormat::kW8A8:
            // TODO: Per-token quantization using workspace.act_int8 / act_scales, W8A8 GEMM.
            throw std::runtime_error("W8A8 GEMM is not implemented");
    }
    if (parallel_ == Parallel::kRow) {
        // TODO: context.all_reduce(out) when tp > 1.
    }
}

GroupedLinear GroupedLinear::load(const Checkpoint& checkpoint,
                                 const std::vector<std::vector<LinearSource>>& experts,
                                 int in_features)
{
    assert(!experts.empty());
    GroupedLinear linear;
    const int out_features = output_size(experts.front());
    const std::size_t expert_bytes = static_cast<std::size_t>(out_features) * in_features
        * dtype_byte_size(DataType::kBf16);
    linear.data_.resize(experts.size() * expert_bytes);
    for (std::size_t i = 0; i < experts.size(); i++) {
        assert(output_size(experts[i]) == out_features);
        upload_weights(checkpoint, experts[i], in_features,
                       linear.data_.data<std::byte>() + i * expert_bytes);
    }
    linear.weight_ = make_tensor<3>(linear.data_.data(), DataType::kBf16,
                                    {static_cast<int>(experts.size()), out_features, in_features});
    return linear;
}

void GroupedLinear::forward(TensorRef<2> in, TensorRef<2> out, const int* sorted_token_ids,
                            const int* expert_ids, const CudaContext& context,
                            const WorkspaceView& workspace) const
{
    // TODO: Grouped GEMM for the local experts; reduction belongs to SparseMoE.
    throw std::runtime_error("grouped GEMM is not implemented");
}

}
