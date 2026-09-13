#include <cassert>
#include <climits>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include <math_constants.h>

#include "ops/sampler.h"
#include "cuda/cuda_utils.h"
#include "scheduler/sample_params.h"
#include "ops/utils.h"
#include "common/util.h"


namespace cuinfer::ops
{

namespace
{

constexpr int kNumThreads = 512;
constexpr int kTileSize = kNumThreads * 8;

struct Range
{
    float high;
    int index;
};

struct MergeRange
{
    __device__ Range operator()(Range a, Range b) const
    {
        const bool take_b = b.high > a.high || (b.high == a.high && b.index < a.index);
        return {fmaxf(a.high, b.high), take_b ? b.index : a.index};
    }
};

__device__ unsigned bf16_key(float value)
{
    const unsigned bits = value == 0.f ? 0 : __float_as_uint(value) >> 16;
    return bits ^ (bits & 0x8000 ? 0xffff : 0x8000);
}

__device__ void histogram_add(int* histogram, unsigned digit, bool valid)
{
    const unsigned mask = __ballot_sync(0xffffffff, valid);
    if (valid) {
        const unsigned peers = __match_any_sync(mask, digit);
        if ((threadIdx.x & 31) == __ffs(peers) - 1) {
            atomicAdd(histogram + digit, __popc(peers));
        }
    }
}

__device__ int select_digit(int count, int& k)
{
    __shared__ typename cub::BlockScan<int, kNumThreads>::TempStorage storage;
    __shared__ int2 selected;
    int prefix;
    cub::BlockScan<int, kNumThreads>(storage).ExclusiveSum(count, prefix);
    if (prefix < k && k <= prefix + count) {
        selected = make_int2(255 - threadIdx.x, k - prefix);
    }
    __syncthreads();
    const int digit = selected.x;
    k = selected.y;
    __syncthreads();
    return digit;
}

struct AddPair
{
    __device__ float2 operator()(float2 a, float2 b) const
    {
        return {a.x + b.x, a.y + b.y};
    }
};

template <typename T, typename Op>
__device__ T block_reduce(T value, Op op)
{
    __shared__ typename cub::BlockReduce<T, kNumThreads>::TempStorage storage;
    __shared__ T result;
    value = cub::BlockReduce<T, kNumThreads>(storage).Reduce(value, op);
    if (threadIdx.x == 0) {
        result = value;
    }
    __syncthreads();
    value = result;
    __syncthreads();
    return value;
}

template <bool kVectorized>
__device__ float4 load(const float* values, int i, int size, float padding)
{
    if constexpr (kVectorized) {
        return reinterpret_cast<const float4*>(values)[i / 4];
    } else {
        return {
            values[i],
            i + 1 < size ? values[i + 1] : padding,
            i + 2 < size ? values[i + 2] : padding,
            i + 3 < size ? values[i + 3] : padding
        };
    }
}

template <bool kVectorized>
__device__ float4 load(const nv_bfloat16* values, int i, int size, float padding)
{
    if constexpr (kVectorized) {
        union {
            uint2 vec;
            nv_bfloat16 bf16[4];
        } packed;
        packed.vec = reinterpret_cast<const uint2*>(values)[i / 4];
        return {
            __bfloat162float(packed.bf16[0]), __bfloat162float(packed.bf16[1]),
            __bfloat162float(packed.bf16[2]), __bfloat162float(packed.bf16[3])
        };
    } else {
        return {
            __bfloat162float(values[i]),
            i + 1 < size ? __bfloat162float(values[i + 1]) : padding,
            i + 2 < size ? __bfloat162float(values[i + 2]) : padding,
            i + 3 < size ? __bfloat162float(values[i + 3]) : padding
        };
    }
}

template <bool kVectorized>
__global__ __launch_bounds__(kNumThreads)
void prepare_logits(
    const nv_bfloat16* logits,
    const SampleParams* params,
    Range* ranges,
    int* histograms,
    int vocab_size
)
{
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * vocab_size;
    const int i = blockIdx.x * kTileSize + threadIdx.x * 8;
    const SampleParams param = params[blockIdx.y];
    const bool greedy = param.temperature == 0.f;
    const bool top_k = !greedy && param.top_k > 1 && param.top_k < vocab_size;
    __shared__ int histogram[256];
    if (top_k) {
        if (threadIdx.x < 256) {
            histogram[threadIdx.x] = 0;
        }
        __syncthreads();
    }
    Range range{-CUDART_INF_F, INT_MAX};
    BF16x8 packed;
    if constexpr (kVectorized) {
        if (i < vocab_size) {
            packed.vec = reinterpret_cast<const uint4*>(logits + row)[i / 8];
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const float value = i + j < vocab_size
                            ? __bfloat162float(kVectorized ? packed.bf16x8[j] : logits[row + i + j])
                            : -CUDART_INF_F;
        range = MergeRange{}(range, {value, i + j});
        if (top_k) {
            histogram_add(histogram, bf16_key(value) >> 8, i + j < vocab_size);
        }
    }
    range = block_reduce(range, MergeRange{});
    if (threadIdx.x == 0) {
        ranges[blockIdx.y * gridDim.x + blockIdx.x] = range;
    }
    if (top_k && threadIdx.x < 256) {
        histograms[(static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * 256
                   + threadIdx.x] = histogram[threadIdx.x];
    }
}

template <bool kVectorized>
__global__ __launch_bounds__(kNumThreads)
void sample_rows(
    const nv_bfloat16* logits,
    float* values,
    const Range* ranges,
    const int* histograms,
    const SampleParams* params,
    int* output,
    int vocab_size,
    int num_tiles,
    std::uint64_t seed,
    std::uint64_t offset
)
{
    const int tid = threadIdx.x;
    const SampleParams param = params[blockIdx.x];
    const nv_bfloat16* input = logits + static_cast<std::size_t>(blockIdx.x) * vocab_size;
    float* row = values + static_cast<std::size_t>(blockIdx.x) * vocab_size;
    Range range{-CUDART_INF_F, INT_MAX};
    for (int i = tid; i < num_tiles; i += kNumThreads) {
        range = MergeRange{}(range, ranges[blockIdx.x * num_tiles + i]);
    }
    range = block_reduce(range, MergeRange{});
    if (param.temperature == 0.f) {
        if (tid == 0) {
            output[blockIdx.x] = range.index;
        }
        return;
    }

    // Select the exact BF16 cutoff in two radix passes, retaining boundary ties.
    // The high-byte histogram was collected while loading the logits.
    float cutoff = -CUDART_INF_F;
    if (param.top_k == 1) {
        cutoff = range.high;
    } else if (param.top_k > 1 && param.top_k < vocab_size) {
        int count = 0;
        if (tid < 256) {
            for (int tile = 0; tile < num_tiles; tile++) {
                count += histograms[(static_cast<std::size_t>(blockIdx.x) * num_tiles + tile) * 256
                                    + 255 - tid];
            }
        }
        int k = param.top_k;
        const unsigned high = select_digit(count, k);
        __shared__ int histogram[256];
        if (tid < 256) {
            histogram[tid] = 0;
        }
        __syncthreads();
        for (int base = 0; base < vocab_size; base += kNumThreads * 4) {
            const int i = base + tid * 4;
            const float4 v = i < vocab_size
                            ? load<kVectorized>(input, i, vocab_size, -CUDART_INF_F)
                            : make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
            const float x[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned key = bf16_key(x[j]);
                histogram_add(histogram, key & 255, i + j < vocab_size && (key >> 8) == high);
            }
        }
        __syncthreads();
        const unsigned key = (high << 8) | select_digit(tid < 256 ? histogram[255 - tid] : 0, k);
        const unsigned bits = key ^ (key & 0x8000 ? 0x8000 : 0xffff);
        cutoff = __uint_as_float(bits << 16);
    }

    const float inv_temperature = 1.f / param.temperature;
    const float scaled_max = range.high * inv_temperature;
    float thread_mass = 0.f;
    for (int i = tid * 4; i < vocab_size; i += kNumThreads * 4) {
        const float4 v = load<kVectorized>(input, i, vocab_size, -CUDART_INF_F);
        float x[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float shifted = param.temperature >= 1.f
                                  ? x[j] * inv_temperature - scaled_max
                                  : (x[j] - range.high) * inv_temperature;
            x[j] = x[j] == range.high ? 1.f
                   : x[j] >= cutoff ? __expf(shifted) : 0.f;
            thread_mass += x[j];
        }
        if constexpr (kVectorized) {
            reinterpret_cast<float4*>(row)[i / 4] = make_float4(x[0], x[1], x[2], x[3]);
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (i + j < vocab_size) {
                    row[i + j] = x[j];
                }
            }
        }
    }
    __syncthreads();

    // FlashInfer top-p rejection sampling. Unnormalized FP32 weights let us
    // fuse temperature, softmax and top-k without another normalization pass.
    // https://github.com/flashinfer-ai/flashinfer/blob/dd12b73b4461c37b467f77133c17c770adfd3a7b/include/flashinfer/sampling.cuh
    curandStatePhilox4_32_10_t rng;
    if (tid == 0) {
        curand_init(seed, offset + blockIdx.x, 0, &rng);
    }
    __shared__ typename cub::BlockScan<float, kNumThreads>::TempStorage scan_storage;
    __shared__ float uniform;
    __shared__ int sampled;
    __shared__ int bucket;
    __shared__ float bucket_prefix;
    float low = 0.f;
    float high = 1.f;
    float target_mass = 0.f;
    bool first_round = true;
    while (true) {
        float prefix;
        float total;
        cub::BlockScan<float, kNumThreads>(scan_storage).ExclusiveSum(thread_mass, prefix, total);
        if (first_round) {
            target_mass = param.top_p * total;
            first_round = false;
        }
        if (tid == 0) {
            uniform = fminf(curand_uniform(&rng) * total, nextafterf(total, 0.f));
            sampled = INT_MAX;
            bucket = INT_MAX;
        }
        __syncthreads();

        // First select a thread bucket, then redistribute that bucket across
        // the block. Both CDF levels run in parallel, including at batch size 1.
        if (thread_mass > 0.f && uniform >= prefix && uniform < prefix + thread_mass) {
            atomicMin(&bucket, tid);
        }
        __syncthreads();
        if (tid == bucket) {
            bucket_prefix = prefix;
        }
        __syncthreads();
        if (bucket != INT_MAX) {
            float bucket_draw = uniform - bucket_prefix;
            for (int base = bucket * 4; base < vocab_size; base += kNumThreads * kNumThreads * 4) {
                const int i = base + tid * kNumThreads * 4;
                const float4 v = i < vocab_size ? load<kVectorized>(row, i, vocab_size, 0.f)
                                               : make_float4(0.f, 0.f, 0.f, 0.f);
                float x[4] = {v.x, v.y, v.z, v.w};
                float mass = 0.f;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    x[j] = x[j] > low ? x[j] : 0.f;
                    mass += x[j];
                }
                cub::BlockScan<float, kNumThreads>(scan_storage).ExclusiveSum(mass, prefix, total);
                float remaining = bucket_draw - prefix;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (remaining >= 0.f && remaining < x[j]) {
                        atomicMin(&sampled, i + j);
                    }
                    remaining -= x[j];
                }
                __syncthreads();
                if (sampled != INT_MAX) {
                    break;
                }
                bucket_draw -= total;
            }
        }
        __syncthreads();
        // FP32 prefix rounding can leave a gap of an ulp in the CDF.
        const int candidate = sampled == INT_MAX ? range.index : sampled;
        if (param.top_p == 1.f) {
            if (tid == 0) {
                output[blockIdx.x] = candidate;
            }
            return;
        }
        const float pivot = row[candidate];
        const float midpoint = (pivot + high) * 0.5f;
        float2 mass{0.f, 0.f};
        for (int i = tid * 4; i < vocab_size; i += kNumThreads * 4) {
            const float4 v = load<kVectorized>(row, i, vocab_size, 0.f);
            const float x[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                mass.x += x[j] > pivot ? x[j] : 0.f;
                mass.y += x[j] > midpoint ? x[j] : 0.f;
            }
        }
        const float2 sum = block_reduce(mass, AddPair{});
        if (sum.x < target_mass) {
            if (tid == 0) {
                output[blockIdx.x] = candidate;
            }
            return;
        }
        if (sum.y < target_mass) {
            low = pivot;
            high = midpoint;
            thread_mass = mass.x;
        } else {
            low = midpoint;
            thread_mass = mass.y;
        }
    }
}

}

std::size_t sampler_workspace_size(int num_reqs, int vocab_size)
{
    const int num_tiles = (vocab_size + kTileSize - 1) / kTileSize;
    return align_up<16>(static_cast<std::size_t>(num_reqs) * vocab_size * sizeof(float))
           + static_cast<std::size_t>(num_reqs) * num_tiles * (sizeof(Range) + 256 * sizeof(int));
}

void sample(
    TensorRef<2> logits,
    const SampleParams* params,
    int* output,
    void* workspace,
    std::uint64_t seed,
    std::uint64_t offset,
    cudaStream_t stream
)
{
    const int num_reqs = logits.shape[0];
    if (num_reqs == 0) {
        return;
    }
    assert(logits && params && output && workspace);
    assert(logits.dtype == DataType::kBf16);
    assert(logits.shape[1] > 0);
    assert(logits.stride[1] == 1 && logits.stride[0] == logits.shape[1]);

    const int vocab_size = logits.shape[1];
    const int num_tiles = (vocab_size + kTileSize - 1) / kTileSize;
    auto* values = static_cast<float*>(workspace);
    auto* ranges = reinterpret_cast<Range*>(
        static_cast<std::byte*>(workspace)
        + align_up<16>(static_cast<std::size_t>(num_reqs) * vocab_size * sizeof(float))
    );
    auto* histograms = reinterpret_cast<int*>(ranges + static_cast<std::size_t>(num_reqs) * num_tiles);
    const auto* input = static_cast<const nv_bfloat16*>(logits.device_ptr);
    const dim3 grid(num_tiles, num_reqs);
    if (vocab_size % 8 == 0) {
        prepare_logits<true><<<grid, kNumThreads, 0, stream>>>(
            input, params, ranges, histograms, vocab_size
        );
    } else {
        prepare_logits<false><<<grid, kNumThreads, 0, stream>>>(
            input, params, ranges, histograms, vocab_size
        );
    }
    CUDA_CHECK(cudaGetLastError());
    if (vocab_size % 4 == 0) {
        sample_rows<true><<<num_reqs, kNumThreads, 0, stream>>>(
            input, values, ranges, histograms, params, output, vocab_size, num_tiles, seed, offset
        );
    } else {
        sample_rows<false><<<num_reqs, kNumThreads, 0, stream>>>(
            input, values, ranges, histograms, params, output, vocab_size, num_tiles, seed, offset
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

}
