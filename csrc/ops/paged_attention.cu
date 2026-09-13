#include <bit>
#include <cassert>
#include <format>

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/paged_attention.h"
#include "ops/utils.h"
#include "ops/ptx.h"
#include "cuda/cuda_utils.h"


namespace cuinfer::ops
{

namespace
{

// m16n8k16
constexpr uint32_t kMmaM = 16;
constexpr uint32_t kMmaN = 8;
constexpr uint32_t kMmaK = 16;

__device__ __forceinline__
uint32_t swizzle_tma_128b(uint32_t row, uint32_t col)
{
    return (col ^ ((row & 0b111) << 3));
}

template <typename T, uint32_t kBr, uint32_t kBc, uint32_t kHeadDim>
struct SharedStorage
{
    union
    {
        struct
        {
            union
            {
                alignas(128) T smem_q[kBr][kHeadDim];
                alignas(128) T smem_k[kBc][kHeadDim];
            };
            alignas(128) T smem_v[kBc][kHeadDim];
        };

        alignas(128) T smem_o[kBr][kHeadDim];
    };
};

// Fold [query token, GQA head] into the MMA row dimension.
template <uint32_t kNumThreads, uint32_t kBr, uint32_t kHeadDim>
__device__ __forceinline__
void load_q_async(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16 (&smem)[kBr][kHeadDim],
    uint32_t row_start,
    uint32_t num_rows,
    uint32_t group_size,
    uint32_t qkv_stride
)
{
    constexpr uint32_t kColsU4 = kHeadDim / kNumBf16sPerVector;
#pragma unroll
    for (uint32_t i = threadIdx.x; i < kBr * kColsU4; i += kNumThreads) {
        const uint32_t y = i / kColsU4;
        const uint32_t x = (i % kColsU4) * kNumBf16sPerVector;
        const uint32_t row = row_start + y;
        const uint32_t offset = row < num_rows
            ? (row / group_size) * qkv_stride + (row % group_size) * kHeadDim + x
            : 0;
        cp_async_cg_16(qkv + offset, &smem[y][swizzle_tma_128b(y, x)], row < num_rows ? 16 : 0);
    }
    cp_async_commit();
}

template <uint32_t kNumThreads, uint32_t kBc, uint32_t kHeadDim>
__device__ __forceinline__
void load_kv_async(
    const __nv_bfloat16* __restrict__ cache,
    __nv_bfloat16 (&smem)[kBc][kHeadDim],
    const int* __restrict__ block_table,
    uint32_t kv_start,
    uint32_t kv_len,
    uint32_t kv_stride,
    uint32_t block_shift
)
{
    constexpr uint32_t kColsU4 = kHeadDim / kNumBf16sPerVector;
    const uint32_t block_mask = (1u << block_shift) - 1;
#pragma unroll
    for (uint32_t i = threadIdx.x; i < kBc * kColsU4; i += kNumThreads) {
        const uint32_t y = i / kColsU4;
        const uint32_t x = (i % kColsU4) * kNumBf16sPerVector;
        const uint32_t pos = kv_start + y;
        uint64_t offset = 0;
        if (pos < kv_len) {
            const uint32_t block = block_table[pos >> block_shift];
            // The physical KV pool is not bounded by the scheduled token budget.
            const uint32_t slot = (block << block_shift) + (pos & block_mask);
            offset = static_cast<uint64_t>(slot) * kv_stride + x;
        }
        cp_async_cg_16(cache + offset, &smem[y][swizzle_tma_128b(y, x)], pos < kv_len ? 16 : 0);
    }
    cp_async_commit();
}

template <
    typename T,
    uint32_t kNumTiles,
    uint32_t kNumRegs,
    uint32_t kNumRows,
    uint32_t kNumCols
>
__device__ __forceinline__
void load_mma_tile_q_s2r(
    uint32_t (&reg)[kNumTiles][kNumRegs],
    const T (&smem)[kNumRows][kNumCols]
)
{
    const uint32_t warp_id = threadIdx.x / kNumThreadsPerWarp;
    const uint32_t lane_id = threadIdx.x % kNumThreadsPerWarp;
    const uint32_t y = warp_id * kMmaM + lane_id % kMmaM;
    const uint32_t x_offset = lane_id / kMmaM * (kMmaK / 2);
#pragma unroll
    for (uint32_t i = 0; i < kNumTiles; i++) {
        const uint32_t x = i * kMmaK + x_offset;
        ldmatrix_x4(reg[i], &smem[y][swizzle_tma_128b(y, x)]);
    }
}

// Adapted from fa2_cp_async_lazy_rescale.cu: register-resident Q, overlapped
// cp.async K/V loads, and lazy online-softmax rescaling.
template <uint32_t kHeadDim>
__global__ __launch_bounds__(128)
void bf16_paged_attention(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    uint32_t num_reqs,
    uint32_t block_table_stride,
    uint32_t block_shift,
    uint32_t q_head_num,
    uint32_t kv_head_num,
    uint32_t group_size
)
{
    using T = __nv_bfloat16;
    constexpr uint32_t kNumThreads = 4 * kNumThreadsPerWarp;
    constexpr uint32_t kBr = 4 * kMmaM;
    constexpr uint32_t kBc = 64;
    constexpr uint32_t kNumQRegsPerThread =
        kMmaM * kMmaK * sizeof(T) / sizeof(uint32_t) / kNumThreadsPerWarp;
    constexpr uint32_t kNumAccRegsPerThread = kMmaM * kMmaN / kNumThreadsPerWarp;
    constexpr uint32_t kHeadDimDivMmaK = kHeadDim / kMmaK;
    constexpr uint32_t kHeadDimDivMmaN = kHeadDim / kMmaN;
    constexpr uint32_t kBcDivMmaN = kBc / kMmaN;
    constexpr uint32_t kBcDivMmaK = kBc / kMmaK;
    constexpr float kMaxExponentGap = 8.f;
    const float kScale = rsqrt(static_cast<float>(kHeadDim));
    const float kScaleLog2e = kScale * 1.44269504f;

    using Shared = SharedStorage<T, kBr, kBc, kHeadDim>;

    __shared__ Shared shared;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / kNumThreadsPerWarp;
    const uint32_t lane_id = tid % kNumThreadsPerWarp;
    const uint32_t kv_head_idx = blockIdx.y;

    // b(r) = floor(query_start_loc[r] * group_size / kBr) + r.
    // Consecutive b(r) reserve enough tiles for every request, with at most
    // one empty tile each; no rectangular max_query_len grid or tile metadata.
    uint32_t lo = 0;
    uint32_t hi = num_reqs;
    while (lo + 1 < hi) {
        const uint32_t mid = (lo + hi) / 2;
        const uint32_t tile = query_start_loc[mid] * group_size / kBr + mid;
        if (tile <= blockIdx.x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const uint32_t req_idx = lo;
    const uint32_t query_begin = query_start_loc[req_idx];
    const uint32_t query_len = query_start_loc[req_idx + 1] - query_begin;
    const uint32_t q_start_idx = (blockIdx.x - (query_begin * group_size / kBr + req_idx)) * kBr;
    const uint32_t num_rows = query_len * group_size;
    if (q_start_idx >= num_rows) {
        return;
    }

    const uint32_t seq_len = seq_lens[req_idx];
    const uint32_t prefix_len = seq_len - query_len;
    const uint32_t warp_start_idx = q_start_idx + warp_id * kMmaM;
    const uint32_t q_stride = q_head_num * kHeadDim;
    const uint32_t kv_stride = kv_head_num * kHeadDim;
    const uint32_t qkv_stride = q_stride + 2 * kv_stride;
    const uint32_t q_head_offset = kv_head_idx * group_size * kHeadDim;

    // Q/O fit in 32-bit offsets for the 8192-token schedule budget.
    Q += query_begin * qkv_stride + q_head_offset;
    O += query_begin * q_stride + q_head_offset;
    K += kv_head_idx * kHeadDim;
    V += kv_head_idx * kHeadDim;
    block_table += req_idx * block_table_stride;

    load_q_async<kNumThreads>(Q, shared.smem_q, q_start_idx, num_rows, group_size, qkv_stride);
    cp_async_wait_group<0>();
    __syncthreads();

    // load Q tile from smem to reg
    uint32_t q_reg[kHeadDim / kMmaK][kNumQRegsPerThread];
    load_mma_tile_q_s2r(q_reg, shared.smem_q);

    float m_ref[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.f, 0.f};

    float acc_o[kHeadDimDivMmaN][kNumAccRegsPerThread] = {0.f};
    const uint32_t kv_len = min(seq_len, prefix_len + (q_start_idx + kBr - 1) / group_size + 1);

    __syncthreads();
    for (uint32_t kv_start_idx = 0; kv_start_idx < kv_len; kv_start_idx += kBc) {
        load_kv_async<kNumThreads>(
            K, shared.smem_k, block_table, kv_start_idx, kv_len, kv_stride, block_shift
        );
        load_kv_async<kNumThreads>(
            V, shared.smem_v, block_table, kv_start_idx, kv_len, kv_stride, block_shift
        );
        // K is ready while V can overlap with QK^T and softmax.
        cp_async_wait_group<1>();
        __syncthreads();

        // S = QK^T
        float acc_s[kBcDivMmaN][kNumAccRegsPerThread] = {0.f};
#pragma unroll
        for (uint32_t k = 0; k < kHeadDimDivMmaK; k++) {
            const uint32_t x = k * kMmaK + ((lane_id / 8) & 0b1) * 8;
#pragma unroll
            for (uint32_t n = 0; n < kBcDivMmaN; n += 2) {
                uint32_t k_reg[4];
                const uint32_t y = n * kMmaN + (lane_id / 16) * 8 + lane_id % 8;
                ldmatrix_x4(k_reg, &shared.smem_k[y][swizzle_tma_128b(y, x)]);
                mma_m16n8k16(q_reg[k], k_reg, acc_s[n]);
                mma_m16n8k16(q_reg[k], k_reg + 2, acc_s[n + 1]);
            }
        }

        const uint32_t row0 = prefix_len + (warp_start_idx + lane_id / 4) / group_size;
        const uint32_t row1 = prefix_len + (warp_start_idx + lane_id / 4 + 8) / group_size;
        const uint32_t col_base = kv_start_idx + (lane_id % 4) * 2;

        // online softmax
        float tile_max[2] = {-CUDART_INF_F, -CUDART_INF_F};

        const bool need_causal_mask = prefix_len + warp_start_idx / group_size < kv_start_idx + kBc
            || kv_start_idx + kBc > kv_len;

#pragma unroll
        for (uint32_t i = 0; i < kBcDivMmaN; i++) {
            const uint32_t col = col_base + i * kMmaN;
            float* s = acc_s[i];
            if (need_causal_mask) {
                s[0] = row0 >= col && col < kv_len ? s[0] : -CUDART_INF_F;
                s[1] = row0 >= col + 1 && col + 1 < kv_len ? s[1] : -CUDART_INF_F;
                s[2] = row1 >= col && col < kv_len ? s[2] : -CUDART_INF_F;
                s[3] = row1 >= col + 1 && col + 1 < kv_len ? s[3] : -CUDART_INF_F;
            }
            tile_max[0] = fmaxf(tile_max[0], fmaxf(s[0], s[1]));
            tile_max[1] = fmaxf(tile_max[1], fmaxf(s[2], s[3]));
        }

        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 1));
        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 2));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 1));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 2));

        tile_max[0] *= kScaleLog2e;
        tile_max[1] *= kScaleLog2e;

        for (uint32_t row = 0; row < 2; row++) {
            const float new_max = tile_max[row];

            if (kv_start_idx == 0) {
                m_ref[row] = new_max;
            } else if (new_max > m_ref[row] + kMaxExponentGap) {
                const float scale = exp2f(m_ref[row] - new_max);

#pragma unroll
                for (uint32_t n = 0; n < kHeadDimDivMmaN; n++) {
                    acc_o[n][row * 2] *= scale;
                    acc_o[n][row * 2 + 1] *= scale;
                }

                l[row] *= scale;
                m_ref[row] = new_max;
            }
        }

#pragma unroll
        for (uint32_t i = 0; i < kBcDivMmaN; i++) {
            float* s = acc_s[i];
            s[0] = exp2f(fmaf(s[0], kScaleLog2e, -m_ref[0]));
            s[1] = exp2f(fmaf(s[1], kScaleLog2e, -m_ref[0]));
            s[2] = exp2f(fmaf(s[2], kScaleLog2e, -m_ref[1]));
            s[3] = exp2f(fmaf(s[3], kScaleLog2e, -m_ref[1]));

            l[0] += s[0] + s[1];
            l[1] += s[2] + s[3];
        }

        cp_async_wait_group<0>();
        __syncthreads();

#pragma unroll
        for (uint32_t k = 0; k < kBcDivMmaK; k++) {
            const float* s0 = acc_s[k * 2]; // row00, row01, row10, row11
            const float* s1 = acc_s[k * 2 + 1];
            uint32_t p_reg[4] = {
                pack_float2(s0[0], s0[1]),
                pack_float2(s0[2], s0[3]),
                pack_float2(s1[0], s1[1]),
                pack_float2(s1[2], s1[3])
            };
            const uint32_t y = k * kMmaK + lane_id % kMmaK;
            const uint32_t x_offset = (lane_id / 16) * kMmaN;
#pragma unroll
            for (uint32_t n = 0; n < kHeadDimDivMmaN; n += 2) {
                uint32_t v_reg[4];
                const uint32_t x = n * kMmaN + x_offset;
                ldmatrix_x4_trans(v_reg, &shared.smem_v[y][swizzle_tma_128b(y, x)]);
                mma_m16n8k16(p_reg, v_reg, acc_o[n]);
                mma_m16n8k16(p_reg, v_reg + 2, acc_o[n + 1]);
            }
        }

        __syncthreads();
    }

    l[0] += __shfl_xor_sync(0xffffffff, l[0], 1);
    l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 1);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);

    // write back
    const float inv_l[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
    const uint32_t row0 = warp_id * kMmaM + lane_id / 4;
    const uint32_t row1 = row0 + 8;
    const uint32_t col_base = (lane_id % 4) * 2;

#pragma unroll
    for (uint32_t i = 0; i < kHeadDimDivMmaN; i++) {
        const uint32_t col = col_base + i * kMmaN;
        float* o = acc_o[i];
        o[0] *= inv_l[0];
        o[1] *= inv_l[0];
        o[2] *= inv_l[1];
        o[3] *= inv_l[1];
        *reinterpret_cast<__nv_bfloat162*>(&shared.smem_o[row0][swizzle_tma_128b(row0, col)]) =
            __float22bfloat162_rn(make_float2(o[0], o[1]));
        *reinterpret_cast<__nv_bfloat162*>(&shared.smem_o[row1][swizzle_tma_128b(row1, col)]) =
            __float22bfloat162_rn(make_float2(o[2], o[3]));
    }
    __syncthreads();

    constexpr uint32_t kHeadDimU4 = kHeadDim / kNumBf16sPerVector;
#pragma unroll
    for (uint32_t i = tid; i < kBr * kHeadDimU4; i += kNumThreads) {
        const uint32_t y = i / kHeadDimU4;
        const uint32_t x = (i % kHeadDimU4) * kNumBf16sPerVector;
        const uint32_t row = q_start_idx + y;
        if (row < num_rows) {
            const uint32_t offset = (row / group_size) * q_stride + (row % group_size) * kHeadDim + x;
            *reinterpret_cast<uint4*>(O + offset) =
                *reinterpret_cast<uint4*>(&shared.smem_o[y][swizzle_tma_128b(y, x)]);
        }
    }
}

template <uint32_t kHeadDim>
void launch_paged_attention(
    TensorRef<2> qkv,
    TensorRef<3> k_cache,
    TensorRef<3> v_cache,
    TensorRef<2> output,
    const ForwardBatch& batch,
    uint32_t block_size,
    cudaStream_t stream
)
{
    constexpr uint32_t kBr = 4 * kMmaM;
    const uint32_t kv_heads = k_cache.shape[1];
    const uint32_t q_heads = output.shape[1] / kHeadDim;
    const uint32_t group_size = q_heads / kv_heads;
    const dim3 grid(batch.num_tokens * group_size / kBr + batch.num_reqs, kv_heads);
    bf16_paged_attention<kHeadDim><<<grid, 4 * kNumThreadsPerWarp, 0, stream>>>(
        qkv.data<__nv_bfloat16>(),
        k_cache.data<__nv_bfloat16>(),
        v_cache.data<__nv_bfloat16>(),
        output.data<__nv_bfloat16>(),
        batch.query_start_loc,
        batch.seq_lens,
        batch.block_table,
        batch.num_reqs,
        batch.block_table_stride,
        std::countr_zero(block_size),
        q_heads,
        kv_heads,
        group_size
    );
}

}

void paged_attention(
    TensorRef<2> qkv,
    TensorRef<3> k_cache,
    TensorRef<3> v_cache,
    TensorRef<2> output,
    const ForwardBatch& batch,
    int block_size,
    cudaStream_t stream
)
{
    assert(qkv && k_cache && v_cache && output);
    assert(batch.query_start_loc && batch.seq_lens && batch.block_table);
    const int head_dim = k_cache.shape[2];
    const int kv_size = k_cache.shape[1] * head_dim;
    const int q_size = output.shape[1];

    if (!std::has_single_bit(static_cast<uint32_t>(block_size))) {
        throw std::runtime_error("paged attention requires a power-of-two block size");
    }
    if (head_dim == 128) {
        launch_paged_attention<128>(qkv, k_cache, v_cache, output, batch, block_size, stream);
    } else if (head_dim == 64) {
        launch_paged_attention<64>(qkv, k_cache, v_cache, output, batch, block_size, stream);
    } else {
        throw std::runtime_error(std::format("unsupported head_dim={}", head_dim));
    }
    CUDA_CHECK(cudaGetLastError());
}

}
