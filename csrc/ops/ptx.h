#pragma once

#include <cuda_runtime.h>

#include "ops/utils.h"


namespace cllm::ops
{

__device__ __forceinline__
void cp_async_cg_16(const void* gmem, void* smem, uint32_t src_size = 16)
{
    const uint32_t smem_addr = cvta_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem), "r"(src_size)
        : "memory"
    );
}

__device__ __forceinline__
void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <uint32_t kPendingGroups>
__device__ __forceinline__
void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(kPendingGroups) : "memory");
}

// ldmatrix
__device__ __forceinline__
void ldmatrix_x4(uint32_t reg[4], const void* smem)
{
    const uint32_t addr = cvta_shared(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr)
    );
}

__device__ __forceinline__
void ldmatrix_x4_trans(uint32_t *reg, const void *smem)
{
    const uint32_t addr = cvta_shared(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr)
    );
}

// MMA
__device__ __forceinline__
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, " // D
        "{%4, %5, %6, %7}, " // A
        "{%8, %9}, " // B
        "{%0, %1, %2, %3};\n" // D
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
    );
}

}