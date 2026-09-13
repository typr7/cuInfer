#pragma once

#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +         \
                                     cudaGetErrorString(err__));           \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t err__ = (call);                                     \
        if (err__ != CUBLAS_STATUS_SUCCESS) {                              \
            throw std::runtime_error(std::string("cuBLAS error: ") +       \
                                     cublasGetStatusString(err__));        \
        }                                                                  \
    } while (0)
