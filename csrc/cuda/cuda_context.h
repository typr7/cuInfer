#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>


namespace cllm
{

// The stream and the cuBLAS handle bound to it, as one RAII unit.
//
// Passed by reference to every layer and kernel launcher, so nothing below
// ModelRunner needs to reach back into it for a stream. The handle's stream is
// set once here rather than at each call site.
class CudaContext
{
public:
    CudaContext();
    ~CudaContext() noexcept;

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    cudaStream_t stream() const noexcept
    {
        return stream_;
    }

    cublasHandle_t cublas() const noexcept
    {
        return cublas_;
    }

    void synchronize() const;

private:
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
};

}
