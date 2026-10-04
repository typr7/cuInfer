#include "cuda/cuda_context.h"
#include "cuda/cuda_utils.h"


namespace cuinfer
{

CudaContext::CudaContext()
{
    CUDA_CHECK(cudaStreamCreate(&stream_));

    try {
        CUBLAS_CHECK(cublasCreate(&cublas_));
        CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
        int device = 0;
        int major = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        cublas_workspace_.resize((major >= 9 ? 32 : 4) * 1024 * 1024);
        CUBLAS_CHECK(cublasSetWorkspace(cublas_, cublas_workspace_.data(), cublas_workspace_.size()));
    } catch (...) {
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
        }
        cudaStreamDestroy(stream_);
        throw;
    }
}

CudaContext::~CudaContext() noexcept
{
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void CudaContext::synchronize() const
{
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}
