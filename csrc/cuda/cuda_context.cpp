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
    } catch (...) {
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
