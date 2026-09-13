cmake_minimum_required(VERSION 3.22)

# Default to Ada Lovelace (compute capability 8.9), while still allowing users
# to select another architecture with -DCMAKE_CUDA_ARCHITECTURES=<value>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
endif()

project(cuinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

add_library(cuinfer_core STATIC
    csrc/common/logger.cpp

    csrc/cuda/cuda_context.cpp
    csrc/cuda/cuda_device_buffer.cpp
    csrc/cuda/pinned_buffer.cpp

    csrc/engine/engine_core.cpp

    csrc/scheduler/kv_cache_manager.cpp
    csrc/scheduler/scheduler.cpp

    csrc/executor/batch_buffer.cpp
    csrc/executor/kv_cache.cpp
    csrc/executor/model_input.cpp
    csrc/executor/model_runner.cpp
    csrc/executor/sampler.cpp
    csrc/executor/workspace.cpp

    csrc/model/causal_lm.cpp
    csrc/model/model_config.cpp
    csrc/model/model_weights.cpp
    csrc/model/rope_cache.cpp

    csrc/ops/embedding.cu
    csrc/ops/paged_attention.cu
    csrc/ops/projection.cu
    csrc/ops/qk_norm_rope_fused.cu
    csrc/ops/qk_norm.cu
    csrc/ops/residual_add.cu
    csrc/ops/rms_norm.cu
    csrc/ops/rope.cu
    csrc/ops/sampler.cu
    csrc/ops/swiglu.cu
    csrc/ops/unified_kv_cache_update.cu
)

set_target_properties(cuinfer_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cuinfer_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_link_libraries(cuinfer_core
    PUBLIC
        msgpack-cxx
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        Threads::Threads
        cppzmq
        nlohmann_json::nlohmann_json
)

# Build the Python extension as _C.so. The Python interpreter is responsible
# for spawning the EngineCore process; a separate C++ executable is not needed.
pybind11_add_module(cuinfer_pybind MODULE
    csrc/bindings/py_binding.cpp
)

target_link_libraries(cuinfer_pybind
    PRIVATE
        cuinfer_core
)

set_target_properties(cuinfer_pybind PROPERTIES
    OUTPUT_NAME "_C"
)

install(
    TARGETS cuinfer_pybind
    LIBRARY DESTINATION cuinfer
)
