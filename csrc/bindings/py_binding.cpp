#include <pybind11/pybind11.h>

#include "engine/engine_core.h"
#include "engine/protocol.h"


PYBIND11_MODULE(_C, module)
{
    namespace py = pybind11;

    py::class_<cllm::Config>(module, "Config")
        .def(py::init<>())
        .def_readwrite("model_path", &cllm::Config::model_path)
        .def_readwrite("gpu_memory_utilization", &cllm::Config::gpu_memory_utilization)
        .def_readwrite("block_size", &cllm::Config::block_size)
        .def_readwrite("max_num_scheduled_tokens", &cllm::Config::max_num_scheduled_tokens)
        .def_readwrite("max_num_seqs", &cllm::Config::max_num_seqs);

    py::class_<cllm::Addresses>(module, "EngineCoreAddresses")
        .def(py::init<>())
        .def_readwrite("input_address", &cllm::Addresses::input_address)
        .def_readwrite("output_address", &cllm::Addresses::output_address);

    // Also expose the wire protocol tags so Python/C++ contract tests can
    // verify the values mirrored by the parent process.
    py::enum_<cllm::RequestType>(module, "RequestType")
        .value("ADD", cllm::RequestType::kAdd)
        .value("ABORT", cllm::RequestType::kAbort)
        .value("SHUTDOWN", cllm::RequestType::kShutdown);

    py::enum_<cllm::OutputType>(module, "OutputType")
        .value("READY", cllm::OutputType::kReady)
        .value("OUTPUTS", cllm::OutputType::kOutputs);

    py::enum_<cllm::FinishReason>(module, "FinishReason")
        .value("RUNNING", cllm::FinishReason::kRunning)
        .value("STOP", cllm::FinishReason::kStop)
        .value("LENGTH", cllm::FinishReason::kLength);

    py::enum_<cllm::EngineCoreShutdownReason>(module, "EngineCoreShutdownReason")
        .value("SHUTDOWN", cllm::EngineCoreShutdownReason::kShutdown)
        .value("ENGINE_CORE_DEAD", cllm::EngineCoreShutdownReason::kEngineCoreDead);

    module.def(
        "run_engine_core",
        &cllm::run_engine_core,
        py::call_guard<py::gil_scoped_release>(),
        py::arg("config"),
        py::arg("addresses")
    );
}
