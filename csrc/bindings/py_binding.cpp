#include <pybind11/pybind11.h>

#include "engine/engine_core.h"
#include "engine/protocol.h"


PYBIND11_MODULE(_C, module)
{
    namespace py = pybind11;

    py::class_<cuinfer::Config>(module, "Config")
        .def(py::init<>())
        .def_readwrite("model_path", &cuinfer::Config::model_path)
        .def_readwrite("gpu_memory_utilization", &cuinfer::Config::gpu_memory_utilization)
        .def_readwrite("block_size", &cuinfer::Config::block_size)
        .def_readwrite("max_num_scheduled_tokens", &cuinfer::Config::max_num_scheduled_tokens)
        .def_readwrite("max_num_seqs", &cuinfer::Config::max_num_seqs);

    py::class_<cuinfer::Addresses>(module, "EngineCoreAddresses")
        .def(py::init<>())
        .def_readwrite("input_address", &cuinfer::Addresses::input_address)
        .def_readwrite("output_address", &cuinfer::Addresses::output_address);

    // Also expose the wire protocol tags so Python/C++ contract tests can
    // verify the values mirrored by the parent process.
    py::enum_<cuinfer::RequestType>(module, "RequestType")
        .value("ADD", cuinfer::RequestType::kAdd)
        .value("ABORT", cuinfer::RequestType::kAbort)
        .value("SHUTDOWN", cuinfer::RequestType::kShutdown);

    py::enum_<cuinfer::OutputType>(module, "OutputType")
        .value("READY", cuinfer::OutputType::kReady)
        .value("OUTPUTS", cuinfer::OutputType::kOutputs);

    py::enum_<cuinfer::FinishReason>(module, "FinishReason")
        .value("RUNNING", cuinfer::FinishReason::kRunning)
        .value("STOP", cuinfer::FinishReason::kStop)
        .value("LENGTH", cuinfer::FinishReason::kLength);

    py::enum_<cuinfer::EngineCoreShutdownReason>(module, "EngineCoreShutdownReason")
        .value("SHUTDOWN", cuinfer::EngineCoreShutdownReason::kShutdown)
        .value("ENGINE_CORE_DEAD", cuinfer::EngineCoreShutdownReason::kEngineCoreDead);

    module.def(
        "run_engine_core",
        &cuinfer::run_engine_core,
        py::call_guard<py::gil_scoped_release>(),
        py::arg("config"),
        py::arg("addresses")
    );
}
