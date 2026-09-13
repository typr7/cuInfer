#pragma once

#include "engine/config.h"
#include "engine/protocol.h"


namespace cuinfer
{

// export to python via pybind
EngineCoreShutdownReason run_engine_core(const Config& config, const Addresses& addresses);

}
