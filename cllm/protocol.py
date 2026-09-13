from enum import IntEnum


# These values mirror csrc/engine/protocol.h; only the engine child imports _C.
class RequestType(IntEnum):
    ADD = 0
    ABORT = 1
    SHUTDOWN = 2


class OutputType(IntEnum):
    READY = 0
    OUTPUTS = 1


class FinishReason(IntEnum):
    RUNNING = 0
    STOP = 1
    LENGTH = 2
