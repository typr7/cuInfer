import pytest

from cuinfer.protocol import FinishReason, OutputType, RequestType


def test_native_protocol_values_match_python():
    native = pytest.importorskip("cuinfer._C", exc_type=ImportError)

    assert int(native.RequestType.ADD) == RequestType.ADD
    assert int(native.RequestType.ABORT) == RequestType.ABORT
    assert int(native.RequestType.SHUTDOWN) == RequestType.SHUTDOWN
    assert int(native.OutputType.READY) == OutputType.READY
    assert int(native.OutputType.OUTPUTS) == OutputType.OUTPUTS
    assert int(native.FinishReason.RUNNING) == FinishReason.RUNNING
    assert int(native.FinishReason.STOP) == FinishReason.STOP
    assert int(native.FinishReason.LENGTH) == FinishReason.LENGTH
