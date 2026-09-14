import json
import sys
import time
from pathlib import Path

import msgpack
import zmq

from cuinfer.engine import EngineConfig
from cuinfer.protocol import FinishReason, OutputType, RequestType


def run_fake_engine(
    config: EngineConfig,
    input_address: str,
    output_address: str,
    *,
    tokens: tuple[int, ...] = (101, 102, 103),
    eos_at: int | None = None,
    eos_token: int = 0,
    step_delay: float = 0.01,
    startup_delay: float = 0,
    crash: str | None = None,
    event_log: str | None = None,
    late_after_abort: bool = False,
    ignore_shutdown: bool = False,
) -> None:
    if crash == "startup":
        sys.exit(1)
    time.sleep(startup_delay)

    def log(event: dict) -> None:
        if event_log is not None:
            with Path(event_log).open("a") as file:
                file.write(json.dumps(event) + "\n")

    with zmq.Context() as context:
        with context.socket(zmq.PULL) as incoming, context.socket(zmq.PUSH) as outgoing:
            incoming.setsockopt(zmq.LINGER, 0)
            outgoing.setsockopt(zmq.LINGER, 1000)
            incoming.connect(input_address)
            outgoing.connect(output_address)
            outgoing.send_multipart([bytes([OutputType.READY])])
            log({"type": "READY"})
            requests: dict[str, tuple[dict, int]] = {}
            next_step = time.monotonic() + step_delay
            while True:
                timeout = max(0, int((next_step - time.monotonic()) * 1000))
                if incoming.poll(timeout if requests else None):
                    while True:
                        frames = incoming.recv_multipart()
                        tag = RequestType(frames[0][0])
                        payload = msgpack.unpackb(frames[1]) if len(frames) > 1 else None
                        if tag == RequestType.SHUTDOWN:
                            log({"type": "SHUTDOWN"})
                            if not ignore_shutdown:
                                return
                        elif tag == RequestType.ADD:
                            log({"type": "ADD", **payload})
                            requests[payload["request_id"]] = (payload, 0)
                        elif tag == RequestType.ABORT:
                            log({"type": "ABORT", "request_ids": payload})
                            for request_id in payload:
                                requests.pop(request_id, None)
                            if late_after_abort:
                                outgoing.send_multipart([
                                    bytes([OutputType.OUTPUTS]),
                                    msgpack.packb([
                                        {
                                            "request_id": request_id,
                                            "new_token_ids": [999],
                                            "finish_reason": FinishReason.LENGTH,
                                        }
                                        for request_id in payload
                                    ]),
                                ])
                        if not incoming.poll(0):
                            break
                if not requests or time.monotonic() < next_step:
                    continue
                outputs = []
                for request_id, (request, index) in list(requests.items()):
                    token = tokens[index % len(tokens)]
                    reason = FinishReason.RUNNING
                    if eos_at == index + 1:
                        token = eos_token
                        if not request.get("ignore_eos", False):
                            reason = FinishReason.STOP
                    if reason == FinishReason.RUNNING and index + 1 >= request["max_output_tokens"]:
                        reason = FinishReason.LENGTH
                    outputs.append({
                        "request_id": request_id,
                        "new_token_ids": [token],
                        "finish_reason": reason,
                    })
                    if reason == FinishReason.RUNNING:
                        requests[request_id] = (request, index + 1)
                    else:
                        del requests[request_id]
                outgoing.send_multipart([
                    bytes([OutputType.OUTPUTS]), msgpack.packb(outputs)
                ])
                if crash == "midrun":
                    sys.exit(1)
                next_step = time.monotonic() + step_delay
