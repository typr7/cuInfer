import asyncio
import multiprocessing
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anyio
import msgpack
import zmq
import zmq.asyncio

from cuinfer.protocol import FinishReason, OutputType, RequestType


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    gpu_memory_utilization: float = 0.8
    block_size: int = 16
    max_num_scheduled_tokens: int = 8192
    max_num_seqs: int = 256

    def __post_init__(self) -> None:
        if not 0 < self.gpu_memory_utilization <= 0.95:
            raise ValueError("gpu_memory_utilization must be in (0, 0.95]")
        for field in ("block_size", "max_num_scheduled_tokens", "max_num_seqs"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be greater than zero")


class EngineDied(RuntimeError):
    pass


def run_engine(config: EngineConfig, input_address: str, output_address: str) -> None:
    from cuinfer import _C

    cfg = _C.Config()
    for name, value in asdict(config).items():
        setattr(cfg, name, value)
    addresses = _C.EngineCoreAddresses()
    addresses.input_address = input_address
    addresses.output_address = output_address
    reason = _C.run_engine_core(cfg, addresses)
    sys.exit(0 if reason == _C.EngineCoreShutdownReason.SHUTDOWN else 1)


class EngineClient:
    def __init__(self, config: EngineConfig, process_target: Callable[..., None]):
        self._directory = Path(tempfile.mkdtemp(prefix="cuinfer-"))
        input_address = f"ipc://{self._directory / 'input'}"
        output_address = f"ipc://{self._directory / 'output'}"
        self._context = zmq.asyncio.Context()
        self._input = self._context.socket(zmq.PUSH)
        self._output = self._context.socket(zmq.PULL)
        self._input.setsockopt(zmq.LINGER, 0)
        self._output.setsockopt(zmq.LINGER, 0)
        self._input.bind(input_address)
        self._output.bind(output_address)
        self._process = multiprocessing.get_context("spawn").Process(
            target=process_target,
            args=(config, input_address, output_address),
        )
        self._loop = asyncio.get_running_loop()
        self._ready: asyncio.Future[None] = self._loop.create_future()
        self._exited: asyncio.Future[None] = self._loop.create_future()
        self._dead: asyncio.Future[EngineDied] = self._loop.create_future()
        self._requests: dict[str, asyncio.Queue[dict[str, Any] | EngineDied | None]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self.error: EngineDied | None = None

    @classmethod
    async def start(
        cls, config: EngineConfig, *, process_target: Callable[..., None] = run_engine
    ) -> "EngineClient":
        client = cls(config, process_target)
        try:
            client._process.start()
            client._started = True
            client._loop.add_reader(client._process.sentinel, client._on_exit)
            client._reader = asyncio.create_task(client._read_outputs())
            await client._ready
            return client
        except BaseException:
            await client.close()
            raise

    @property
    def alive(self) -> bool:
        return (
            self._started
            and not self._closing
            and self.error is None
            and self._ready.done()
            and self._process.is_alive()
        )

    async def wait_dead(self) -> EngineDied:
        return await asyncio.shield(self._dead)

    def _fail_requests(self, error: EngineDied) -> None:
        for queue in self._requests.values():
            queue.put_nowait(error)
        self._requests.clear()

    def _mark_dead(self, error: EngineDied) -> None:
        if self.error is not None:
            return
        self.error = error
        self._dead.set_result(error)
        if not self._ready.done():
            self._ready.set_exception(error)
        self._fail_requests(error)

    def _on_exit(self) -> None:
        self._process.join(timeout=0)
        exitcode = self._process.exitcode
        if exitcode is None:
            return
        self._loop.remove_reader(self._process.sentinel)
        self._exited.set_result(None)
        if not self._closing:
            self._mark_dead(EngineDied(f"Engine process exited with code {exitcode}"))
            if self._reader is not None:
                self._reader.cancel()

    async def _read_outputs(self) -> None:
        try:
            frames = await self._output.recv_multipart()
            if frames != [bytes([OutputType.READY])]:
                raise ValueError("Expected READY as the first engine output")
            if not self._ready.done():
                self._ready.set_result(None)
            while True:
                frames = await self._output.recv_multipart()
                if len(frames) != 2 or frames[0] != bytes([OutputType.OUTPUTS]):
                    raise ValueError("Expected an OUTPUTS message from the engine")
                for output in msgpack.unpackb(frames[1]):
                    queue = self._requests.get(output["request_id"])
                    if queue is not None:
                        queue.put_nowait(output)
        except Exception as exc:
            self._mark_dead(EngineDied(f"Engine output reader failed: {exc}"))

    async def _send(self, tag: RequestType, payload: Any = None) -> None:
        frames = [bytes([tag])]
        if payload is not None:
            frames.append(msgpack.packb(payload))
        send = self._input.send_multipart(frames)
        if tag == RequestType.SHUTDOWN:
            await send
            return
        try:
            done, _ = await asyncio.wait((send, self._dead), return_when=asyncio.FIRST_COMPLETED)
            if self._dead in done:
                raise self._dead.result()
            await send
        finally:
            if not send.done():
                send.cancel()

    async def generate(
        self, request_id: str, token_ids: list[int], params: dict[str, int | float | bool]
    ) -> AsyncIterator[dict[str, Any]]:
        if not self.alive:
            raise self.error or EngineDied("Engine is not alive")
        if request_id in self._requests:
            raise ValueError(f"Request id is already in flight: {request_id}")
        queue: asyncio.Queue[dict[str, Any] | EngineDied | None] = asyncio.Queue()
        self._requests[request_id] = queue
        try:
            await self._send(
                RequestType.ADD,
                {**params, "request_id": request_id, "token_ids": token_ids},
            )
            while True:
                output = await queue.get()
                if isinstance(output, EngineDied):
                    raise output
                if output is None:
                    return
                finished = output["finish_reason"] != FinishReason.RUNNING
                if finished:
                    self._requests.pop(request_id, None)
                yield output
                if finished:
                    return
        finally:
            if self._requests.get(request_id) is queue:
                with anyio.CancelScope(shield=True):
                    await self.abort(request_id)

    async def abort(self, request_id: str) -> None:
        queue = self._requests.pop(request_id, None)
        if queue is not None:
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(None)
            if self.alive:
                await self._send(RequestType.ABORT, [request_id])

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closing = True
        self._fail_requests(EngineDied("Engine is closed"))
        try:
            if self._started:
                if self._process.is_alive():
                    try:
                        async with asyncio.timeout(3):
                            await self._send(RequestType.SHUTDOWN)
                            await asyncio.shield(self._exited)
                    except TimeoutError:
                        self._process.kill()
                await asyncio.shield(self._exited)
                self._process.join(timeout=0)
        finally:
            if self._reader is not None:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
            self._process.close()
            self._input.close()
            self._output.close()
            self._context.term()
            shutil.rmtree(self._directory)
