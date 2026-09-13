import asyncio
import json
import multiprocessing
from functools import partial
from pathlib import Path

import anyio
import pytest
import pytest_asyncio

from cllm.engine import EngineClient, EngineConfig, EngineDied
from cllm.protocol import FinishReason
from tests.fake_engine import run_fake_engine


PARAMS = {"max_output_tokens": 3, "temperature": 1.0, "top_k": 0, "top_p": 1.0}


@pytest_asyncio.fixture
async def engine_factory():
    clients = []

    async def start(**options):
        client = await EngineClient.start(
            EngineConfig(model_path="unused"),
            process_target=partial(run_fake_engine, **options),
        )
        clients.append(client)
        return client

    yield start
    await asyncio.gather(*(client.close() for client in clients))


async def wait_event(path: Path, event_type: str) -> dict:
    async with asyncio.timeout(3):
        while True:
            if path.exists():
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if event["type"] == event_type:
                        return event
            await asyncio.sleep(0.01)


async def collect(client: EngineClient, request_id: str, **params) -> list[dict]:
    return [output async for output in client.generate(request_id, [1], PARAMS | params)]


@pytest.mark.parametrize(
    "options",
    [
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 0.96},
        {"gpu_memory_utilization": float("nan")},
        {"block_size": 0},
        {"max_num_scheduled_tokens": -1},
        {"max_num_seqs": 0},
    ],
)
def test_config_matches_engine_validation(options):
    with pytest.raises(ValueError):
        EngineConfig(model_path="unused", **options)


@pytest.mark.asyncio
async def test_ready_handshake_and_ordered_concurrent_outputs(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    client = await engine_factory(event_log=str(events))
    assert client.alive
    assert client.error is None
    await wait_event(events, "READY")

    streams = await asyncio.gather(collect(client, "first"), collect(client, "second"))
    for request_id, outputs in zip(("first", "second"), streams):
        assert [output["request_id"] for output in outputs] == [request_id] * 3
        assert [output["new_token_ids"] for output in outputs] == [[101], [102], [103]]
        assert [output["finish_reason"] for output in outputs] == [
            FinishReason.RUNNING, FinishReason.RUNNING, FinishReason.LENGTH
        ]


@pytest.mark.asyncio
async def test_eos_is_preserved_in_stop_output(engine_factory):
    client = await engine_factory(eos_at=2, eos_token=999)
    outputs = await collect(client, "eos")
    assert [output["new_token_ids"] for output in outputs] == [[101], [999]]
    assert outputs[-1]["finish_reason"] == FinishReason.STOP


@pytest.mark.asyncio
async def test_abort_stops_stream_and_ignores_late_output(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    client = await engine_factory(event_log=str(events), late_after_abort=True)
    stream = client.generate("aborted", [1], PARAMS | {"max_output_tokens": 100})
    assert (await anext(stream))["new_token_ids"] == [101]
    await client.abort("aborted")
    assert [output async for output in stream] == []
    assert (await wait_event(events, "ABORT"))["request_ids"] == ["aborted"]
    outputs = await collect(client, "next")
    assert [output["new_token_ids"] for output in outputs] == [[101], [102], [103]]
    assert client.alive


@pytest.mark.asyncio
async def test_cancelled_consumer_aborts_request(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    client = await engine_factory(event_log=str(events), step_delay=0.05)
    consumer = asyncio.create_task(collect(client, "cancelled", max_output_tokens=1000))
    await wait_event(events, "ADD")
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert (await wait_event(events, "ABORT"))["request_ids"] == ["cancelled"]


@pytest.mark.asyncio
async def test_cancel_scope_aborts_request(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    client = await engine_factory(event_log=str(events), step_delay=0.05)
    scope_ready = asyncio.get_running_loop().create_future()

    async def consume():
        with anyio.CancelScope() as scope:
            scope_ready.set_result(scope)
            await collect(client, "disconnected", max_output_tokens=1000)

    consumer = asyncio.create_task(consume())
    scope = await scope_ready
    await wait_event(events, "ADD")
    scope.cancel()
    await consumer
    assert (await wait_event(events, "ABORT"))["request_ids"] == ["disconnected"]


@pytest.mark.asyncio
async def test_closing_generator_aborts_request(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    client = await engine_factory(event_log=str(events))
    stream = client.generate("closed", [1], PARAMS | {"max_output_tokens": 100})
    await anext(stream)
    await stream.aclose()
    assert (await wait_event(events, "ABORT"))["request_ids"] == ["closed"]


@pytest.mark.asyncio
async def test_duplicate_in_flight_id_is_rejected(engine_factory):
    client = await engine_factory()
    first = client.generate("same", [1], PARAMS | {"max_output_tokens": 100})
    await anext(first)
    with pytest.raises(ValueError, match="already in flight"):
        await collect(client, "same")
    await first.aclose()


@pytest.mark.asyncio
async def test_startup_crash_raises_without_hanging():
    children_before = {child.pid for child in multiprocessing.active_children()}
    async with asyncio.timeout(3):
        with pytest.raises(EngineDied, match="code 1"):
            await EngineClient.start(
                EngineConfig(model_path="unused"),
                process_target=partial(run_fake_engine, crash="startup"),
            )
    assert {child.pid for child in multiprocessing.active_children()} == children_before


@pytest.mark.asyncio
async def test_cancelled_startup_joins_child():
    children_before = {child.pid for child in multiprocessing.active_children()}
    startup = asyncio.create_task(EngineClient.start(
        EngineConfig(model_path="unused"),
        process_target=partial(run_fake_engine, startup_delay=0.2),
    ))
    async with asyncio.timeout(3):
        while {child.pid for child in multiprocessing.active_children()} == children_before:
            await asyncio.sleep(0.01)
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
    assert {child.pid for child in multiprocessing.active_children()} == children_before


@pytest.mark.asyncio
async def test_midrun_crash_fails_every_request_and_new_requests(engine_factory):
    client = await engine_factory(crash="midrun")
    async with asyncio.timeout(3):
        outcomes = await asyncio.gather(
            collect(client, "first", max_output_tokens=100),
            collect(client, "second", max_output_tokens=100),
            return_exceptions=True,
        )
        error = await client.wait_dead()
    assert all(isinstance(outcome, EngineDied) for outcome in outcomes)
    assert "code 1" in str(error)
    assert client.error is error
    assert not client.alive
    with pytest.raises(EngineDied, match="code 1"):
        await collect(client, "later")


@pytest.mark.asyncio
async def test_cancelled_death_waiter_does_not_cancel_watch(engine_factory):
    client = await engine_factory(crash="midrun")
    waiter = asyncio.create_task(client.wait_dead())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    with pytest.raises(EngineDied):
        await collect(client, "crash", max_output_tokens=100)
    assert isinstance(await client.wait_dead(), EngineDied)


@pytest.mark.asyncio
async def test_close_sends_shutdown_and_joins(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    children_before = {child.pid for child in multiprocessing.active_children()}
    client = await engine_factory(event_log=str(events))
    await client.close()
    await wait_event(events, "SHUTDOWN")
    assert not client.alive
    assert {child.pid for child in multiprocessing.active_children()} == children_before
    await client.close()


@pytest.mark.asyncio
async def test_close_kills_unresponsive_engine_without_blocking_loop(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    children_before = {child.pid for child in multiprocessing.active_children()}
    client = await engine_factory(event_log=str(events), ignore_shutdown=True)
    close_task = asyncio.create_task(client.close())
    await wait_event(events, "SHUTDOWN")
    assert not close_task.done()
    async with asyncio.timeout(5):
        await close_task
    assert {child.pid for child in multiprocessing.active_children()} == children_before


@pytest.mark.asyncio
async def test_cancelled_close_keeps_cleanup_running(engine_factory, tmp_path):
    events = tmp_path / "events.jsonl"
    children_before = {child.pid for child in multiprocessing.active_children()}
    client = await engine_factory(event_log=str(events), ignore_shutdown=True)
    closer = asyncio.create_task(client.close())
    await wait_event(events, "SHUTDOWN")
    closer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closer
    async with asyncio.timeout(5):
        await client.close()
    assert {child.pid for child in multiprocessing.active_children()} == children_before
