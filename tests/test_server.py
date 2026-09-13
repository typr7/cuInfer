import asyncio
import json
from functools import partial
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from starlette.requests import ClientDisconnect

from cuinfer.engine import EngineClient, EngineConfig
from cuinfer.server import create_app
from tests.fake_engine import run_fake_engine


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is False
        assert messages == [{"role": "user", "content": "Say hi"}]
        return [99, 104, 97, 116]

    def decode(self, token_ids, *, skip_special_tokens, **kwargs):
        assert skip_special_tokens is True
        return bytes(token for token in token_ids if token != 0).decode(
            errors="replace"
        )


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def wait_for_event(path: Path, kind: str) -> dict:
    async with asyncio.timeout(3):
        while True:
            for event in events(path):
                if event["type"] == kind:
                    return event
            await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def engine(request, tmp_path):
    options = {"tokens": (72, 105), "eos_at": 3, "step_delay": 0.02}
    options.update(getattr(request, "param", {}))
    client = await EngineClient.start(
        EngineConfig(model_path="fake-model"),
        process_target=partial(
            run_fake_engine, event_log=str(tmp_path / "engine.jsonl"), **options
        ),
    )
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def app(engine, tokenizer):
    return create_app(engine, tokenizer, "fake-model", 32, vocab_size=256)


@pytest_asyncio.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


CHAT = {"model": "fake-model", "messages": [{"role": "user", "content": "Say hi"}]}


async def test_health_and_models(client):
    assert (await client.get("/health")).status_code == 200
    response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert response.json()["data"][0]["id"] == "fake-model"


async def test_chat_completion_and_defaults(client, tmp_path):
    response = await client.post("/v1/chat/completions", json=CHAT)
    assert response.status_code == 200
    result = response.json()
    assert result["id"].startswith("chatcmpl-")
    assert result["object"] == "chat.completion"
    assert result["model"] == "fake-model"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "Hi"}
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7
    }
    added = await wait_for_event(tmp_path / "engine.jsonl", "ADD")
    assert added["token_ids"] == [99, 104, 97, 116]
    assert added["max_output_tokens"] == 28
    assert (added["temperature"], added["top_k"], added["top_p"]) == (1.0, 0, 1.0)


@pytest.mark.parametrize("prompt", ["A", [65]])
async def test_text_completion(client, prompt):
    response = await client.post(
        "/v1/completions", json={"prompt": prompt, "max_tokens": 1, "temperature": 0}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["object"] == "text_completion"
    assert result["choices"][0]["text"] == "H"
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"] == {
        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
    }


@pytest.mark.parametrize("chat", [True, False])
@pytest.mark.parametrize("max_tokens", [1, 3])
async def test_streaming_completion(client, chat, max_tokens):
    path = "/v1/chat/completions" if chat else "/v1/completions"
    payload = CHAT if chat else {"prompt": "A"}
    async with client.stream(
        "POST", path, json={**payload, "stream": True, "max_tokens": max_tokens}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = (await response.aread()).decode()
    assert body.endswith("data: [DONE]\n\n")
    frames = body.split("\n\n")
    assert frames[-1] == ""
    assert all(frame.startswith("data: ") for frame in frames[:-1])
    chunks = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-2]]
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert all(chunk["model"] == "fake-model" for chunk in chunks)
    assert all(chunk["choices"][0]["index"] == 0 for chunk in chunks)
    expected_object = "chat.completion.chunk" if chat else "text_completion"
    assert all(chunk["object"] == expected_object for chunk in chunks)
    choices = [chunk["choices"][0] for chunk in chunks]
    if chat:
        assert choices[0]["delta"]["role"] == "assistant"
        text = "".join(choice["delta"].get("content", "") for choice in choices)
    else:
        text = "".join(choice["text"] for choice in choices)
    assert text == ("H" if max_tokens == 1 else "Hi")
    assert choices[-1]["finish_reason"] == ("length" if max_tokens == 1 else "stop")
    assert all(choice["finish_reason"] is None for choice in choices[:-1])


@pytest.mark.parametrize("params", [
    {"prompt": ""}, {"prompt": []}, {"prompt": [65] * 32},
    {"prompt": [65] * 31, "max_tokens": 2},
    {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": 1.5},
    {"temperature": -0.1}, {"top_k": -1}, {"top_p": 0}, {"top_p": 1.1},
    {"prompt": [-1]}, {"prompt": [256]}, {"prompt": [True]}, {"prompt": ["65"]},
    {"prompt": [2**64]}, {"max_tokens": 2**64}, {"top_k": 2**64},
    {"temperature": 1e100},
])
async def test_invalid_requests_never_reach_engine(client, tmp_path, params):
    response = await client.post("/v1/completions", json={"prompt": "A", **params})
    assert response.status_code == 400
    assert not any(event["type"] == "ADD" for event in events(tmp_path / "engine.jsonl"))


async def test_malformed_json_never_reaches_engine(client, tmp_path):
    response = await client.post(
        "/v1/completions", content='{"prompt":',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not any(event["type"] == "ADD" for event in events(tmp_path / "engine.jsonl"))


@pytest.mark.parametrize("field,value", [
    ("temperature", float("nan")), ("temperature", float("inf")),
    ("top_p", float("nan")), ("top_p", float("-inf")),
])
async def test_nonfinite_sampling_never_reaches_engine(client, tmp_path, field, value):
    response = await client.post(
        "/v1/completions", content=json.dumps({"prompt": "A", field: value}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert not any(event["type"] == "ADD" for event in events(tmp_path / "engine.jsonl"))


@pytest.mark.parametrize("payload", [
    {}, {"messages": []}, {"messages": "Say hi"},
    {"messages": [{"role": "user"}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    {"messages": [{"role": "tool", "content": "tool result"}]},
])
async def test_invalid_messages_never_reach_engine(client, tmp_path, payload):
    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert not any(event["type"] == "ADD" for event in events(tmp_path / "engine.jsonl"))


@pytest.mark.parametrize("params", [
    {"n": 2}, {"best_of": 1}, {"logprobs": 1}, {"top_logprobs": 1},
    {"stop": "end"}, {"seed": 1}, {"tools": []}, {"functions": []},
    {"response_format": {"type": "text"}}, {"presence_penalty": 0.1},
    {"frequency_penalty": -0.1}, {"logit_bias": {}}, {"echo": True},
    {"suffix": "end"},
])
async def test_unsupported_parameters_never_reach_engine(client, tmp_path, params):
    response = await client.post("/v1/chat/completions", json={**CHAT, **params})
    assert response.status_code == 400
    assert not any(event["type"] == "ADD" for event in events(tmp_path / "engine.jsonl"))


async def test_accepted_parameters_are_forwarded(client, tmp_path):
    response = await client.post("/v1/completions", json={
        "prompt": "A", "max_tokens": 2, "temperature": 0.5, "top_k": 10,
        "top_p": 0.8, "n": 1, "presence_penalty": 0, "frequency_penalty": 0,
        "user": "test", "stream_options": {}, "store": False, "metadata": {},
    })
    assert response.status_code == 200
    added = await wait_for_event(tmp_path / "engine.jsonl", "ADD")
    assert added["max_output_tokens"] == 2
    assert (added["temperature"], added["top_k"], added["top_p"]) == (0.5, 10, 0.8)


@pytest.mark.parametrize("engine", [{"crash": "midrun", "eos_at": None}], indirect=True)
async def test_dead_engine_returns_503(client, engine):
    response = await client.post("/v1/chat/completions", json={**CHAT, "max_tokens": 10})
    assert response.status_code == 503
    await engine.wait_dead()
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 503
    assert (await client.get("/health")).status_code == 503


@pytest.mark.parametrize("engine", [{"crash": "midrun", "eos_at": None}], indirect=True)
async def test_engine_crash_during_stream_returns_sse_error(client):
    response = await client.post(
        "/v1/chat/completions", json={**CHAT, "max_tokens": 10, "stream": True}
    )
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    chunks = [json.loads(frame[6:]) for frame in response.text.split("\n\n")[:-2]]
    assert chunks[-1]["error"]["type"] == "server_error"
    assert "code 1" in chunks[-1]["error"]["message"]
    assert all(chunk["choices"][0]["finish_reason"] is None for chunk in chunks[:-1])


@pytest.mark.parametrize("stream", [False, True])
async def test_handler_failure_aborts_request(client, tokenizer, tmp_path, monkeypatch, stream):
    decode = tokenizer.decode

    def fail_on_output(token_ids, **kwargs):
        if 72 in token_ids:
            raise RuntimeError("decode failed")
        return decode(token_ids, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", fail_on_output)
    with pytest.raises(RuntimeError, match="decode failed"):
        await client.post("/v1/chat/completions", json={**CHAT, "stream": stream})
    added = await wait_for_event(tmp_path / "engine.jsonl", "ADD")
    aborted = await wait_for_event(tmp_path / "engine.jsonl", "ABORT")
    assert added["request_id"] in aborted["request_ids"]


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_stream_disconnect_aborts_request(app, tmp_path, spec_version):
    disconnect = asyncio.Event()
    request_sent = False
    responses = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": json.dumps({**CHAT, "stream": True}).encode()}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        responses.append(message)
        if message["type"] == "http.response.body":
            for frame in message.get("body", b"").decode().split("\n\n"):
                if frame.startswith("data: {"):
                    choice = json.loads(frame[6:])["choices"][0]
                    if choice["delta"].get("content"):
                        disconnect.set()
                        if spec_version == "2.4":
                            raise OSError("client closed the connection")

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
        "query_string": b"", "headers": [(b"content-type", b"application/json")],
        "server": ("test", 80), "client": ("test", 1234),
    }
    if spec_version == "2.4":
        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(app(scope, receive, send), timeout=3)
    else:
        await asyncio.wait_for(app(scope, receive, send), timeout=3)
    assert disconnect.is_set()
    assert responses[0]["status"] == 200
    added = await wait_for_event(tmp_path / "engine.jsonl", "ADD")
    aborted = await wait_for_event(tmp_path / "engine.jsonl", "ABORT")
    assert added["request_id"] in aborted["request_ids"]
