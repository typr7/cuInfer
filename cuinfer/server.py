import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from anyio import CancelScope
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jinja2 import TemplateError
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import Receive, Scope, Send

from cuinfer.engine import EngineClient, EngineDied
from cuinfer.protocol import FinishReason
from cuinfer.tokenizer import IncrementalDetokenizer


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    include_usage: bool = False


class CompletionParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0, le=3.4028234663852886e38)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=2**31 - 1)
    ignore_eos: bool = False
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    presence_penalty: float = Field(default=0, ge=0, le=0)
    frequency_penalty: float = Field(default=0, ge=0, le=0)
    repetition_penalty: Literal[1.0] = 1.0
    logprobs: None = None
    user: Any = None
    stream_options: StreamOptions | None = None
    store: Any = None
    metadata: Any = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None = None


class ChatRequest(CompletionParams):
    messages: list[ChatMessage] = Field(min_length=1)


class TextRequest(CompletionParams):
    prompt: str | list[int]


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "server_error" if status_code == 503
                   else "invalid_request_error", "param": None, "code": None}},
        status_code=status_code,
    )


def sse(data: dict | str) -> str:
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"


class CompletionStream(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # ASGI 2.4 reports disconnects through send(), outside the iterator.
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


def create_app(
    engine: EngineClient,
    tokenizer: Any,
    model_name: str,
    max_model_len: int,
    vocab_size: int | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    if vocab_size is None:
        vocab_size = len(tokenizer)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> Response:
        return error_response(str(exc), 400)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Response:
        return error_response(str(exc.detail), exc.status_code)

    @app.exception_handler(EngineDied)
    async def engine_error(request: Request, exc: EngineDied) -> Response:
        return error_response(str(exc), 503)

    def require_engine() -> None:
        if not engine.alive:
            raise HTTPException(503, str(engine.error or "Engine is not running"))

    @app.get("/health")
    async def health() -> dict:
        require_engine()
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        require_engine()
        return {"object": "list", "data": [{
            "id": model_name, "object": "model", "created": 0, "owned_by": "cuinfer",
        }]}

    async def complete(body: ChatRequest | TextRequest) -> Response:
        require_engine()
        if body.model is not None and body.model != model_name:
            raise HTTPException(404, f"Model {body.model!r} is not served")
        chat = isinstance(body, ChatRequest)
        try:
            if chat:
                token_ids = tokenizer.apply_chat_template(
                    [message.model_dump(exclude_none=True) for message in body.messages],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            elif isinstance(body.prompt, str):
                token_ids = tokenizer.encode(body.prompt)
            else:
                token_ids = body.prompt
        except (ValueError, TypeError, TemplateError) as exc:
            raise HTTPException(400, str(exc)) from exc

        prompt_tokens = len(token_ids)
        max_tokens = body.max_tokens if body.max_tokens is not None else max_model_len - prompt_tokens
        if not 0 < prompt_tokens < max_model_len:
            raise HTTPException(400, f"Prompt must contain between 1 and {max_model_len - 1} tokens")
        if prompt_tokens + max_tokens > max_model_len:
            raise HTTPException(400, f"Prompt plus max_tokens exceeds {max_model_len} tokens")
        if any(token < 0 or token >= vocab_size for token in token_ids):
            raise HTTPException(400, "Prompt contains a token id outside the model vocabulary")

        request_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        decoder = IncrementalDetokenizer(tokenizer, token_ids)
        outputs = engine.generate(request_id, token_ids, {
            "max_output_tokens": max_tokens,
            "temperature": body.temperature,
            "top_k": body.top_k,
            "top_p": body.top_p,
            "ignore_eos": body.ignore_eos,
        })
        base = {"id": request_id, "created": created, "model": model_name}

        async def cleanup() -> None:
            # Starlette's disconnect cancellation must not interrupt ABORT.
            with CancelScope(shield=True):
                await outputs.aclose()
                await engine.abort(request_id)

        def chunk(text: str = "", finish_reason: str | None = None, role: bool = False) -> dict:
            if chat:
                delta = {"role": "assistant"} if role else ({"content": text} if text else {})
                choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            else:
                choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}
            return {**base, "object": "chat.completion.chunk" if chat else "text_completion",
                    "choices": [choice]}

        async def stream() -> AsyncIterator[str]:
            completion_tokens = 0
            try:
                if chat:
                    yield sse(chunk(role=True))
                async for output in outputs:
                    completion_tokens += len(output["new_token_ids"])
                    reason = FinishReason(output["finish_reason"])
                    delta = decoder.push(output["new_token_ids"], reason)
                    finish_reason = None if reason == FinishReason.RUNNING else (
                        "stop" if reason == FinishReason.STOP else "length"
                    )
                    yield sse(chunk(delta, finish_reason=finish_reason))
                if body.stream_options is not None and body.stream_options.include_usage:
                    yield sse({
                        **base,
                        "object": "chat.completion.chunk" if chat else "text_completion",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    })
                yield sse("[DONE]")
            except EngineDied as exc:
                yield sse({"error": {"message": str(exc), "type": "server_error"}})
                yield sse("[DONE]")
            finally:
                await cleanup()

        if body.stream:
            return CompletionStream(stream(), media_type="text/event-stream")

        text_parts = []
        completion_tokens = 0
        try:
            async for output in outputs:
                completion_tokens += len(output["new_token_ids"])
                reason = FinishReason(output["finish_reason"])
                text_parts.append(decoder.push(output["new_token_ids"], reason))
            finish_reason = "stop" if reason == FinishReason.STOP else "length"
            text = "".join(text_parts)
            choice = {"index": 0, "finish_reason": finish_reason}
            if chat:
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice.update(text=text, logprobs=None)
            return JSONResponse({
                **base,
                "object": "chat.completion" if chat else "text_completion",
                "choices": [choice],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            })
        finally:
            await cleanup()

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest) -> Response:
        return await complete(body)

    @app.post("/v1/completions")
    async def text_completions(body: TextRequest) -> Response:
        return await complete(body)

    return app
