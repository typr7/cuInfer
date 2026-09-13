import asyncio
from pathlib import Path

import httpx
import pytest

from cllm.engine import EngineClient, EngineConfig
from cllm.server import create_app
from cllm.tokenizer import load_model


@pytest.mark.gpu
async def test_real_engine_chat_completion():
    pytest.importorskip("cllm._C", exc_type=ImportError)
    model_path = Path(__file__).resolve().parents[1] / "models" / "qwen3-0.6b"
    if not model_path.is_dir():
        pytest.skip("local Qwen3-0.6B model is unavailable")
    model, tokenizer = load_model(str(model_path))
    engine = await asyncio.wait_for(
        EngineClient.start(EngineConfig(model_path=model.path, gpu_memory_utilization=0.9)),
        timeout=60,
    )
    try:
        app = create_app(engine, tokenizer, model.name, model.max_model_len, model.vocab_size)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await asyncio.wait_for(client.post(
                "/v1/chat/completions", json={
                    "model": model.name,
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "temperature": 0, "max_tokens": 128,
                },
            ), timeout=120)
        assert response.status_code == 200, response.text
        choice = response.json()["choices"][0]
        assert choice["message"]["content"].strip()
        assert choice["finish_reason"] in {"stop", "length"}
    finally:
        await engine.close()
