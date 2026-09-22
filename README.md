# cuInfer

cuInfer is a large language model inference framework written in C++20 and CUDA. It
serves a Hugging Face checkpoint behind an OpenAI-compatible HTTP API: a Python
frontend owns HTTP, tokenization and streaming, while a separate C++/CUDA process
owns the GPU, the KV cache and the scheduler.

## Features

- OpenAI-compatible HTTP API
- Continuous batching with chunked prefill
- Paged KV cache
- Prefix caching
- Preemptive scheduling
- Hand-written CUDA kernels
- Request abort
- Supported Models: Qwen3-0.6B, Llama3.2-1B

## Architecture

```
        HTTP client
              │  OpenAI HTTP (streaming or not)
              ▼
   ┌─────────────────────── Python process ────────────────────────┐
   │  uvicorn + FastAPI          cuinfer/server.py                 │
   │  tokenize / detokenize      cuinfer/tokenizer.py              │
   │  EngineClient  ──── spawns ──────────────┬                    │
   └──────────┬───────────────────────────────┼────────────────────┘
              │  ZeroMQ PUSH/PULL over ipc:// │
              ▼                               ▼
   ┌────────────────── EngineCore process (C++/CUDA) ──────────────┐
   │  Scheduler → ModelRunner → CausalLM → ops::* → sampler        │
   └───────────────────────────────────────────────────────────────┘
```

## Build

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate # make sure cmake uses venv's python

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -G Ninja
cmake --build build
cmake --install build --prefix .

pip install -e ".[dev]"
```

## Run

```bash
source .venv/bin/activate
python -m cuinfer serve --model models/qwen3-0.6b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Say hi"}],
       "max_tokens": 64, "temperature": 0}'
```

## Benchmark

```bash
bash benchmark.sh {cuinfer/vllm}
```

The results below compare cuInfer and vLLM on Qwen3-0.6B with a single
A100-PCIE-40GB:

| Concurrency | cuInfer (tok/s) | vLLM (tok/s) | cuInfer TTFT p50 (ms) | vLLM TTFT p50 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 245.8 | 357.8 | 18.40 | 34.77 |
| 8 | 1536.6 | 1865.4 | 31.97 | 81.34 |
| 32 | 3479.1 | 3807.6 | 56.68 | 192.70 |
| 128 | 5074.4 | 5209.0 | 272.09 | 563.22 |
| 256 | 5550.6 | 5419.0 | 514.20 | 851.86 |
