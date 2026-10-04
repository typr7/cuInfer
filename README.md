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
- CUDA Graph
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
| 1 | 245.8 | **357.8** | **18.40** | 34.77 |
| 8 | 1605.6 | **1883.4** | **29.17** | 88.79 |
| 32 | 3457.4 | **3633.5** | **42.70** | 212.33 |
| 128 | **5213.6** | 5120.0 | **193.72** | 704.60 |
| 256 | **5499.3** | 5186.4 | **399.58** | 1382.04 |
