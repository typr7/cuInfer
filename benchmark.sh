#!/usr/bin/env bash
set -euo pipefail

ENGINE=${1:-}
if [[ "$ENGINE" != "cuinfer" && "$ENGINE" != "vllm" ]]; then
    echo "Usage: $0 {cuinfer|vllm}" >&2
    exit 2
fi

CUINFER_DIR=/path/to/cuinfer
VLLM_DIR=/path/to/vllm
MODEL=$CUINFER_DIR/models/qwen3-0.6b
MODEL_NAME=qwen3-0.6b
PORT=8000
RESULT_DIR=$CUINFER_DIR/benchmark-results/${ENGINE}-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RESULT_DIR/raw" "$RESULT_DIR/logs"

SERVER_PID=
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID"
        wait "$SERVER_PID" || true
    fi
}
trap stop_server EXIT

if [[ "$ENGINE" == "cuinfer" ]]; then
    (
        cd "$CUINFER_DIR"
        exec .venv/bin/python -m cuinfer serve \
            --model "$MODEL" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --disable-access-log \
            --gpu-memory-utilization 0.9 \
            --block-size 16 \
            --max-num-scheduled-tokens 8192 \
            --max-num-seqs 256
    ) >"$RESULT_DIR/logs/server.log" 2>&1 &
else
    (
        cd "$VLLM_DIR"
        exec .venv/bin/vllm serve "$MODEL" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --dtype bfloat16 \
            --max-model-len 40960 \
            --served-model-name "$MODEL_NAME" \
            --generation-config vllm \
            --gpu-memory-utilization 0.9 \
            --block-size 16 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 256 \
            --no-enable-prefix-caching \
            --disable-uvicorn-access-log
    ) >"$RESULT_DIR/logs/server.log" 2>&1 &
fi
SERVER_PID=$!

until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        cat "$RESULT_DIR/logs/server.log" >&2
        exit 1
    fi
    sleep 1
done

for repetition in 1 2 3; do
    for concurrency in 1 8 32 128 256; do
        name=${ENGINE}_i1024_o128_c${concurrency}_r${repetition}
        "$VLLM_DIR/.venv/bin/vllm" bench serve \
            --backend openai \
            --base-url "http://127.0.0.1:$PORT" \
            --endpoint /v1/completions \
            --model "$MODEL_NAME" \
            --tokenizer "$MODEL" \
            --dataset-name random \
            --random-input-len 1024 \
            --random-output-len 128 \
            --random-range-ratio 0 \
            --random-prefix-len 0 \
            --num-prompts 1000 \
            --num-warmups 20 \
            --request-rate inf \
            --max-concurrency "$concurrency" \
            --temperature 0 \
            --ignore-eos \
            --seed 20260914 \
            --percentile-metrics ttft,tpot,e2el \
            --metric-percentiles 50,90,99 \
            --disable-tqdm \
            --save-result \
            --save-detailed \
            --result-dir "$RESULT_DIR/raw" \
            --result-filename "$name.json" \
            >"$RESULT_DIR/logs/$name.log" 2>&1
    done
done

stop_server
trap - EXIT
echo "Results: $RESULT_DIR"
