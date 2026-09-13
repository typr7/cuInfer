import argparse
import asyncio
import sys

import uvicorn

from cuinfer.engine import EngineClient, EngineConfig, EngineDied
from cuinfer.server import create_app
from cuinfer.tokenizer import load_model


async def serve(args: argparse.Namespace) -> int:
    info, tokenizer = load_model(args.model)
    config = EngineConfig(
        model_path=info.path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        max_num_scheduled_tokens=args.max_num_scheduled_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    engine = await EngineClient.start(config)
    server = uvicorn.Server(uvicorn.Config(
        create_app(engine, tokenizer, info.name, info.max_model_len, info.vocab_size),
        host=args.host,
        port=args.port,
        workers=1,
        timeout_graceful_shutdown=5,
    ))

    async def watch_engine() -> None:
        error = await engine.wait_dead()
        print(error, file=sys.stderr)
        server.should_exit = True

    watchdog = asyncio.create_task(watch_engine())
    try:
        await server.serve()
        return 1 if engine.error is not None else 0
    finally:
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m cuinfer")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("serve")
    command.add_argument("--model", required=True)
    command.add_argument("--host", default="0.0.0.0")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    command.add_argument("--block-size", type=int, default=16)
    command.add_argument("--max-num-scheduled-tokens", type=int, default=8192)
    command.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()
    try:
        EngineConfig(
            model_path=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            block_size=args.block_size,
            max_num_scheduled_tokens=args.max_num_scheduled_tokens,
            max_num_seqs=args.max_num_seqs,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        sys.exit(asyncio.run(serve(args)))
    except EngineDied as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
