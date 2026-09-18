"""Launch a native Ayaka HTTP server from a local QWen checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ayaka")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve a native QWen safetensors checkpoint")
    serve.add_argument("model_path", type=Path)
    serve.add_argument("--tokenizer", type=Path)
    serve.add_argument("--served-model-name", default="ayaka")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="cuda")
    serve.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    serve.add_argument("--backend", choices=("triton", "reference"), default="triton")
    serve.add_argument("--kv-pages", type=int, default=1024)
    serve.add_argument("--page-size", type=int, default=16)
    serve.add_argument("--max-requests", type=int, default=32)
    serve.add_argument("--batch-tokens", type=int, default=256)
    serve.add_argument("--prefill-chunk", type=int, default=128)
    serve.add_argument("--max-concurrent-requests", type=int, default=0)
    serve.add_argument("--api-key", help="comma-separated keys; defaults to AYAKA_API_KEY")
    serve.add_argument("--reasoning-parser", choices=("none", "think"), default="none")
    serve.add_argument("--tool-parser", choices=("none", "hermes"), default="none")
    serve.add_argument("--disable-structured-outputs", action="store_true")
    args = parser.parse_args(argv)
    import torch
    import uvicorn

    from ayaka.configs.serving import ServingConfig
    from ayaka.models.qwen import QwenConfig, QwenForCausalLM, load_qwen_weights
    from ayaka.runtime.serving import ServingRuntime

    values = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    if values.get("model_type") != "qwen":
        parser.error("this native runner currently supports model_type=qwen (original QWen)")
    config = QwenConfig.from_dict(values)
    model = QwenForCausalLM(
        config,
        device=args.device,
        dtype=getattr(
            torch, args.dtype or ("float16" if args.device.startswith("cuda") else "float32")
        ),
        backend="torch" if args.backend == "reference" else "triton",
    )
    load_qwen_weights(model, args.model_path)
    model.eval()
    keys = args.api_key if args.api_key is not None else os.getenv("AYAKA_API_KEY", "")
    runtime = ServingRuntime(
        model,
        args.tokenizer or args.model_path,
        config=ServingConfig(
            model=args.served_model_name,
            api_keys=tuple(k.strip() for k in keys.split(",") if k.strip()),
            max_concurrent_requests=args.max_concurrent_requests,
            reasoning_parser=args.reasoning_parser,
            tool_parser=args.tool_parser,
            structured_outputs=not args.disable_structured_outputs,
        ),
        pages=args.kv_pages,
        page_size=args.page_size,
        max_requests=args.max_requests,
        batch_tokens=args.batch_tokens,
        prefill_chunk=args.prefill_chunk,
        backend=args.backend,
    )
    try:
        uvicorn.run(runtime.app(), host=args.host, port=args.port, workers=1)
    finally:
        if not runtime.close():
            raise RuntimeError("engine did not drain; KV storage remains retained")


if __name__ == "__main__":
    main()
