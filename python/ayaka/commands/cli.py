"""Launch a native Ayaka HTTP server from a local QWen, Qwen3, GPT-2, Phi or LLaMA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ayaka")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve", help="serve a native QWen, Qwen3, GPT-2, Phi or LLaMA safetensors checkpoint"
    )
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
    serve.add_argument(
        "--max-admitted-requests",
        type=int,
        default=0,
        help="service handle ceiling; 0 derives no extra bound beyond the scheduler",
    )
    serve.add_argument("--max-model-len", type=int, default=None)
    serve.add_argument("--default-request-timeout", type=float, default=None)
    serve.add_argument("--batch-tokens", type=int, default=256)
    serve.add_argument("--prefill-chunk", type=int, default=128)
    serve.add_argument("--max-concurrent-requests", type=int, default=0)
    serve.add_argument("--api-key", help="comma-separated keys; defaults to AYAKA_API_KEY")
    serve.add_argument("--reasoning-parser", choices=("none", "think"), default="none")
    serve.add_argument("--tool-parser", choices=("none", "hermes"), default="none")
    serve.add_argument("--disable-structured-outputs", action="store_true")
    serve.add_argument(
        "--decode-graph",
        action="store_true",
        help="capture pure-decode CUDA graphs at bootstrap (CUDA + triton backend only)",
    )
    serve.add_argument(
        "--graph-buckets",
        default=None,
        help="comma-separated padded decode batch sizes; defaults to a power-of-two ladder",
    )
    serve.add_argument(
        "--graph-pool-bytes",
        type=int,
        default=0,
        help="explicit graph-private reserve; 0 auto-sizes it on the triton backend",
    )
    args = parser.parse_args(argv)
    import torch
    import uvicorn

    from ayaka.configs.serving import ServingConfig
    from ayaka.models.gpt2 import GPT2Config, GPT2ForCausalLM, load_gpt2_weights
    from ayaka.models.llama import LlamaConfig, LlamaForCausalLM, load_llama_weights
    from ayaka.models.phi import PhiConfig, PhiForCausalLM, load_phi_weights
    from ayaka.models.qwen import QwenConfig, QwenForCausalLM, load_qwen_weights
    from ayaka.models.qwen2 import Qwen2Config, Qwen2ForCausalLM, load_qwen2_weights
    from ayaka.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, load_qwen3_weights
    from ayaka.runtime.serving import ServingRuntime

    values = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    model_type = values.get("model_type")
    dtype = getattr(
        torch, args.dtype or ("float16" if args.device.startswith("cuda") else "float32")
    )
    backend = "torch" if args.backend == "reference" else "triton"
    if model_type == "qwen":
        model = QwenForCausalLM(
            QwenConfig.from_dict(values), device=args.device, dtype=dtype, backend=backend
        )
        load_qwen_weights(model, args.model_path)
    elif model_type == "qwen3":
        model = Qwen3ForCausalLM(
            Qwen3Config.from_dict(values), device=args.device, dtype=dtype, backend=backend
        )
        load_qwen3_weights(model, args.model_path)
    elif model_type == "qwen2":
        model = Qwen2ForCausalLM(
            Qwen2Config.from_dict(values), device=args.device, dtype=dtype, backend=backend
        )
        load_qwen2_weights(model, args.model_path)
    elif model_type == "gpt2":
        model = GPT2ForCausalLM(
            GPT2Config.from_dict(values), device=args.device, dtype=dtype, backend=backend
        )
        load_gpt2_weights(model, args.model_path)
    elif model_type == "phi":
        model = PhiForCausalLM(
            PhiConfig.from_dict(values), device=args.device, dtype=dtype, backend=backend
        )
        load_phi_weights(model, args.model_path)
    elif model_type == "llama":
        model = LlamaForCausalLM(
            LlamaConfig.from_dict(values), device=args.device, dtype=dtype, backend=backend
        )
        load_llama_weights(model, args.model_path)
    else:
        parser.error(
            "this native runner currently supports model_type in "
            "{qwen, qwen2, qwen3, gpt2, phi, llama}"
        )
    model.eval()
    keys = args.api_key if args.api_key is not None else os.getenv("AYAKA_API_KEY", "")
    graph_buckets = None
    if args.graph_buckets:
        try:
            graph_buckets = tuple(sorted({int(part) for part in args.graph_buckets.split(",")}))
        except ValueError:
            parser.error("--graph-buckets must be a comma-separated list of positive integers")
        if not graph_buckets or graph_buckets[0] < 1:
            parser.error("--graph-buckets must contain positive integers")
    runtime = ServingRuntime(
        model,
        args.tokenizer or args.model_path,
        config=ServingConfig(
            model=args.served_model_name,
            api_keys=tuple(k.strip() for k in keys.split(",") if k.strip()),
            max_concurrent_requests=args.max_concurrent_requests,
            max_admitted_requests=args.max_admitted_requests,
            default_request_timeout_seconds=args.default_request_timeout,
            reasoning_parser=args.reasoning_parser,
            tool_parser=args.tool_parser,
            structured_outputs=not args.disable_structured_outputs,
            decode_graph=args.decode_graph,
            graph_buckets=graph_buckets,
        ),
        pages=args.kv_pages,
        page_size=args.page_size,
        max_requests=args.max_requests,
        batch_tokens=args.batch_tokens,
        prefill_chunk=args.prefill_chunk,
        max_model_len=args.max_model_len,
        backend=args.backend,
        graph_pool_bytes=args.graph_pool_bytes,
    )
    try:
        uvicorn.run(runtime.app(), host=args.host, port=args.port, workers=1)
    finally:
        if not runtime.close():
            raise RuntimeError("engine did not drain; KV storage remains retained")


if __name__ == "__main__":
    main()
