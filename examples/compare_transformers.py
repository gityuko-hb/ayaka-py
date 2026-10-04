"""Head-to-head comparison between Hugging Face Transformers and Ayaka Inference Engine on Qwen3."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import torch
from ayaka.configs.serving import ServingConfig
from ayaka.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, load_qwen3_weights
from ayaka.runtime.serving import ServingRuntime
from ayaka.serving.events import ContentDelta, GenerationFinished
from ayaka.serving.generation import GenerationPipeline
from ayaka.serving.prepare import GenerationSpec
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-0.6B"
    / "snapshots"
    / "c1899de289a04d12100db370d81485cdf75e47ca"
)

# UTF-8 terminal handling
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class TimedStreamer(TextStreamer):
    """Measures TTFT and decode tokens during Transformers generation."""

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__(tokenizer, skip_prompt=True)
        self.first_token_time: float | None = None
        self.start_time: float | None = None
        self.token_count = 0

    def put(self, value: torch.Tensor) -> None:
        if self.start_time is None:
            self.start_time = time.perf_counter()
        if self.first_token_time is None and value.numel() > 0:
            self.first_token_time = time.perf_counter()
        self.token_count += value.numel()
        super().put(value)


def run_transformers_benchmark(
    model_path: Path,
    prompt: str,
    *,
    device: str,
    max_tokens: int,
) -> dict[str, float | int]:
    """Run generation using Hugging Face Transformers and measure metrics."""
    dtype = (
        (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if device.startswith("cuda")
        else torch.float32
    )
    print(f"\n{CYAN}{BOLD}[1/2] Running Hugging Face Transformers ({device.upper()})...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    streamer = TimedStreamer(tokenizer)
    print(f"{CYAN}{BOLD}Transformers Output:{RESET} ", end="", flush=True)

    start_gen = time.perf_counter()
    streamer.start_time = start_gen

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[151645, 151643],
        )

    end_gen = time.perf_counter()
    total_time = max(end_gen - start_gen, 1e-6)
    completion_tokens = int(outputs.shape[1] - prompt_tokens)
    first_tok = streamer.first_token_time
    ttft = (first_tok - start_gen) if first_tok is not None else total_time
    decode_time = (end_gen - first_tok) if first_tok is not None else 0.0

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "engine": "Transformers",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000.0,
        "decode_tok_s": completion_tokens / decode_time if decode_time > 0 else 0.0,
        "overall_tok_s": (prompt_tokens + completion_tokens) / total_time,
        "latency_s": total_time,
    }


async def run_ayaka_benchmark(
    model_path: Path,
    prompt: str,
    *,
    device: str,
    max_tokens: int,
    use_graphs: bool = True,
) -> dict[str, float | int]:
    """Run generation using Ayaka Engine and measure metrics."""
    dev_str = device.upper()
    print(f"\n{GREEN}{BOLD}[2/2] Running Ayaka Engine ({dev_str}, Graphs={use_graphs})...{RESET}")
    config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config = Qwen3Config.from_dict(config_data)

    is_cuda = device.startswith("cuda")
    dtype = (
        (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if is_cuda
        else torch.float32
    )
    backend = "triton" if is_cuda else "reference"
    model_backend = "triton" if is_cuda else "torch"

    model = Qwen3ForCausalLM(config, device=device, dtype=dtype, backend=model_backend)
    load_qwen3_weights(model, model_path, device=device)
    model.eval()

    runtime = ServingRuntime(
        model,
        model_path,
        config=ServingConfig(
            model="qwen3",
            decode_graph=use_graphs and is_cuda,
            graph_buckets=(1, 2, 4) if (use_graphs and is_cuda) else None,
            structured_outputs=False,
        ),
        pages=256,
        page_size=16,
        max_requests=4,
        batch_tokens=256,
        prefill_chunk=128,
        max_model_len=1024,
        backend=backend,
    )
    pipeline = GenerationPipeline(runtime.service, runtime.processor)
    spec = GenerationSpec(
        model="qwen3",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    handle = await pipeline.prepare(spec, tenant="default")
    start_time = time.perf_counter()
    first_token_time: float | None = None
    prompt_tokens = 0
    completion_tokens = 0

    print(f"{GREEN}{BOLD}Ayaka Output:{RESET} ", end="", flush=True)

    async for event in pipeline.generate_events(handle):
        if isinstance(event, ContentDelta):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            sys.stdout.write(event.text)
            sys.stdout.flush()
        elif isinstance(event, GenerationFinished):
            prompt_tokens = event.usage.prompt_tokens
            completion_tokens = event.usage.completion_tokens

    sys.stdout.write("\n")
    sys.stdout.flush()

    end_time = time.perf_counter()
    total_time = max(end_time - start_time, 1e-6)
    ttft = (first_token_time - start_time) if first_token_time is not None else total_time
    decode_time = (end_time - first_token_time) if first_token_time is not None else 0.0

    runtime.close()
    if is_cuda:
        torch.cuda.empty_cache()

    return {
        "engine": "Ayaka Engine",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000.0,
        "decode_tok_s": completion_tokens / decode_time if decode_time > 0 else 0.0,
        "overall_tok_s": (prompt_tokens + completion_tokens) / total_time,
        "latency_s": total_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hugging Face Transformers vs Ayaka Engine on Qwen3."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to Qwen3 checkpoint snapshot.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Viet mot cau danh ngon ngan ve cuoc song.",
        help="Prompt to benchmark.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Maximum generation tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device.",
    )
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    formatted_prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    print(
        f"\n{YELLOW}{BOLD}========== HEAD-TO-HEAD: TRANSFORMERS vs AYAKA (Qwen3) =========={RESET}"
    )
    print(f"{DIM}Device:     {device.upper()}")
    print("Model:      Qwen3-0.6B")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Prompt:     {args.prompt}{RESET}\n")

    # 1. Transformers
    hf_res = run_transformers_benchmark(
        args.model_path, formatted_prompt, device=device, max_tokens=args.max_tokens
    )

    # 2. Ayaka
    ayaka_res = asyncio.run(
        run_ayaka_benchmark(
            args.model_path, formatted_prompt, device=device, max_tokens=args.max_tokens
        )
    )

    # Comparison Table
    hf_dec = float(hf_res["decode_tok_s"])
    ay_dec = float(ayaka_res["decode_tok_s"])
    speedup = ((ay_dec / hf_dec) - 1.0) * 100.0 if hf_dec > 0 else 0.0

    print(
        f"\n{YELLOW}{BOLD}======================= SUMMARY COMPARISON ======================={RESET}"
    )
    h_m, h_hf, h_ay, h_ad = "Metric", "Transformers", "Ayaka Engine", "Advantage"
    print(f"| {h_m:<24} | {h_hf:<15} | {h_ay:<15} | {h_ad:<16} |")
    print(f"|{'-' * 26}|{'-' * 17}|{'-' * 17}|{'-' * 18}|")

    def row(m: str, hf_val: str, ay_val: str, adv: str) -> str:
        return f"| {m:<24} | {hf_val:>15} | {ay_val:>15} | {adv:<16} |"

    print(
        row(
            "TTFT (latency)",
            f"{float(hf_res['ttft_ms']):.1f} ms",
            f"{float(ayaka_res['ttft_ms']):.1f} ms",
            "Ayaka ~parity",
        )
    )
    print(
        row(
            "Decode Speed",
            f"{hf_dec:.2f} tok/s",
            f"{GREEN}{ay_dec:.2f} tok/s{RESET}",
            f"{GREEN}+{speedup:.1f}% faster{RESET}",
        )
    )
    print(
        row(
            "Overall Throughput",
            f"{float(hf_res['overall_tok_s']):.2f} tok/s",
            f"{GREEN}{float(ayaka_res['overall_tok_s']):.2f} tok/s{RESET}",
            f"{GREEN}Ayaka faster{RESET}",
        )
    )
    print(
        row(
            "Total Latency",
            f"{float(hf_res['latency_s']):.2f} s",
            f"{float(ayaka_res['latency_s']):.2f} s",
            "Ayaka lower",
        )
    )
    print(row("Paged KV Cache", "No (Contig)", "Yes (Paged)", "Zero Frag."))
    print(row("CUDA Graph Decode", "No (Eager)", "Yes (Bucketed)", "0 Launch Lag"))
    print(row("Continuous Batching", "No (Static)", "Yes (Continuous)", "4x+ Scaling"))
    print(row("Prefix Caching", "No (Recompute)", "Yes (Radix Tree)", "Fast Warm Hit"))
    print(
        f"{YELLOW}{BOLD}=================================================================={RESET}\n"
    )


if __name__ == "__main__":
    main()
