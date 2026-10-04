"""Interactive chat example running Qwen3 with Ayaka and streaming throughput metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch
from ayaka.configs.serving import ServingConfig
from ayaka.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, load_qwen3_weights
from ayaka.runtime.serving import ServingRuntime
from ayaka.serving.events import ContentDelta, GenerationFailed, GenerationFinished
from ayaka.serving.generation import GenerationPipeline
from ayaka.serving.prepare import GenerationSpec

DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-0.6B"
    / "snapshots"
    / "c1899de289a04d12100db370d81485cdf75e47ca"
)

# Reconfigure standard streams to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ANSI terminal styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass(slots=True)
class GenerationMetrics:
    """Detailed latency and throughput metrics for one generation turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_seconds: float = 0.0
    decode_seconds: float = 0.0
    total_seconds: float = 0.0
    prefill_throughput: float = 0.0
    decode_throughput: float = 0.0
    overall_throughput: float = 0.0
    finish_reason: str = "unknown"

    def format_summary(self) -> str:
        """Return a formatted string displaying throughput and timing breakdown."""
        lines = [
            f"{YELLOW}{BOLD}[Metrics] Throughput & Performance Breakdown:{RESET}",
            f"  {DIM}*{RESET} Prompt Tokens:       {BOLD}{self.prompt_tokens}{RESET} tokens",
            f"  {DIM}*{RESET} Completion Tokens:   {BOLD}{self.completion_tokens}{RESET} tokens "
            f"({DIM}Total:{RESET} {self.total_tokens} tokens)",
            f"  {DIM}*{RESET} Time to First Token: {BOLD}{self.ttft_seconds * 1000:.1f}{RESET} ms "
            f"({CYAN}{self.prefill_throughput:.2f}{RESET} tok/s prefill)",
            f"  {DIM}*{RESET} Decode Time:         {BOLD}{self.decode_seconds:.2f}{RESET} s "
            f"({GREEN}{self.decode_throughput:.2f}{RESET} tok/s decode)",
            f"  {DIM}*{RESET} Total Latency:       {BOLD}{self.total_seconds:.2f}{RESET} s "
            f"({MAGENTA}{self.overall_throughput:.2f}{RESET} tok/s overall)",
            f"  {DIM}*{RESET} Finish Reason:       {DIM}{self.finish_reason}{RESET}",
        ]
        return "\n".join(lines)


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Format a list of chat message dictionaries into Qwen's ChatML prompt format."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


async def generate_response(
    pipeline: GenerationPipeline,
    prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    out: TextIO = sys.stdout,
) -> tuple[str, GenerationMetrics]:
    """Stream generation tokens to output and calculate throughput metrics."""
    spec = GenerationSpec(
        model="qwen3",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    handle = await pipeline.prepare(spec, tenant="default")

    start_time = time.perf_counter()
    first_token_time: float | None = None
    accumulated_text: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "unknown"

    out.write(f"{CYAN}{BOLD}Assistant:{RESET} ")
    out.flush()

    try:
        async for event in pipeline.generate_events(handle):
            if isinstance(event, ContentDelta):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                accumulated_text.append(event.text)
                out.write(event.text)
                out.flush()
            elif isinstance(event, GenerationFinished):
                finish_reason = event.finish_reason
                prompt_tokens = event.usage.prompt_tokens
                completion_tokens = event.usage.completion_tokens
            elif isinstance(event, GenerationFailed):
                out.write(f"\n{YELLOW}[Generation failed: {event.code} - {event.message}]{RESET}\n")
                out.flush()
                finish_reason = f"failed ({event.code})"
    finally:
        out.write("\n")
        out.flush()

    end_time = time.perf_counter()
    total_time = max(end_time - start_time, 1e-6)
    ttft = (first_token_time - start_time) if first_token_time is not None else total_time
    ttft = max(ttft, 1e-6)
    decode_time = (end_time - first_token_time) if first_token_time is not None else 0.0

    total_tokens = prompt_tokens + completion_tokens
    prefill_throughput = prompt_tokens / ttft if ttft > 0 else 0.0
    decode_throughput = completion_tokens / decode_time if decode_time > 0 else 0.0
    overall_throughput = total_tokens / total_time if total_time > 0 else 0.0

    metrics = GenerationMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        ttft_seconds=ttft,
        decode_seconds=decode_time,
        total_seconds=total_time,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        overall_throughput=overall_throughput,
        finish_reason=finish_reason,
    )
    return "".join(accumulated_text), metrics


def build_runtime(
    model_path: Path,
    *,
    device: str = "auto",
    max_model_len: int = 2048,
) -> tuple[ServingRuntime, GenerationPipeline]:
    """Load Qwen3 weights and initialize Ayaka ServingRuntime and GenerationPipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_path}")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config = Qwen3Config.from_dict(config_data)

    target_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    if target_device.startswith("cuda"):
        backend = "triton"
        model_backend = "triton"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        backend = "reference"
        model_backend = "torch"
        dtype = torch.float32

    print(
        f"{DIM}Loading Qwen3 ({config.num_hidden_layers} layers, {config.hidden_size} dim) "
        f"on {target_device.upper()} with backend='{backend}', dtype={dtype}...{RESET}"
    )

    model = Qwen3ForCausalLM(
        config,
        device=target_device,
        dtype=dtype,
        backend=model_backend,
    )
    load_qwen3_weights(model, model_path, device=target_device)
    model.eval()

    # Determine pages ensuring (pages - 1) * page_size >= max_model_len
    page_size = 16
    min_pages = (max_model_len + page_size - 1) // page_size + 1
    pages = max(128, min_pages + 16)
    batch_tokens = min(max_model_len, 512)
    prefill_chunk = min(batch_tokens, 128)

    runtime = ServingRuntime(
        model,
        model_path,
        config=ServingConfig(
            model="qwen3",
            structured_outputs=False,
        ),
        pages=pages,
        page_size=page_size,
        max_requests=2,
        batch_tokens=batch_tokens,
        prefill_chunk=prefill_chunk,
        max_model_len=max_model_len,
        backend=backend,
    )
    pipeline = GenerationPipeline(runtime.service, runtime.processor)
    return runtime, pipeline


async def chat_loop(
    pipeline: GenerationPipeline,
    *,
    system_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
) -> None:
    """Run an interactive conversation loop in the terminal."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print(f"\n{GREEN}{BOLD}=== Qwen3 Interactive Chat (Ayaka Engine) ==={RESET}")
    print(f"{DIM}Type your message and press Enter.")
    print("Commands: /clear (reset history), /exit or /quit (quit), /help (commands){RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            break
        if cmd in ("/clear", "/reset"):
            messages.clear()
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print(f"{YELLOW}Conversation history cleared.{RESET}\n")
            continue
        if cmd == "/help":
            print(f"{DIM}Available commands:")
            print("  /clear, /reset  - Clear conversation context")
            print("  /exit, /quit    - Exit the chat application")
            print("  /help           - Show this help message{RESET}\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = format_chat_prompt(messages)

        try:
            assistant_reply, metrics = await generate_response(
                pipeline,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            messages.append({"role": "assistant", "content": assistant_reply})
            print()
            print(metrics.format_summary())
            print()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[Generation cancelled by user]{RESET}\n")
        except Exception as exc:
            print(f"\n{YELLOW}[Error during generation: {exc}]{RESET}\n")


async def async_main() -> None:
    """Parse CLI arguments, initialize engine, and start single-shot or interactive chat."""
    # Enable Windows ANSI colors if possible
    if os.name == "nt":
        os.system("")

    parser = argparse.ArgumentParser(
        description="Run Qwen3 using Ayaka with real-time throughput metrics and interactive chat."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to the Qwen3 safetensors model snapshot directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional single prompt to run once without entering interactive chat mode.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="System prompt to guide the model tone and persona.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for execution ('auto', 'cuda', or 'cpu').",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generation tokens per response.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum context length ceiling allocated in the KV cache.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling parameter.",
    )

    args = parser.parse_args()

    runtime, pipeline = build_runtime(
        args.model_path,
        device=args.device,
        max_model_len=args.max_model_len,
    )

    try:
        if args.prompt is not None:
            # Single-shot prompt mode
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": args.prompt})
            prompt = format_chat_prompt(messages)

            print(f"\n{GREEN}{BOLD}Prompt:{RESET} {args.prompt}\n")
            _, metrics = await generate_response(
                pipeline,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print()
            print(metrics.format_summary())
        else:
            # Interactive chat REPL mode
            await chat_loop(
                pipeline,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
    finally:
        print(f"{DIM}Shutting down Ayaka runtime and releasing KV caches...{RESET}")
        runtime.close()


def main() -> None:
    """Entry point for the script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
