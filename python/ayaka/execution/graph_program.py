"""Explicit model segment boundaries for serving graph strategies."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from ayaka.execution.forward_context import ForwardContext, forward_context
from ayaka.runner.graph.backend import EagerSpan, Spans


@dataclass(frozen=True, slots=True)
class CompilerCacheKey:
    """Compiler identity; executable CUDA graphs remain resource-local."""

    model: str
    binding: str
    framework: str
    compiler: str
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    capability: tuple[int, int]
    options: tuple[str, ...] = ("inductor", "fullgraph", "static", "cudagraphs=False")

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()


@dataclass
class SegmentedProgram:
    spans: Spans
    compiler_key: CompilerCacheKey | None = None
    compile_seconds: float = 0.0

    def __call__(self) -> Spans:
        return self.spans


def compiler_report(program: Callable[[], Any]) -> dict[str, str | float] | None:
    if not isinstance(program, SegmentedProgram) or program.compiler_key is None:
        return None
    return {"key": program.compiler_key.digest, "compile_seconds": program.compile_seconds}


def model_program(
    owner: Any,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    metadata: dict,
    *,
    backend: str,
    compile_seconds: int = 120,
) -> Callable[[], Any]:
    """Return a persistent program; eager/graph bridges outlive every replay.

    Piecewise v1 compiles the logits piece, with the model body an explicit
    eager span. It does not claim whole-model compiler or CUDA coverage.
    Compiler warmup is bootstrap-only; subsequent guard misses raise.
    """
    if backend == "full":
        return owner._model_forward(tokens, positions, metadata)
    context = ForwardContext(owner.backends, owner.layers, metadata)
    hidden: list[torch.Tensor] = []

    def body() -> torch.Tensor:
        variant = getattr(owner, "model_variant_context", lambda tokens: nullcontext())
        with torch.inference_mode(), forward_context(context), variant(tokens):
            output = owner._model.forward_hidden(tokens, positions, context.attention)
            if backend == "torch_compile_piecewise" and hidden:
                hidden[0].copy_(output)
            else:
                hidden[:] = [output]
            return hidden[0]

    project = owner._model.logits_from_hidden
    compiled = backend == "torch_compile_piecewise"
    if compiled:
        project = torch.compile(
            project,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
    warmed = False

    def logits() -> torch.Tensor:
        nonlocal warmed
        with torch.inference_mode():
            if compiled and not warmed:
                import triton

                program.compiler_key = CompilerCacheKey(
                    type(owner._model).__qualname__,
                    owner._graph_binding_digest(),
                    torch.__version__,
                    triton.__version__,
                    tuple(hidden[0].shape),
                    str(hidden[0].dtype),
                    hidden[0].stride(),
                    torch.cuda.get_device_capability(owner.device),
                )
                started = time.monotonic()
                output = project(hidden[0])
                program.compile_seconds = time.monotonic() - started
                if program.compile_seconds > compile_seconds:
                    raise TimeoutError("piecewise compile exceeded bootstrap time budget")
                warmed = True
                return output
            if compiled:
                with torch.compiler.set_stance("fail_on_recompile"):
                    return project(hidden[0])
            return project(hidden[0])

    spans = Spans((EagerSpan(body), logits) if compiled else (body, EagerSpan(logits)))
    program = SegmentedProgram(spans)
    return program
