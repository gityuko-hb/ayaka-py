"""A real serving stack on CPU for end-to-end tests: no GPU, network or checkpoint.

``CpuServing`` builds an offline word-level tokenizer and a two-layer Qwen2 with
seeded random weights, then the production ``ServingRuntime`` on the reference
attention backend, driven through ``GenerationPipeline`` exactly like HTTP. Every
owner is real: service owner thread, engine, scheduler, KV allocator, prefix cache,
``LocalWorker`` and completion coordinator. Only the hardware differs, so these
tests prove host contracts and output parity, never GPU timing or CUDA safety.

Tokens are ``w0``..``w123`` (ids 4..127); ids 0..3 are ``<unk> <s> </s> <pad>``
and ``</s>`` (id 2) is EOS.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from ayaka.configs.serving import ServingConfig
from ayaka.model_loader.dummy import DUMMY_ARCH_JSON
from ayaka.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from ayaka.runtime.serving import ServingRuntime
from ayaka.serving.events import (
    ContentDelta,
    GenerationFailed,
    GenerationFinished,
    ServingEvent,
)
from ayaka.serving.generation import GenerationPipeline
from ayaka.serving.prepare import GenerationSpec

SPECIALS = ("<unk>", "<s>", "</s>", "<pad>")
EOS_ID = 2
MODEL = "tiny"


def word(index: int) -> str:
    """Surface form of word ``index``; its token id is ``index + len(SPECIALS)``."""
    return f"w{index}"


def prompt_of(*indices: int) -> str:
    return " ".join(word(i) for i in indices)


def build_tokenizer(directory: Path, vocab_size: int) -> Path:
    """Write a whitespace word-level fast tokenizer with ``vocab_size`` entries."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [word(i) for i in range(vocab_size - len(SPECIALS))]
    vocab = {token: index for index, token in enumerate((*SPECIALS, *words))}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.WordPiece(prefix="##")  # joins words with spaces
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        # Chat requests render as their message texts, so a chat prompt equals
        # the completion prompt of the same words.
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}",
    ).save_pretrained(directory)
    return directory


def tiny_config(**overrides: Any) -> dict[str, Any]:
    config = dict(DUMMY_ARCH_JSON)
    config["torch_dtype"] = "float32"
    config.update(overrides)
    return config


def build_model(*, seed: int = 0, scale: float = 4.0, **overrides: Any) -> Qwen2ForCausalLM:
    """Seeded random weights.

    ``scale=4`` makes greedy continuations depend on the whole context (about one
    distinct token per step) with argmax margins far above float32 noise, so a
    position, KV or batching bug changes the text instead of hiding behind a
    model that repeats one token.
    """
    model = Qwen2ForCausalLM(
        Qwen2Config.from_dict(tiny_config(**overrides)),
        device="cpu",
        dtype=torch.float32,
        backend="torch",
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _name, parameter in sorted(model.named_parameters()):
            if parameter.dim() == 1:  # norm gains stay near one
                parameter.copy_(1.0 + 0.1 * torch.randn(parameter.shape, generator=generator))
            else:
                std = scale / parameter.shape[-1] ** 0.5
                parameter.copy_(std * torch.randn(parameter.shape, generator=generator))
    return model.eval()


@dataclass
class Generation:
    text: str
    finish: GenerationFinished | GenerationFailed | None
    events: list[ServingEvent] = field(default_factory=list)

    @property
    def finish_reason(self) -> str | None:
        if isinstance(self.finish, GenerationFinished):
            return self.finish.finish_reason
        return None

    @property
    def completion_tokens(self) -> int:
        if isinstance(self.finish, GenerationFinished):
            return self.finish.usage.completion_tokens
        return -1


class CpuServing:
    """One ``ServingRuntime`` on CPU plus helpers to generate through it.

    Use as a context manager; ``close`` asserts the runtime drained.
    """

    def __init__(
        self,
        *,
        model: Qwen2ForCausalLM | None = None,
        seed: int = 0,
        config: ServingConfig | None = None,
        tokenizer_dir: Path | None = None,
        **runtime: Any,
    ) -> None:
        self.model = model or build_model(seed=seed)
        self._tmp = None
        if tokenizer_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="ayaka-cpu-e2e-")
            tokenizer_dir = build_tokenizer(Path(self._tmp.name), self.model.config.vocab_size)
        self.tokenizer_dir = tokenizer_dir
        options: dict[str, Any] = {
            "pages": 64,
            "page_size": 16,
            "max_requests": 4,
            "batch_tokens": 64,
            "prefill_chunk": 32,
            "max_model_len": 256,
            "backend": "reference",
            "detect_hardware_snapshot": False,
            "tokenizer_workers": 1,
        }
        options.update(runtime)
        self.config = config or ServingConfig(model=MODEL, structured_outputs=False)
        self.runtime = ServingRuntime(self.model, tokenizer_dir, config=self.config, **options)
        assert self.runtime.service is not None
        self.service = self.runtime.service
        self.pipeline = GenerationPipeline(self.service, self.runtime.processor)
        self._closed = False

    # -- generation ------------------------------------------------------

    def spec(self, prompt: str | list[int], **kwargs: Any) -> GenerationSpec:
        kwargs.setdefault("temperature", 0.0)
        return GenerationSpec(model=self.config.model, prompt=prompt, **kwargs)

    async def agenerate(
        self, prompt: str | list[int], *, cancel_after: int | None = None, **kwargs: Any
    ) -> Generation:
        """Stream one generation; ``cancel_after`` cancels after that many deltas."""
        handle = await self.pipeline.prepare(self.spec(prompt, **kwargs), tenant="default")
        result = Generation("", None)
        deltas = 0
        async for event in self.pipeline.generate_events(handle):
            result.events.append(event)
            if isinstance(event, ContentDelta):
                result.text += event.text
                deltas += 1
                if cancel_after is not None and deltas == cancel_after:
                    handle.cancel()
            elif isinstance(event, (GenerationFinished, GenerationFailed)):
                result.finish = event
        return result

    def generate(self, prompt: str | list[int], **kwargs: Any) -> Generation:
        return asyncio.run(asyncio.wait_for(self.agenerate(prompt, **kwargs), 60))

    def generate_concurrently(self, prompts: list[str], **kwargs: Any) -> list[Generation]:
        async def all_of() -> list[Generation]:
            return list(await asyncio.gather(*(self.agenerate(p, **kwargs) for p in prompts)))

        return asyncio.run(asyncio.wait_for(all_of(), 120))

    def http(self):
        """A Starlette test client on the production HTTP app over this stack.

        The app's lifespan is not entered, so this stack keeps ownership of
        shutdown; ``close`` drains it as usual.
        """
        from ayaka.serving.http import create_app
        from starlette.testclient import TestClient

        return TestClient(create_app(self.service, self.runtime.processor, close=lambda: True))

    # -- oracle ----------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        assert self.runtime.tokenizer is not None and self.runtime.tokenizer.tokenizer is not None
        return list(self.runtime.tokenizer.tokenizer.encode(text, add_special_tokens=True))

    def decode(self, ids: list[int]) -> str:
        assert self.runtime.tokenizer is not None and self.runtime.tokenizer.tokenizer is not None
        return self.runtime.tokenizer.tokenizer.decode(ids, skip_special_tokens=True)

    def dense_greedy(self, prompt: str, tokens: int) -> list[int]:
        """Greedy ids from the model's dense forward: no KV cache, pages or scheduler."""
        sequence = self.encode(prompt)
        generated: list[int] = []
        with torch.inference_mode():
            for _ in range(tokens):
                logits = self.model.forward_dense(torch.tensor(sequence))
                token = int(torch.argmax(logits[-1]))
                generated.append(token)
                sequence.append(token)
        return generated

    # -- accounting ------------------------------------------------------

    def memory(self):
        """Page accounting of the bound KV slab (thread-safe snapshot)."""
        assert self.runtime.kv is not None
        return self.runtime.kv.snapshot()

    def assert_quiescent_accounting(self) -> None:
        """No request owns a page and every partition sums to the slab total."""
        assert self.runtime.kv is not None
        backend = self.runtime.kv.backend
        backend.assert_invariants()
        snap = self.memory()
        assert (
            snap.free_pages
            + snap.reserved_pages
            + snap.live_pages
            + snap.deferred_free_pages
            + snap.permanent_pages
            == snap.total_pages
        ), snap
        assert snap.request_owned_pages == 0, snap
        assert snap.reserved_pages == 0 and snap.inflight_pages == 0, snap
        assert snap.live_sequences == 0 and snap.active_leases == 0, snap

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            assert self.runtime.close(), "CPU serving runtime did not drain"
        finally:
            if self._tmp is not None:
                self._tmp.cleanup()

    def __enter__(self) -> CpuServing:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
