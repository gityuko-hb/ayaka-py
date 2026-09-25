"""Standalone, single-flight CUDA graph runner for one-token decode.

Ayaka's serving path uses :class:`DecodeGraphPool` because it also owns flight
slots, admission and KV generations. This smaller runner is for a model whose
decode call already accepts flat, batch-leading attention metadata. The caller
is responsible for reserving a harmless KV page/slot for every dummy lane.
"""

from __future__ import annotations

import bisect
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch


class DecodeModel(Protocol):
    """Graph-safe one-token forward; returns ``[batch, ...]`` logits."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache_ptrs: torch.Tensor,
        attention_metadata: Mapping[str, torch.Tensor],
        kv_cache: Any,
    ) -> torch.Tensor: ...


InputBuffers = Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]]


@dataclass(slots=True)
class _Bucket:
    graph: torch.cuda.CUDAGraph
    inputs: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]
    logits: torch.Tensor


class CudaGraphRunner:
    """Capture one graph per decode batch bucket and replay on the caller's stream.

    ``capture`` takes CUDA templates named ``input_ids``, ``positions``,
    ``kv_cache_ptrs``, ``attention_metadata`` (a mapping of flat tensors), and
    ``out_logits``. Each tensor has a leading dimension of ``max_batch_size``.
    Template rows are copied into each bucket before warmup/capture and serve
    as the safe padding values on replay. Padded slot/page indices must point
    to reserved KV storage (distinct write slots unless the kernel masks stores).
    Zero sequence lengths are safe only if the backend supports empty rows;
    otherwise use a valid dummy sequence. No dummy may write live KV data.

    The returned logits are a view of a reusable static output. Consume or
    copy them on the replay stream before calling ``replay`` again. Calls are
    serialized across CPU threads and CUDA streams, but this runner cannot
    track consumers launched later on unrelated streams. The caller must keep
    the model, KV allocations, and any backend workspaces at the same addresses
    until ``close``; resizing/offloading them requires a new capture.

    FlashAttention, FlashInfer and PagedAttention can only be captured when
    their selected decode path is graph-safe. Prepare FlashInfer wrappers and
    workspaces before capture. Keep page-table widths, launch grids, tensor
    addresses, and backend dispatch fixed for each bucket. Data-dependent
    Python branches, ``.item()``, host synchronization and CPU planning inside
    the model's captured forward are unsupported.

    Capture must run during an exclusive GPU bootstrap window. Buckets share
    scratch memory and must be independent: no graph-private intermediate may
    become persistent state for another bucket. An external ``mempool`` requires
    the caller to serialize all other users too. Calls from different threads
    may hand off ownership, but consumers must be enqueued before the next call.
    """

    _INPUT_NAMES = ("input_ids", "positions", "kv_cache_ptrs")

    def __init__(
        self,
        model: DecodeModel,
        max_batch_size: int,
        allowed_batch_sizes: Sequence[int] | None = None,
        mempool: Any | None = None,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool):
            raise TypeError("max_batch_size must be an integer")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if not callable(model):
            raise TypeError("model must be callable")
        sizes = (
            self._default_buckets(max_batch_size)
            if allowed_batch_sizes is None
            else tuple(allowed_batch_sizes)
        )
        if not sizes or any(
            not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= max_batch_size
            for size in sizes
        ):
            raise ValueError("allowed_batch_sizes must contain positive sizes <= max_batch_size")
        if len(set(sizes)) != len(sizes):
            raise ValueError("allowed_batch_sizes must not contain duplicates")

        self.model = model
        self.max_batch_size = max_batch_size
        self.allowed_batch_sizes = tuple(sorted(sizes))
        self._pool = mempool
        self._device: torch.device | None = None
        self._padding: dict[str, torch.Tensor] = {}
        self._padding_metadata: dict[str, torch.Tensor] = {}
        self._kv_cache: Any = None
        self._buckets: dict[int, _Bucket] = {}
        self._last_replay: torch.cuda.Event | None = None
        self._last_stream: torch.cuda.Stream | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _default_buckets(max_batch_size: int) -> tuple[int, ...]:
        sizes: set[int] = {max_batch_size}
        scale = 1
        while scale <= max_batch_size:
            sizes.add(scale)
            if scale >= 16 and scale * 3 // 2 <= max_batch_size:
                sizes.add(scale * 3 // 2)
            scale *= 2
        return tuple(sorted(sizes))

    @staticmethod
    def _check_tensor(name: str, tensor: object, batch: int, device: torch.device) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device != device or tensor.ndim < 1 or tensor.shape[0] != batch:
            raise ValueError(f"{name} must be a {device} tensor with {batch} batch rows")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use strided tensor storage")
        return tensor

    def capture(self, input_buffers: InputBuffers, kv_cache: Any) -> None:
        """Warm up three times on a side stream, then capture every bucket.

        Templates must already contain valid dummy values; warmup and capture
        execute the model and may write KV cache. Capture is one-shot. If it
        fails, partial graphs are discarded and the runner can be retried.
        """
        with self._lock:
            if self._buckets:
                raise RuntimeError("graphs are already captured; close before recapturing")
            if isinstance(self.model, torch.nn.Module) and self.model.training:
                raise ValueError("model must be in eval mode before CUDA graph capture")
            if set(input_buffers) != {*self._INPUT_NAMES, "attention_metadata", "out_logits"}:
                raise ValueError("input_buffers must contain the three inputs, metadata and logits")
            input_ids = input_buffers["input_ids"]
            if not isinstance(input_ids, torch.Tensor) or input_ids.device.type != "cuda":
                raise ValueError("input_ids must be a CUDA tensor")
            device = input_ids.device
            templates: dict[str, torch.Tensor] = {}
            for name in (*self._INPUT_NAMES, "out_logits"):
                tensor = input_buffers[name]
                checked = self._check_tensor(name, tensor, self.max_batch_size, device)
                templates[name] = checked.detach().clone()
            raw_metadata = input_buffers["attention_metadata"]
            if not isinstance(raw_metadata, Mapping) or not raw_metadata:
                raise ValueError("attention_metadata must be a non-empty tensor mapping")
            metadata: dict[str, torch.Tensor] = {}
            for name, tensor in raw_metadata.items():
                if not isinstance(name, str):
                    raise TypeError("attention_metadata keys must be strings")
                self._check_tensor(
                    f"attention_metadata[{name}]", tensor, self.max_batch_size, device
                )
                metadata[name] = tensor.detach().clone()

            pool = self._pool if self._pool is not None else torch.cuda.graphs.graph_pool_handle()
            caller_stream = torch.cuda.current_stream(device)
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(caller_stream)
            captured: dict[int, _Bucket] = {}
            try:
                with torch.inference_mode():
                    # Largest first permits graph-private allocations to reuse the pool.
                    for size in reversed(self.allowed_batch_sizes):
                        with torch.cuda.stream(capture_stream):
                            inputs = {
                                name: templates[name][:size].clone() for name in self._INPUT_NAMES
                            }
                            static_metadata = {
                                name: value[:size].clone() for name, value in metadata.items()
                            }
                            logits = torch.empty_like(templates["out_logits"][:size])

                        def forward(
                            inputs: dict[str, torch.Tensor] = inputs,
                            static_metadata: dict[str, torch.Tensor] = static_metadata,
                            logits: torch.Tensor = logits,
                        ) -> None:
                            result = self.model(
                                inputs["input_ids"],
                                inputs["positions"],
                                inputs["kv_cache_ptrs"],
                                static_metadata,
                                kv_cache,
                            )
                            logits.copy_(result)

                        with torch.cuda.stream(capture_stream):
                            sample = self.model(
                                inputs["input_ids"],
                                inputs["positions"],
                                inputs["kv_cache_ptrs"],
                                static_metadata,
                                kv_cache,
                            )
                            if (
                                not isinstance(sample, torch.Tensor)
                                or sample.shape != logits.shape
                                or sample.dtype != logits.dtype
                                or sample.device != device
                            ):
                                raise ValueError(
                                    f"model logits for bucket {size} must match out_logits "
                                    f"shape, dtype and device: {tuple(logits.shape)}, "
                                    f"{logits.dtype}, {device}"
                                )
                            logits.copy_(sample)
                            del sample
                            for _ in range(2):
                                forward()
                        capture_stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                            forward()
                        captured[size] = _Bucket(graph, inputs, static_metadata, logits)
            except BaseException:
                capture_stream.synchronize()
                raise
            finally:
                caller_stream.wait_stream(capture_stream)

            self._buckets = captured
            self._padding = templates
            self._padding_metadata = metadata
            self._kv_cache = kv_cache
            self._device = device
            self._pool = pool
            # Keep startup dependencies even when the first replay changes stream.
            self._last_replay = torch.cuda.Event()
            self._last_replay.record(capture_stream)
            self._last_stream = capture_stream

    def replay(
        self,
        current_batch_size: int,
        *dynamic_inputs: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Stage ``(ids, positions, KV pointers, metadata)`` then replay.

        ``current_batch_size`` above the largest captured bucket executes the
        model eagerly with the original inputs. GPU inputs are required; host
        sources may be pinned and copied to CUDA by the caller before replay.
        """
        with self._lock, torch.inference_mode():
            if not self._buckets or self._device is None:
                raise RuntimeError("capture must complete before replay")
            if isinstance(self.model, torch.nn.Module) and self.model.training:
                raise ValueError("model must remain in eval mode after CUDA graph capture")
            if not isinstance(current_batch_size, int) or isinstance(current_batch_size, bool):
                raise TypeError("current_batch_size must be an integer")
            if current_batch_size < 1:
                raise ValueError("current_batch_size must be positive")
            if len(dynamic_inputs) != 4:
                raise TypeError("replay requires ids, positions, KV pointers and metadata")
            raw_inputs: dict[str, torch.Tensor] = {
                name: self._check_tensor(name, tensor, current_batch_size, self._device)
                for name, tensor in zip(self._INPUT_NAMES, dynamic_inputs[:3], strict=True)
            }
            raw_metadata = dynamic_inputs[3]
            if not isinstance(raw_metadata, Mapping) or set(raw_metadata) != set(
                self._padding_metadata
            ):
                raise ValueError("attention_metadata keys must match capture")
            for name, tensor in raw_inputs.items():
                template = self._padding[name]
                if tensor.dtype != template.dtype or tensor.shape[1:] != template.shape[1:]:
                    raise ValueError(f"{name} dtype or trailing shape differs from capture")
            for name, tensor in raw_metadata.items():
                self._check_tensor(
                    f"attention_metadata[{name}]", tensor, current_batch_size, self._device
                )
                template = self._padding_metadata[name]
                if tensor.dtype != template.dtype or tensor.shape[1:] != template.shape[1:]:
                    raise ValueError(f"attention_metadata[{name}] differs from capture")

            stream = torch.cuda.current_stream(self._device)
            assert self._last_replay is not None
            if self._last_stream != stream:
                # Include consumers enqueued AFTER the previous graph launch.
                # Waiting only for replay could race with a clone or sampler.
                self._last_replay.record(self._last_stream)
                stream.wait_event(self._last_replay)
            self._last_stream = stream
            # Protect allocator lifetime, not producer readiness: the caller
            # must first wait for producers running on unrelated streams.
            for tensor in (*raw_inputs.values(), *raw_metadata.values()):
                tensor.record_stream(stream)
            if current_batch_size > self.allowed_batch_sizes[-1]:
                result = self.model(
                    raw_inputs["input_ids"],
                    raw_inputs["positions"],
                    raw_inputs["kv_cache_ptrs"],
                    raw_metadata,
                    self._kv_cache,
                )
                return result

            index = bisect.bisect_left(self.allowed_batch_sizes, current_batch_size)
            bucket = self._buckets[self.allowed_batch_sizes[index]]
            for name, static in bucket.inputs.items():
                static[:current_batch_size].copy_(raw_inputs[name], non_blocking=True)
                if current_batch_size < static.shape[0]:
                    static[current_batch_size:].copy_(
                        self._padding[name][current_batch_size : static.shape[0]], non_blocking=True
                    )
            for name, static in bucket.metadata.items():
                static[:current_batch_size].copy_(raw_metadata[name], non_blocking=True)
                if current_batch_size < static.shape[0]:
                    static[current_batch_size:].copy_(
                        self._padding_metadata[name][current_batch_size : static.shape[0]],
                        non_blocking=True,
                    )
            bucket.graph.replay()
            return bucket.logits[:current_batch_size]

    def close(self) -> None:
        """Release captured graphs after all users of returned logits finish."""
        with self._lock:
            if self._last_stream is not None:
                self._last_stream.synchronize()
            self._buckets.clear()
            self._padding.clear()
            self._padding_metadata.clear()
            self._kv_cache = None
            self._device = None
            self._last_replay = None
            self._last_stream = None
