"""CUDA-graph backends: memory pool, GC control, output-structure
helpers, and the capture/replay strategies themselves
(FullGraphBackend, BreakableGraphBackend) plus the optional exec-dedup
registry.

Division of labor, preserved from the original files:
  - GraphMemoryPool / freeze_gc: shared setup every backend needs.
  - row_count / slice_rows: shared output-slicing helpers; backends
    return raw (padded) output, callers slice to the real batch size.
  - FullGraphBackend: one torch.cuda.CUDAGraph per ShapeKey. The
    default per Ayaka's "đúng → đủ → nhanh" roadmap.
  - BreakableGraphBackend: segmented capture via explicit Spans /
    EagerSpan for attention/prefix-caching paths that cannot be
    recorded start-to-finish. Original design for this codebase, not
    a port of any existing engine's internals — validate on a real
    GPU before trusting it in production.
  - DedupedExecRegistry: optional, off by default, needs cuda-python.
    Not wired into either backend above by default; integrating it is
    a follow-up step for whoever's buckets show enough structural
    repeats to be worth it.
"""

from __future__ import annotations

import heapq
import importlib
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from ayaka.runner.graph.graph import CanRun, GraphBackend, GraphCapabilityError, ShapeKey
from ayaka.utils.gc_control import freeze_gc

try:
    cuda_drv: Any = importlib.import_module("cuda.bindings.driver")
    cuda_rt: Any = importlib.import_module("cuda.bindings.runtime")
except ImportError:  # pragma: no cover - optional dependency
    cuda_drv = None
    cuda_rt = None


logger = logging.getLogger(__name__)


class GraphMemoryPool:
    """Lazily creates one torch.cuda.graph_pool_handle() per device
    and hands the same handle back to every caller on that device.

    Deliberately a process-global registry (not per-runner or
    per-backend state): the whole point is for a decode runner, a
    prefill runner, and any future graph runner sharing a device to
    end up in the same pool. A runner-owned pool would defeat that.
    """

    _pools: dict[torch.device, Any] = {}

    @classmethod
    def get(cls, device: torch.device) -> Any:
        pool = cls._pools.get(device)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            cls._pools[device] = pool
        return pool

    @classmethod
    def reset(cls, device: torch.device | None = None) -> None:
        """Test-only: drop the cached handle(s) so the next get()
        creates a fresh pool. Never call this while any backend still
        holds graphs captured against the old pool — they would start
        writing into memory the new pool considers free."""
        if device is None:
            cls._pools.clear()
        else:
            cls._pools.pop(device, None)


def row_count(output: Any, cap: int) -> int:
    """How many leading-dim rows `output` actually carries, clamped to
    `cap`. A structure that shards/prunes along dim 0 in a sub-branch
    (pipeline parallel, MoE token drop) reports the minimum across
    branches; anything with no meaningful dim 0 is treated as fully
    valid (`cap`)."""
    if torch.is_tensor(output):
        return min(cap, output.shape[0])
    if isinstance(output, Mapping):
        values = [v for v in output.values() if v is not None]
        return min([cap, *(row_count(v, cap) for v in values)]) if values else cap
    if isinstance(output, (list, tuple)) and output:
        return min(row_count(o, cap) for o in output if o is not None)
    return cap


def slice_rows(output: Any, n: int) -> Any:
    """View onto the first n rows of every tensor leaf — same
    address, smaller logical size; never copies."""
    if output is None:
        return None
    if torch.is_tensor(output):
        return output[:n]
    if isinstance(output, Mapping):
        mapping_cls: Any = type(output)
        return mapping_cls({k: slice_rows(v, n) for k, v in output.items()})
    if isinstance(output, tuple) and hasattr(output, "_fields"):
        return type(output)(*(slice_rows(v, n) for v in output))
    if isinstance(output, tuple):
        return tuple(slice_rows(o, n) for o in output)
    if isinstance(output, list):
        return [slice_rows(o, n) for o in output]
    raise TypeError(f"unsupported graph output leaf type: {type(output)}")


class GraphCaptureFailed(RuntimeError):
    """Wraps whatever torch/CUDA raised during capture with guidance
    that's actually actionable — a bare CUDA error at this layer is
    almost always OOM from an undersized memory pool, an eager-mode
    allocation still alive when capture started, or a host-syncing op
    inside forward_fn that capture cannot record."""

    def __init__(self, shape_key: ShapeKey, cause: BaseException) -> None:
        super().__init__(
            f"CUDA graph capture failed for {shape_key}. Common causes: "
            f"(1) GPU OOM — the capture pool plus every other live "
            f"allocation must fit at once; lower the largest capture "
            f"bucket or --max-running-requests. (2) an op in forward_fn "
            f"is not capture-safe (host sync, data-dependent control flow, "
            f"or a tensor whose address is expected to change every call — "
            f"see GraphBufferArena). (3) two runners captured against "
            f"different memory pools on the same device — see "
            f"GraphMemoryPool. Underlying error: {cause!r}"
        )
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class CapturedMeta:
    """What a shape_key was captured *with* — checked against what's
    requested at replay time so a mismatch raises instead of silently
    replaying the wrong thing (invariant I4)."""

    hidden_mode: str | None = None


class FullGraphBackend(GraphBackend):
    def __init__(
        self,
        *,
        device: torch.device,
        enable_gc_freeze: bool = True,
        barrier_fn: Callable[[], None] | None = None,
        warmup_iters: int = 2,
    ) -> None:
        """barrier_fn: called after each warmup iteration when set —
        pass a TP/DP barrier here in a multi-rank run, so every rank
        finishes warming up a shape before any rank starts capturing
        it. Capture itself must never include a collective; a
        recorded NCCL op needs its buffers pre-registered with the
        graph pool, which is out of scope for this module."""
        self._device = device
        self._pool = GraphMemoryPool.get(device)
        self._enable_gc_freeze = enable_gc_freeze
        self._barrier_fn = barrier_fn
        self._warmup_iters = warmup_iters
        self._graphs: dict[ShapeKey, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[ShapeKey, Any] = {}
        self._captured_meta: dict[ShapeKey, CapturedMeta] = {}
        self._capture_stream: torch.cuda.Stream | None = None

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        self._capture_stream = stream
        with freeze_gc(self._enable_gc_freeze):
            try:
                yield
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        post_warmup_hook: Callable[[], None] | None = None,
        *,
        hidden_mode: str | None = None,
    ) -> None:
        if self._capture_stream is None:
            raise RuntimeError("capture_one() called outside capture_session()")

        with torch.cuda.stream(self._capture_stream):
            for _ in range(self._warmup_iters):
                torch.cuda.synchronize(self._device)
                if self._barrier_fn is not None:
                    self._barrier_fn()
                forward_fn()  # warmup output is never the captured one — only
                # kernel JIT / one-time setup needs to happen before capture
                if post_warmup_hook is not None:
                    post_warmup_hook()
        torch.cuda.synchronize(self._device)  # clean, idle stream before capture

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, pool=self._pool, stream=self._capture_stream):
                out = forward_fn()
        except Exception as e:  # noqa: BLE001 — re-raised with capture-specific context
            raise GraphCaptureFailed(shape_key, e) from e

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out
        self._captured_meta[shape_key] = CapturedMeta(hidden_mode=hidden_mode)

    def can_run(self, shape_key: ShapeKey, *, requested_hidden_mode: str | None = None) -> CanRun:
        if shape_key not in self._graphs:
            return CanRun.NOT_CAPTURED
        captured = self._captured_meta[shape_key]
        if (
            requested_hidden_mode is not None
            and captured.hidden_mode is not None
            and requested_hidden_mode != captured.hidden_mode
        ):
            return CanRun.CAPABILITY_MISMATCH
        return CanRun.RUNNABLE

    @contextmanager
    def replay_session(self):
        yield

    def replay(self, shape_key: ShapeKey, **kwargs: Any) -> Any:
        result = self.can_run(shape_key, requested_hidden_mode=kwargs.get("requested_hidden_mode"))
        if result is CanRun.CAPABILITY_MISMATCH:
            raise GraphCapabilityError(
                f"{shape_key} was captured with a different hidden_mode than "
                f"requested at replay — this is a routing bug (the caller "
                f"must call can_run() before replay(), not after), not a "
                f"missing bucket."
            )
        if result is CanRun.NOT_CAPTURED:
            raise KeyError(f"{shape_key} was never captured; caller must check can_run() first")
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._captured_meta.clear()

    def stats(self) -> dict[str, int]:
        """Not part of the GraphBackend contract — extra introspection
        this concrete backend happens to offer."""
        return {"captured_shapes": len(self._graphs)}


@dataclass(frozen=True, slots=True)
class EagerSpan:
    """A span that must run for real on every capture pass AND every
    replay — never recorded into a graph."""

    fn: Callable[[], Any]


Span = Callable[[], Any] | EagerSpan


@dataclass(frozen=True, slots=True)
class Spans:
    """What forward_fn must return for this backend. Must have the
    same length, and the same positions must be EagerSpan, on every
    call for a given shape_key — capture_one records that shape once;
    replay() re-checks it every call and raises GraphCapabilityError
    on mismatch rather than silently misaligning segments with the
    wrong eager/captured slot (I4)."""

    items: Sequence[Span]


ForwardFn = Callable[[], Spans]


@dataclass(slots=True)
class _CapturedShape:
    segments: list[torch.cuda.CUDAGraph] = field(default_factory=list)
    is_eager: list[bool] = field(default_factory=list)
    # Only set when the LAST span was captured: the graph's own output
    # tensor(s), refreshed in place by that segment's replay(). Sliced to
    # output_cap on every replay — never copied, its address is stable.
    captured_output: Any = None
    output_cap: int = 0


class BreakableGraphBackend(GraphBackend):
    def __init__(
        self,
        *,
        device: torch.device,
        enable_gc_freeze: bool = True,
        barrier_fn: Callable[[], None] | None = None,
        warmup_iters: int = 2,
    ) -> None:
        self._device = device
        self._pool = GraphMemoryPool.get(device)
        self._enable_gc_freeze = enable_gc_freeze
        self._barrier_fn = barrier_fn
        self._warmup_iters = warmup_iters
        self._shapes: dict[ShapeKey, _CapturedShape] = {}
        self._capture_stream: torch.cuda.Stream | None = None

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        self._capture_stream = stream
        with freeze_gc(self._enable_gc_freeze):
            try:
                yield
            finally:
                self._capture_stream = None

    @staticmethod
    def _run(span: Span) -> Any:
        return span.fn() if isinstance(span, EagerSpan) else span()

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: ForwardFn,
        post_warmup_hook: Callable[[], None] | None = None,
    ) -> None:
        if self._capture_stream is None:
            raise RuntimeError("capture_one() called outside capture_session()")

        with torch.cuda.stream(self._capture_stream):
            for _ in range(self._warmup_iters):
                torch.cuda.synchronize(self._device)
                if self._barrier_fn is not None:
                    self._barrier_fn()
                for span in forward_fn().items:
                    self._run(span)
                if post_warmup_hook is not None:
                    post_warmup_hook()
        torch.cuda.synchronize(self._device)

        spans = forward_fn().items
        if not spans:
            raise ValueError(f"{shape_key}: forward_fn returned an empty Spans")

        segments: list[torch.cuda.CUDAGraph] = []
        is_eager: list[bool] = []
        last_out: Any = None
        for i, span in enumerate(spans):
            if isinstance(span, EagerSpan):
                last_out = span.fn()
                is_eager.append(True)
            else:
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, pool=self._pool, stream=self._capture_stream):
                        last_out = span()
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(
                        f"{shape_key}: capture failed at span {i}/{len(spans)}"
                    ) from e
                segments.append(graph)
                is_eager.append(False)

        if is_eager[-1]:
            shape = _CapturedShape(segments=segments, is_eager=is_eager)
        else:
            cap = row_count(last_out, shape_key.size)
            shape = _CapturedShape(
                segments=segments, is_eager=is_eager, captured_output=last_out, output_cap=cap
            )
        self._shapes[shape_key] = shape

    def can_run(self, shape_key: ShapeKey, *, requested_hidden_mode: str | None = None) -> CanRun:
        return CanRun.RUNNABLE if shape_key in self._shapes else CanRun.NOT_CAPTURED

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self, shape_key: ShapeKey, *, forward_fn: ForwardFn | None = None, **kwargs: Any
    ) -> Any:
        shape = self._shapes.get(shape_key)
        if shape is None:
            raise KeyError(f"{shape_key} was never captured; caller must check can_run() first")
        if forward_fn is None:
            raise GraphCapabilityError(
                f"{shape_key}: BreakableGraphBackend.replay() needs forward_fn "
                f"passed again every call — its eager spans must run for real, "
                f"not just once at capture time."
            )

        spans = forward_fn().items
        if len(spans) != len(shape.is_eager):
            raise GraphCapabilityError(
                f"{shape_key}: span count changed since capture "
                f"({len(spans)} vs {len(shape.is_eager)}) — routing bug: "
                f"a shape_key's forward_fn must not restructure its spans "
                f"across calls, this is not a missing-bucket case."
            )

        graph_iter = iter(shape.segments)
        last_out: Any = None
        for i, (span, was_eager) in enumerate(zip(spans, shape.is_eager, strict=False)):
            if isinstance(span, EagerSpan) != was_eager:
                raise GraphCapabilityError(
                    f"{shape_key}: span {i} changed capture/eager kind since "
                    f"capture — routing bug, not a missing bucket."
                )
            if isinstance(span, EagerSpan):
                last_out = span.fn()
            else:
                next(graph_iter).replay()

        if shape.is_eager[-1]:
            cap = row_count(last_out, shape_key.size)
            return slice_rows(last_out, cap)
        return slice_rows(shape.captured_output, shape.output_cap)

    def cleanup(self) -> None:
        self._shapes.clear()


def available() -> bool:
    return cuda_drv is not None and cuda_rt is not None


def _check(result):
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver/runtime call failed: {err}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def _kernel_name(params) -> str:
    for handle, getter in (
        (getattr(params, "kern", None), cuda_drv.cuKernelGetName),
        (getattr(params, "func", None), cuda_drv.cuFuncGetName),
    ):
        if handle is None or int(handle) == 0:
            continue
        err, name = getter(handle)
        if int(err) == 0:
            return name.decode("utf-8", "replace")
    return f"func:{int(getattr(params, 'func', 0))}"


def _node_payload(node) -> tuple[str, tuple]:
    node_type = _check(cuda_drv.cuGraphNodeGetType(node))
    if node_type == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
        params = _check(cuda_drv.cuGraphKernelNodeGetParams(node))
        payload = (
            _kernel_name(params),
            (int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)),
            (int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ)),
            int(params.sharedMemBytes),
        )
    elif node_type == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY:
        params = _check(cuda_drv.cuGraphMemcpyNodeGetParams(node))
        payload = (int(params.srcMemoryType), int(params.dstMemoryType))
    elif node_type == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET:
        params = _check(cuda_drv.cuGraphMemsetNodeGetParams(node))
        payload = (int(params.elementSize),)
    elif node_type == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
        child = _check(cuda_drv.cuGraphChildGraphNodeGetGraph(node))
        payload = graph_signature(child)
    else:
        payload = ()
    return (node_type.name, payload)


def graph_signature(raw_graph: int) -> tuple:
    """A structural fingerprint of `raw_graph`, invariant to which
    specific addresses/values its kernels were launched with. Node
    order is normalized via a topological sort with a deterministic
    tie-break (heapq over node index) so two graphs built by the same
    code path always hash the same way regardless of any
    nondeterminism in CUDA's own node-enumeration order.
    """
    if not available():
        raise RuntimeError("cuda-python is not installed; call available() first")

    _, num_nodes = _check(cuda_drv.cuGraphGetNodes(raw_graph, 0))
    nodes, _ = _check(cuda_drv.cuGraphGetNodes(raw_graph, num_nodes))
    index_of = {int(n): i for i, n in enumerate(nodes)}

    _, _, _, num_edges = _check(cuda_drv.cuGraphGetEdges(raw_graph, 0))
    src, dst, _, _ = _check(cuda_drv.cuGraphGetEdges(raw_graph, num_edges))
    edges = [(index_of[int(s)], index_of[int(d)]) for s, d in zip(src, dst, strict=False)]

    children: list[list[int]] = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    for s, d in edges:
        children[s].append(d)
        indegree[d] += 1

    ready = [i for i, deg in enumerate(indegree) if deg == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for c in sorted(children[i]):
            indegree[c] -= 1
            if indegree[c] == 0:
                heapq.heappush(ready, c)
    if len(order) != len(nodes):
        raise RuntimeError("graph contains a dependency cycle — cannot fingerprint")

    topo = {n: i for i, n in enumerate(order)}
    topo_edges = tuple(sorted((topo[s], topo[d]) for s, d in edges))
    return (tuple(_node_payload(nodes[i]) for i in order), topo_edges)


@dataclass(slots=True)
class _ExecGroup:
    exec_handle: int
    current_raw_graph: int
    members: int = 1


class DedupedExecRegistry:
    """One instance per capture_session. Call register() once per
    captured torch.cuda.CUDAGraph, in capture order. Call replay()
    instead of graph.replay() for anything register() was called on.
    close() releases every cudaGraphExec_t this registry owns — call
    it from the owning backend's cleanup().
    """

    def __init__(self) -> None:
        if not available():
            raise RuntimeError(
                "cuda-python is not installed; check available() before constructing this"
            )
        self._groups: dict[tuple, _ExecGroup] = {}
        self._group_of: dict[int, _ExecGroup] = {}
        self._raw_graph_of: dict[int, int] = {}

    def register(self, graph: torch.cuda.CUDAGraph) -> None:
        if not hasattr(graph, "raw_cuda_graph"):
            raise RuntimeError(
                "this torch build's CUDAGraph has no raw_cuda_graph() — dedup "
                "needs torch.cuda.CUDAGraph(keep_graph=True) support"
            )
        raw_graph = graph.raw_cuda_graph()
        sig = graph_signature(raw_graph)

        group = self._groups.get(sig)
        if group is None:
            exec_handle = _check(cuda_rt.cudaGraphInstantiateWithFlags(raw_graph, 0))
            group = _ExecGroup(exec_handle=exec_handle, current_raw_graph=raw_graph)
            self._groups[sig] = group
        else:
            if not self._update(group.exec_handle, raw_graph):
                raise RuntimeError(
                    "cudaGraphExecUpdate failed for two graphs with the same "
                    "structural signature — either the signature function "
                    "missed a distinguishing property, or this pair genuinely "
                    "cannot share an exec; disable dedup for this backend"
                )
            group.current_raw_graph = raw_graph
            group.members += 1

        self._group_of[id(graph)] = group
        self._raw_graph_of[id(graph)] = raw_graph

    @staticmethod
    def _update(exec_handle: int, raw_graph: int) -> bool:
        err, info = cuda_rt.cudaGraphExecUpdate(exec_handle, raw_graph)
        if info is None:
            return False
        return (
            int(err) == int(cuda_rt.cudaError_t.cudaSuccess)
            and info.result == cuda_rt.cudaGraphExecUpdateResult.cudaGraphExecUpdateSuccess
        )

    def replay(self, graph: torch.cuda.CUDAGraph, stream: int | None = None) -> None:
        group = self._group_of.get(id(graph))
        if group is None:
            raise KeyError("replay() called on a graph this registry never register()ed")
        raw_graph = self._raw_graph_of[id(graph)]
        if group.current_raw_graph != raw_graph:
            if not self._update(group.exec_handle, raw_graph):
                raise RuntimeError(
                    "cudaGraphExecUpdate failed during replay — this captured "
                    "graph is no longer compatible with its dedup group"
                )
            group.current_raw_graph = raw_graph
        stream_handle = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        _check(cuda_rt.cudaGraphLaunch(group.exec_handle, stream_handle))

    def stats(self) -> tuple[int, int]:
        """(total graphs registered, distinct exec groups) — the ratio
        is the dedup payoff."""
        return sum(g.members for g in self._groups.values()), len(self._groups)

    def close(self) -> None:
        for group in self._groups.values():
            cuda_rt.cudaGraphExecDestroy(group.exec_handle)
        self._groups.clear()
        self._group_of.clear()
        self._raw_graph_of.clear()
