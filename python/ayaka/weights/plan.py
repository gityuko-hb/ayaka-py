from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass, field
from typing import Final

from ayaka.distributed.device import DeviceRef
from ayaka.exceptions import CheckpointCorruptError
from ayaka.types import DType
from ayaka.weights.spec import ShardKind, TensorSpec, WeightSource, WeightSpec


class CheckpointFormat(enum.StrEnum):
    """How the bytes are stored on disk.

    ``SAFETENSORS_SHARDED`` is a distinct member rather than a flag because the
    discovery path differs: a sharded checkpoint is reached through
    ``model.safetensors.index.json`` and its ``weight_map``, and a loader that
    treats the index as "just another file" reads the wrong thing.
    """

    SAFETENSORS = "safetensors"
    SAFETENSORS_SHARDED = "safetensors_sharded"
    PYTORCH_BIN = "pytorch_bin"
    DUMMY = "dummy"

class LoadState(enum.StrEnum):
    # Bootstrap state machine

    CREATED = "created"
    RESOLVED = "resolved"
    DISCOVERED = "discovered"
    PLANNED = "planned"
    LOADING = "loading"
    BOUND = "bound"
    VERIFIED = "verified"
    WEIGHTS_READY = "weights_ready"
    RUNTIME_READY = "runtime_ready"
    FAILED = "failed"

class WeightTransform(enum.StrEnum):
    """One step of the materialization pipeline.

    Declaration order *is* the execution order.  ``TRANSFORM_ORDER`` freezes it
    and :class:`WeightLoadEntry` rejects any tuple that is not a subsequence of
    it, so "cast before slice" — which silently multiplies staging bytes by the
    world size — cannot be expressed, let alone executed.
    """

    DECODE = "decode"  # storage dtype → a dtype we can address
    SLICE = "slice"  # take this rank's shard, before anything widens it
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    CAST = "cast"  # to the compute dtype, last, on the smallest tensor
    CONTIGUOUS = "contiguous"

TRANSFORM_ORDER: Final[tuple[WeightTransform, ...]] = (
    WeightTransform.DECODE,
    WeightTransform.SLICE,
    WeightTransform.RESHAPE,
    WeightTransform.TRANSPOSE,
    WeightTransform.CAST,
    WeightTransform.CONTIGUOUS,
)

@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One tensor as the *checkpoint* describes it — before any planning.

    ``byte_offset`` is absolute within ``file_uri``, already resolved from the
    safetensors header's data-section base, so a reader can ``pred`` it with no
    further arithmetic and no knowledge of the header format.
    """

    tensor_key: str
    file_uri: str
    dtype: DType
    shape: tuple[int, ...]
    byte_offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if not self.tensor_key:
            raise CheckpointCorruptError("manifest entry with an empty tensor key")
        if not self.file_uri:
            raise CheckpointCorruptError(f"{self.tensor_key}: empty file uri")
        if self.byte_offset < 0 or self.nbytes < 0:
            raise CheckpointCorruptError(
                f"{self.tensor_key}: negative offset/size "
                f"({self.byte_offset}, {self.nbytes}) — truncated or overflowed header"
            )
        if any(d < 0 for d in self.shape):
            raise CheckpointCorruptError(f"{self.tensor_key}: negative dim in {self.shape}")
        declared = self.dtype.nbytes(math.prod(self.shape) if self.shape else 1)
        if declared != self.nbytes:
            raise CheckpointCorruptError(
                f"{self.tensor_key}: header says {self.nbytes} bytes but "
                f"{self.shape}×{self.dtype.label} is {declared}"
            )

    @property
    def end_offset(self) -> int:
        return self.byte_offset + self.nbytes

@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """What the checkpoint provides.  Produced without reading a tensor payload.

    Entries are in *physical* order — ``(file_uri, byte_offset)`` — not the JSON
    or filesystem order they were discovered in.  That makes the manifest
    reproducible across machines and makes the overlap check below a linear
    scan instead of an O(n²) one.
    """

    format: CheckpointFormat
    root_uri: str
    entries: tuple[ManifestEntry, ...] = ()
    files: tuple[str, ...] = ()
    revision: str = ""

    def __post_init__(self) -> None:
        if not self.root_uri:
            raise ValueError("manifest needs a root uri")
        keys: set[str] = set()
        for e in self.entries:
            if e.tensor_key in keys:
                raise CheckpointCorruptError(
                    f"duplicate tensor key {e.tensor_key!r} — two shards claim the same weight"
                )
            keys.add(e.tensor_key)
        ordered = tuple(sorted(self.entries, key=lambda e: (e.file_uri, e.byte_offset)))
        if ordered != self.entries:
            raise ValueError(
                "manifest entries must be sorted by (file_uri, byte_offset); "
                "unsorted input means discovery leaked filesystem or dict order"
            )
        prev: ManifestEntry | None = None
        for e in self.entries:
            if prev is not None and prev.file_uri == e.file_uri and e.byte_offset < prev.end_offset:
                raise CheckpointCorruptError(
                    f"{prev.tensor_key} [{prev.byte_offset}:{prev.end_offset}) overlaps "
                    f"{e.tensor_key} [{e.byte_offset}:{e.end_offset}) in {e.file_uri}"
                )
            prev = e
        if self.files:
            declared = set(self.files)
            used = {e.file_uri for e in self.entries}
            if not used <= declared:
                raise CheckpointCorruptError(
                    f"manifest references files not in the index: {sorted(used - declared)}"
                )

    @property
    def total_bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    @property
    def tensor_keys(self) -> frozenset[str]:
        return frozenset(e.tensor_key for e in self.entries)

    def entry(self, tensor_key: str) -> ManifestEntry | None:
        for e in self.entries:
            if e.tensor_key == tensor_key:
                return e
        return None

@dataclass(frozen=True, slots=True)
class WeightPlacement:
    """Which rank and which device own one shard.

    This is on the *plan entry*, never on ``WeightSpec`` or ``TensorSpec``:
    Rule 9 says the model never sees a rank, and the way to keep that true is to
    make it structurally impossible for a rank to reach the model through a
    tensor description it holds.
    """

    device: DeviceRef = field(default_factory=DeviceRef)
    tp_rank: int = 0
    tp_size: int = 1
    pp_stage: int = 0
    pp_size: int = 1

    # Expert parallelism, added in schema 6.  A genuinely independent axis: a
    # rank can be tp_rank=0 and ep_rank=3 at the same time, so this cannot be
    # folded into the TP fields.  Dense models leave it at 0/1 and pay nothing.
    ep_rank: int = 0
    ep_size: int = 1

    def __post_init__(self) -> None:
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(f"invalid tp rank {self.tp_rank}/{self.tp_size}")
        if self.pp_size < 1 or not 0 <= self.pp_stage < self.pp_size:
            raise ValueError(f"invalid pp stage {self.pp_stage}/{self.pp_size}")
        if self.ep_size < 1 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"invalid ep rank {self.ep_rank}/{self.ep_size}")

    @property
    def is_single_device(self) -> bool:
        return self.tp_size == 1 and self.pp_size == 1 and self.ep_size == 1


@dataclass(frozen=True, slots=True)
class FileReadPlan:
    """One ``pred``.  To decides these *before* opening anything.

    A row shard of a row-major tensor is not one contiguous range: it is
    ``row_count`` runs of ``run_bytes``, ``stride_bytes`` apart.  Expressing that
    here is what stops the loader from materializing the full logical tensor to
    keep one rank's slice — the failure mode that makes a TP=8 load peak at 8×
    the host memory it needs.
    """

    file_uri: str
    byte_offset: int
    nbytes: int
    run_bytes: int = 0  # 0 = one contiguous run of `nbytes`
    stride_bytes: int = 0
    run_count: int = 1

    def __post_init__(self) -> None:
        if not self.file_uri:
            raise ValueError("read plan needs a file uri")
        if self.byte_offset < 0 or self.nbytes <= 0:
            raise ValueError(f"{self.file_uri}: bad read range ({self.byte_offset}, {self.nbytes})")
        if self.run_count < 1:
            raise ValueError(f"{self.file_uri}: run_count must be >= 1")
        if self.run_bytes:
            if self.run_bytes * self.run_count != self.nbytes:
                raise ValueError(
                    f"{self.file_uri}: {self.run_count}×{self.run_bytes} != {self.nbytes}"
                )
            if self.stride_bytes and self.stride_bytes < self.run_bytes:
                raise ValueError(
                    f"{self.file_uri}: stride {self.stride_bytes} < run {self.run_bytes} — "
                    "overlapping runs would read the same bytes twice"
                )
        elif self.run_count != 1:
            raise ValueError(f"{self.file_uri}: contiguous read cannot have run_count > 1")

    @property
    def is_contiguous(self) -> bool:
        return not self.run_bytes or self.run_count == 1

    @property
    def span_bytes(self) -> int:
        """Distance from first byte to last, which is what an mmap must cover —
        larger than ``nbytes`` for a strided read."""
        if self.is_contiguous or not self.stride_bytes:
            return self.nbytes
        return (self.run_count - 1) * self.stride_bytes + self.run_bytes

@dataclass(frozen=True, slots=True)
class TensorSlice:
    """A half-open range along one dimension.  ``[start, stop)``.

    One dimension, not a full multi-dim selector, because every case A1 needs —
    a fused QKV split, a merged gate/up, a TP shard — cuts along exactly one
    axis.  A general N-d selector would be more expressive and would make the
    coverage check below impossible to state.
    """

    dim: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"negative slice dim {self.dim}")
        if self.start < 0 or self.stop < self.start:
            raise ValueError(f"invalid slice [{self.start}, {self.stop}) on dim {self.dim}")

    @property
    def extent(self) -> int:
        return self.stop - self.start

    def covers(self, shape: tuple[int, ...]) -> bool:
        return self.dim < len(shape) and self.start == 0 and self.stop == shape[self.dim]

    @classmethod
    def whole(cls, shape: tuple[int, ...], dim: int = 0) -> TensorSlice:
        return cls(dim=dim, start=0, stop=shape[dim] if shape else 1)


@dataclass(frozen=True, slots=True)
class WeightSourceSlice:
    """Which part of a *physical* checkpoint tensor one logical weight comes from.

    ``source_shape`` is the shape of the tensor as it sits in the file, which is
    NOT the logical weight's shape whenever the checkpoint fuses several weights
    into one tensor.  GLM ships ``query_key_value.weight`` as
    ``[q_total + 2*kv_total, hidden]``; the logical ``k_proj`` is a slice of it.

    Keeping the two shapes in separate fields is what lets
    :class:`~ayaka.protocol.tensor.WeightSpec` keep its invariant intact.  The
    alternative — widening ``WeightSpec.full_shape`` to hold the fused shape and
    skipping the shape check — turns an early, loud failure into silent
    corruption, because nothing downstream would then notice a component reading
    the wrong band.
    """

    source: WeightSource
    source_shape: tuple[int, ...]
    # The dtype the bytes are stored in, which is a property of the *physical*
    # tensor and not of the logical weight.  It has to live here for the same
    # reason `source_shape` does: an fp32 checkpoint feeding a bf16 model has
    # two dtypes, and a materialize that reads the target dtype off
    # `weight.spec` would decode fp32 bytes as bf16 — same byte count for half
    # the elements, so nothing raises and every value is garbage.
    source_dtype: DType
    source_slices: tuple[TensorSlice, ...] = ()

    def __post_init__(self) -> None:
        if not self.source.tensor_key:
            raise ValueError("weight source slice needs a tensor key")
        if any(d < 0 for d in self.source_shape):
            raise ValueError(f"{self.source.tensor_key}: negative dim in {self.source_shape}")
        for sl in self.source_slices:
            if sl.dim >= len(self.source_shape):
                raise ValueError(
                    f"{self.source.tensor_key}: slice dim {sl.dim} but source shape is "
                    f"{self.source_shape}"
                )
            if sl.stop > self.source_shape[sl.dim]:
                raise ValueError(
                    f"{self.source.tensor_key}: slice [{sl.start}, {sl.stop}) runs past "
                    f"dim {sl.dim} of {self.source_shape}"
                )

    @property
    def is_whole_tensor(self) -> bool:
        return not self.source_slices or all(
            sl.covers(self.source_shape) for sl in self.source_slices
        )

    @property
    def selected_shape(self) -> tuple[int, ...]:
        """The shape after the slices are applied."""
        dims = list(self.source_shape)
        for sl in self.source_slices:
            dims[sl.dim] = sl.extent
        return tuple(dims)


@dataclass(frozen=True, slots=True)
class WeightComponent:
    """One logical weight, and where in the checkpoint and in the destination it lives.

    ``weight.full_shape`` is the *logical* shape — Q's shape, not the fused
    QKV's — so ``WeightSpec``'s ``spec.shape == shard.shard_shape(full_shape)``
    check keeps catching real errors on it.
    """

    weight: WeightSpec
    source: WeightSourceSlice
    destination_slice: TensorSlice

    def __post_init__(self) -> None:
        if self.destination_slice.extent != self.weight.spec.shape[self.destination_slice.dim]:
            raise ValueError(
                f"{self.weight.name}: destination slice extent "
                f"{self.destination_slice.extent} != sharded shape "
                f"{self.weight.spec.shape} on dim {self.destination_slice.dim}"
            )

    @property
    def name(self) -> str:
        return self.weight.name

    @property
    def nbytes(self) -> int:
        return self.weight.spec.nbytes


@dataclass(frozen=True, slots=True)
class WeightLoadEntry:
    """One destination buffer, and every logical weight that writes into it.

    It used to be "one weight, one source",
    which cannot describe either direction of the real mapping:

        1 physical tensor -> N logical components   (fused QKV)
        N physical tensors -> 1 destination         (merged gate/up)
        1 logical component -> N reads              (a strided row shard)

    All three are now expressible, and the invariant that replaces the old
    one-to-one assumption is stronger: the components' destination slices must
    *exactly tile* the destination, with no gap and no overlap.  A gap is a band
    of the parameter nobody writes — which survives every shape check and shows
    up as degraded output; an overlap is two weights fighting over the same
    bytes, and the loser depends on iteration order.
    """

    destination: TensorSpec
    placement: WeightPlacement = field(default_factory=WeightPlacement)
    components: tuple[WeightComponent, ...] = ()
    reads: tuple[FileReadPlan, ...] = ()
    transforms: tuple[WeightTransform, ...] = ()
    layer_index: int | None = None
    tied_alias: str = ""

    def __post_init__(self) -> None:
        if not self.destination.name:
            raise ValueError("a load entry's destination must be named")
        seen = -1
        for t in self.transforms:
            i = TRANSFORM_ORDER.index(t)
            if i <= seen:
                raise ValueError(
                    f"{self.name}: transforms {list(self.transforms)} are not a "
                    f"subsequence of the fixed order {list(TRANSFORM_ORDER)}"
                )
            seen = i

        if self.tied_alias:
            if self.components or self.reads:
                raise ValueError(
                    f"{self.name}: tied alias of {self.tied_alias!r} must not read from "
                    "the checkpoint or declare components"
                )
            return
        if not self.components:
            raise ValueError(
                f"{self.name}: entry has no components and is not a tied alias — "
                "a destination with no source is a planning bug, not an empty tensor"
            )
        self._check_tiling()

    def _check_tiling(self) -> None:
        dims = {c.destination_slice.dim for c in self.components}
        if len(dims) > 1:
            raise ValueError(
                f"{self.name}: components tile along more than one dim {sorted(dims)}; "
                "a destination is filled along exactly one axis"
            )
        dim = dims.pop()
        if dim >= len(self.destination.shape):
            raise ValueError(
                f"{self.name}: tiling dim {dim} but destination shape is {self.destination.shape}"
            )
        ordered = sorted(self.components, key=lambda c: c.destination_slice.start)
        cursor = 0
        for component in ordered:
            sl = component.destination_slice
            if sl.start != cursor:
                kind = "gap" if sl.start > cursor else "overlap"
                raise ValueError(
                    f"{self.name}: {kind} at offset {cursor} on dim {dim} — "
                    f"{component.name} starts at {sl.start}. Components must tile the "
                    "destination exactly."
                )
            cursor = sl.stop
        extent = self.destination.shape[dim]
        if cursor != extent:
            raise ValueError(
                f"{self.name}: components cover [0, {cursor}) but the destination "
                f"needs [0, {extent}) on dim {dim}"
            )

    @property
    def name(self) -> str:
        return self.destination.name

    @property
    def read_bytes(self) -> int:
        return sum(r.nbytes for r in self.reads)

    @property
    def charged_bytes(self) -> int:
        """What the ledger is charged.  Padded, and zero for a tied alias, which
        shares storage with its source and must not be counted twice."""
        return 0 if self.tied_alias else self.destination.padded_nbytes

    @property
    def is_tied(self) -> bool:
        return bool(self.tied_alias)

    @property
    def is_replicated(self) -> bool:
        return all(c.weight.shard.kind is ShardKind.REPLICATED for c in self.components)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components)

    @property
    def sort_key(self) -> tuple[int, int, int, int, str]:
        layer = -1 if self.layer_index is None else self.layer_index
        return (
            self.placement.pp_stage,
            self.placement.ep_rank,
            self.placement.tp_rank,
            layer,
            self.name,
        )


@dataclass(frozen=True, slots=True)
class WeightAdmission:
    """The pre-I/O go/no-go (node 1.19).

    Only two device numbers are compared, and neither is an estimate: exact
    padded weight bytes against the capacity the ledger will actually hand out.
    Activation peak is absent on purpose — it is not measurable until A2 runs a
    warmup, and guessing it here is how the "two sources of truth" bug that A0
    just removed comes back.
    """

    device_capacity_bytes: int
    device_charge_bytes: int
    staging_buffer_bytes: int = 0
    staging_buffer_count: int = 0

    def __post_init__(self) -> None:
        for name in ("device_capacity_bytes", "device_charge_bytes", "staging_buffer_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.staging_buffer_count < 0:
            raise ValueError("staging_buffer_count must be >= 0")
        if bool(self.staging_buffer_bytes) != bool(self.staging_buffer_count):
            raise ValueError(
                "staging_buffer_bytes and staging_buffer_count must both be set or both be zero"
            )

    @property
    def host_staging_peak_bytes(self) -> int:
        """Derived, never stored: the peak is the pool, and a stored copy is a
        second number that can disagree with it."""
        return self.staging_buffer_bytes * self.staging_buffer_count

    @property
    def headroom_bytes(self) -> int:
        return self.device_capacity_bytes - self.device_charge_bytes

    @property
    def fits(self) -> bool:
        return self.device_charge_bytes <= self.device_capacity_bytes


@dataclass(frozen=True, slots=True)
class WeightLoadPlan:
    """The whole load, decided.  Hashable via :meth:`digest`, and deterministic
    so that hash means the same thing on every rank."""

    plan_id: str
    manifest: CheckpointManifest
    admission: WeightAdmission
    entries: tuple[WeightLoadEntry, ...] = ()
    schema_version: int = 0

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("weight load plan needs a plan_id")
        names: set[str] = set()
        for e in self.entries:
            if e.name in names:
                raise ValueError(f"duplicate destination name in plan: {e.name!r}")
            names.add(e.name)
        keys = [e.sort_key for e in self.entries]
        if keys != sorted(keys):
            raise ValueError(
                "plan entries must be sorted by (pp_stage, tp_rank, layer_index, name); "
                "an unsorted plan is not reproducible and its digest is meaningless"
            )
        for e in self.entries:
            if e.tied_alias and e.tied_alias not in names:
                raise ValueError(f"{e.name}: tied alias target {e.tied_alias!r} is not in the plan")
        charge = sum(e.charged_bytes for e in self.entries)
        if charge != self.admission.device_charge_bytes:
            raise ValueError(
                f"admission charges {self.admission.device_charge_bytes} B but entries sum to "
                f"{charge} B — admission must be computed from the entries, not alongside them"
            )
        peak = self.admission.staging_buffer_bytes
        if peak:
            largest = max((r.nbytes for e in self.entries for r in e.reads), default=0)
            if largest > peak:
                raise ValueError(
                    f"largest single read is {largest} B but a staging buffer is {peak} B — "
                    "the pipeline would have to split a tensor it cannot address"
                )

    @property
    def total_charged_bytes(self) -> int:
        return self.admission.device_charge_bytes

    @property
    def total_read_bytes(self) -> int:
        """Bytes actually pulled off disk — less than ``manifest.total_bytes``
        whenever this rank owns a shard rather than the whole tensor."""
        return sum(e.read_bytes for e in self.entries)

    @property
    def transform_count(self) -> int:
        return sum(len(e.transforms) for e in self.entries)

    def entries_for_stage(self, pp_stage: int) -> tuple[WeightLoadEntry, ...]:
        return tuple(e for e in self.entries if e.placement.pp_stage == pp_stage)

    def entries_for_expert_rank(self, ep_rank: int) -> tuple[WeightLoadEntry, ...]:
        return tuple(e for e in self.entries if e.placement.ep_rank == ep_rank)

    def entries_for_rank(self, tp_rank: int) -> tuple[WeightLoadEntry, ...]:
        return tuple(e for e in self.entries if e.placement.tp_rank == tp_rank)

    def bytes_for_stage(self, pp_stage: int) -> int:
        return sum(e.charged_bytes for e in self.entries_for_stage(pp_stage))

    def bytes_for_rank(self, tp_rank: int) -> int:
        return sum(e.charged_bytes for e in self.entries_for_rank(tp_rank))

    def digest(self) -> str:
        """Stable across processes and Python runs.

        Deliberately not ``hash()``: PYTHONHASHSEED randomizes string hashing,
        so a built-in hash compared across ranks would disagree for reasons that
        have nothing to do with the plan.
        """
        h = hashlib.sha256()
        h.update(f"{self.schema_version}\x1f{self.manifest.root_uri}\x1f".encode())
        h.update(f"{self.manifest.format}\x1f{self.manifest.revision}\x1e".encode())
        for e in self.entries:
            p = e.placement
            h.update(
                (
                    f"{e.name}\x1f{e.destination.shape}\x1f{e.destination.dtype.label}"
                    f"\x1f{e.tied_alias}\x1f{p.device}\x1f{p.tp_rank}/{p.tp_size}"
                    f"\x1f{p.pp_stage}/{p.pp_size}\x1f{p.ep_rank}/{p.ep_size}"
                    f"\x1f{[t.value for t in e.transforms]}\x1e"
                ).encode()
            )
            for c in e.components:
                w = c.weight
                d = c.destination_slice
                h.update(
                    (
                        f"{w.name}\x1f{w.full_shape}\x1f{w.spec.shape}\x1f{w.shard.kind}"
                        f"\x1f{w.shard.dim}\x1f{w.shard.rank}/{w.shard.world_size}"
                        f"\x1f{c.source.source.tensor_key}\x1f{c.source.source_shape}"
                        f"\x1f{[(x.dim, x.start, x.stop) for x in c.source.source_slices]}"
                        f"\x1f{d.dim},{d.start},{d.stop}\x1c"
                    ).encode()
                )
            for r in e.reads:
                h.update(
                    (
                        f"{r.file_uri}\x1f{r.byte_offset}\x1f{r.nbytes}"
                        f"\x1f{r.run_bytes}\x1f{r.stride_bytes}\x1f{r.run_count}\x1d"
                    ).encode()
                )
        return h.hexdigest()

@dataclass(frozen=True, slots=True)
class ModelLoadReport:

    state: LoadState = LoadState.CREATED
    plan_digest: str = ""
    checkpoint_bytes: int = 0
    read_bytes: int = 0
    device_bytes: int = 0
    replicated_bytes: int = 0
    sharded_bytes: int = 0
    host_staging_peak_bytes: int = 0
    weight_count: int = 0
    tied_count: int = 0
    transform_count: int = 0
    component_count: int = 0
    bytes_per_rank: tuple[int, ...] = ()
    bytes_per_stage: tuple[int, ...] = ()
    unsupported: tuple[str, ...] = ()
    discover_ns: int = 0
    read_ns: int = 0
    transform_ns: int = 0
    h2d_ns: int = 0
    bind_ns: int = 0

    @property
    def succeeded(self) -> bool:
        return self.state is LoadState.WEIGHTS_READY

    @property
    def total_ns(self) -> int:
        return self.discover_ns + self.read_ns + self.transform_ns + self.h2d_ns + self.bind_ns

    @property
    def read_amplification(self) -> float:
        """``read_bytes / device_bytes``.  Above 1.0 means bytes were read and
        thrown away — the signature of a shard plan that reads whole tensors."""
        return self.read_bytes / self.device_bytes if self.device_bytes else 0.0
