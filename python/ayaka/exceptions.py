from __future__ import annotations


class RuntimeMemoryError(RuntimeError):
    """Base class for runtime-memory failures.

    Every control-plane failure in this package derives from this type, giving
    callers a single catch-all for memory-subsystem errors.
    """


class ModelLoadError(RuntimeError):
    """Base class for checkpoint discovery, planning and materialization
    failures.

    The rule this type exists to enforce: bootstrap never sees a raw
    ``KeyError``, ``OSError``, ``struct.error`` or torch exception.  A caller
    that catches this knows the model did not load; it does not have to know
    which library was in the call stack when it happened.

    Deliberately *not* a subclass of :class:`RuntimeMemoryError`.  Loading fails
    for reasons that have nothing to do with memory — a truncated header, a
    missing shard, a quantized checkpoint we cannot read — and folding them
    together would make ``except RuntimeMemoryError`` catch things no memory
    subsystem can do anything about.  The one genuine intersection is
    :class:`WeightMaterializationError`, which inherits from both.
    """


class CheckpointCorruptError(ModelLoadError):
    """The checkpoint contradicts itself.

    Raised by the safetensors header parser and by manifest construction:
    truncated file, header length past EOF, declared byte count that disagrees
    with shape × dtype, overlapping tensor ranges, duplicate tensor keys, or an
    integer overflow in an offset.  Distinct from
    :class:`WeightMismatchError` because *this* file is unreadable regardless of
    which model wants it.
    """


class CheckpointSecurityError(ModelLoadError):
    """Resolving the source would escape the snapshot.

    Path traversal in a ``weight_map`` entry, a symlink pointing outside the
    snapshot directory, or an ambiguous revision in reproducible mode.  Separate
    from ``CheckpointCorruptError`` so that "malformed" and "hostile" are
    distinguishable in a log, and so a caller can choose to retry the first and
    never the second.
    """


class UnsupportedWeightFormatError(ModelLoadError):
    """Unsupported Weight Format Error.

    AWQ/GPTQ packed layouts, FP8 blockwise scales, anything needing
    ``trust_remote_code``.  A structured refusal, not a silent dequantize to
    bf16: quantized weights loaded as if they were dense produce a model that
    runs and emits garbage, which costs more to debug than a failed boot.
    """


class WeightPlanningError(ModelLoadError):
    """The checkpoint is fine; the *requested parallel layout* is not.

    Deliberately not a :class:`WeightMismatchError`.  Nothing about the
    checkpoint is wrong when 14 query heads refuse to divide across 4 ranks —
    the same files load perfectly at tp=1, 2, 7 or 14.  Reporting it as a
    mismatch sends the reader to re-download a checkpoint that was never the
    problem; what they need to change is ``--tensor-parallel-size``.

    ``code`` is a stable machine-readable token (``TP_QUERY_HEADS_NOT_DIVISIBLE``
    and friends) so a launcher can branch on the reason without regex over a
    message, and so the message itself stays free to improve.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


class WeightMismatchError(ModelLoadError):
    """The checkpoint and the model disagree.

    Carries the three-set diff — missing, unexpected, and consumed — because
    naming only the first weight that disagreed sends the reader back for
    another run to learn whether it was one weight or four hundred.
    """


class WeightMaterializationError(ModelLoadError, RuntimeMemoryError):
    """A weight was planned and read but could not be placed.

    Inherits from both bases on purpose: an allocation failure here is a real
    memory-subsystem event that the ledger must reconcile, *and* a load failure
    that must roll the whole bundle back.  MRO is
    ``WeightMaterializationError → ModelLoadError → RuntimeMemoryError →
    RuntimeError``, so ``except RuntimeMemoryError`` and
    ``except ModelLoadError`` both catch it.
    """


class InvalidHandleError(RuntimeMemoryError):
    """A handle is stale, out of range, or belongs to no live object.

    Raised when a ``SequenceHandle``/``KVPageHandle`` generation no longer
    matches the arena/allocator generation, or when the index is out of range.
    Terminated transactions, reservations, and leases also surface here.
    """


class InvalidStateTransitionError(RuntimeMemoryError):
    """An operation is not valid in the object's current lifecycle state.

    For example, releasing a page that is not ``LIVE``, preparing an empty
    transaction, or completing a step that is not ``IN_FLIGHT``.
    """


class InvariantViolationError(RuntimeMemoryError):
    """An internal accounting or ownership invariant was violated.

    Raised by the debug ``assert_invariants`` paths and by any operation that
    detects refcount underflow, duplicated free entries, or an inconsistent
    page-state partition. Indicates a bug in the memory subsystem itself.
    """


class SequenceCapacityError(RuntimeMemoryError):
    """The sequence arena has no free slots.

    Raised when ``max_sequences`` concurrent sequences are already live and a
    new ``create_sequence`` call cannot be satisfied.
    """


class SequenceBusyError(RuntimeMemoryError):
    """A sequence already participates in a transaction or execution lease.

    A sequence may participate in at most one transaction or execution lease;
    cache/attach/release operations on a busy sequence fail with this error.
    """


class TransactionClosedError(RuntimeMemoryError):
    """A transaction is no longer open for reservation.

    Raised when reservations are attempted after the transaction was prepared,
    rolled back, or already closed.
    """


class StorageUnavailableError(RuntimeMemoryError):
    """The requested optional tensor-storage backend is unavailable.

    Raised when PyTorch (or a specific dtype) is missing while constructing
    ``TorchMHAKVStorage``/``TorchMLAKVStorage`` or running storage adapters.
    """
