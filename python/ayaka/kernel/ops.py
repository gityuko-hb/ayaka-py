from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
from collections.abc import Callable, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

from ayaka.utils.import_utils import CapabilityError

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = logging.getLogger(__name__)

__all__ = [
    "OpHandle",
    "custom_op",
    "force_reference",
    "registered_ops",
    "verify_against_reference",
    "wrap_extern_op",
]

F = TypeVar("F", bound=Callable[..., Any])

#: namespace -> torch.library.Library, created lazily.
_LIBRARIES: dict[str, Any] = {}

#: op name -> OpHandle, for introspection and testing.
_REGISTRY: dict[str, OpHandle] = {}

#: Names forced to their reference implementation, on top of the env gate.
_forced_reference: set[str] = set()


def _force_reference_env() -> bool:
    """Check if AYAKA_FORCE_REFERENCE_OPS environment variable is enabled."""
    return os.environ.get("AYAKA_FORCE_REFERENCE_OPS", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _get_library(namespace: str) -> Any:
    import torch.library

    lib = _LIBRARIES.get(namespace)
    if lib is None:
        # FRAGMENT so several Ayaka components can add to one namespace.
        # The op's lifetime is tied to this object, so it is kept alive
        # in the module-level dict deliberately -- letting it be
        # collected unregisters the op while callers still hold
        # references to it.
        lib = torch.library.Library(namespace, "FRAGMENT")
        _LIBRARIES[namespace] = lib
    return lib


class OpHandle:
    """A registered op, plus everything needed to reason about it.

    Calling the handle dispatches to the kernel, or to the reference when
    forced. Keeping both reachable from one object is what makes
    `verify_against_reference` a two-line test rather than a fixture.
    """

    # No __slots__ here on purpose: functools.update_wrapper below writes
    # __module__, __name__, __qualname__, __doc__ and merges __dict__,
    # none of which exist on a slotted instance. A slotted OpHandle raises
    # AttributeError in its own constructor, so every @custom_op in the
    # process fails at import. There are tens of ops, not millions, so the
    # per-instance dict costs nothing worth having.

    def __init__(
        self,
        *,
        name: str,
        namespace: str,
        kernel: Callable[..., Any],
        reference: Callable[..., Any] | None,
        mutates_args: list[str],
    ) -> None:
        self.name = name
        self.namespace = namespace
        self.kernel = kernel
        self.reference = reference
        self.mutates_args = mutates_args
        self._qualified = f"{namespace}::{name}"
        self._dispatch: Callable[..., Any] | None = None
        functools.update_wrapper(self, kernel)

    def _resolve(self) -> Callable[..., Any]:
        dispatch = self._dispatch
        if dispatch is None:
            import torch

            dispatch = cast(
                Callable[..., Any],
                getattr(getattr(torch.ops, self.namespace), self.name),
            )
            self._dispatch = dispatch

        return dispatch

    def use_reference(self) -> bool:
        return self.name in _forced_reference or _force_reference_env()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_reference():
            if self.reference is None:
                raise CapabilityError(
                    f"reference::{self.name}",
                    detail=(
                        "reference implementations are forced but this op "
                        "does not declare one"
                    ),
                    remedy=f"add reference= to the @custom_op on {self.name}",
                )
            return self.reference(*args, **kwargs)

        try:
            return self._resolve()(*args, **kwargs)
        except (NotImplementedError, RuntimeError) as exc:
            msg = str(exc)
            # Enforce Invariant I4: Raise CapabilityError instead of raw dispatcher errors
            if (
                "Could not run" in msg
                or "backend" in msg.lower()
                or "no kernel image" in msg.lower()
                or "not implemented" in msg.lower()
            ):
                raise CapabilityError(
                    self._qualified,
                    detail=f"kernel is unavailable on this device: {exc}",
                    remedy="run on a supported device or pass AYAKA_FORCE_REFERENCE_OPS=1",
                ) from exc
            raise

    def __repr__(self) -> str:
        mode = "reference" if self.use_reference() else "kernel"
        return f"<OpHandle {self._qualified} mode={mode}>"


def _build_fake_impl(
    fn: Callable[..., Any],
    out_shape: int | str | None,
    out_dtype: torch.dtype | None,
    op_name: str,
) -> Callable[..., Any]:
    """Synthesise a meta kernel from 'output looks like argument N'.

    Most kernels produce an output whose shape matches one of the inputs,
    so hand-writing a meta kernel per op is repetitive and therefore
    something people skip -- and a skipped meta kernel is a graph break
    they will not notice until they profile.

    `out_shape` is the position or name of the argument to mirror;
    ``None`` means the op is in-place and returns nothing.
    """
    if out_shape is None:
        return lambda *args, **kwargs: None

    signature = inspect.signature(fn)

    def fake_impl(*args: Any, **kwargs: Any) -> Any:
        import torch

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
            if isinstance(out_shape, int):
                # Safely extract positional argument by index regardless of how called
                ref = list(bound.arguments.values())[out_shape]
            else:
                ref = bound.arguments[out_shape]
        except (IndexError, KeyError):
            raise RuntimeError(
                f"custom op {op_name!r} declares out_shape={out_shape!r}, "
                f"which is not present in its signature {signature}"
            ) from None

        if not isinstance(ref, torch.Tensor):
            raise TypeError(
                f"custom op {op_name!r} out_shape mirrored parameter {out_shape!r} "
                f"which is not a torch.Tensor (got {type(ref).__name__})"
            )

        if out_dtype is not None:
            return torch.empty(ref.shape, dtype=out_dtype, device=ref.device)
        return torch.empty_like(ref)

    return fake_impl


def _reduce_signature(
    fn: Callable[..., Any], computed_args: dict[str, Callable[..., Any]]
) -> Callable[..., Any]:
    """Hide dynamically-computed arguments from the op schema.

    An argument whose value changes with batch shape -- a tuning hint
    derived from ``next_power_of_2(num_tokens)``, say -- becomes a Dynamo
    guard if it appears in the schema, and every new value triggers a
    recompile. Computing it *inside* the op body makes it invisible to
    the tracer: one compiled graph instead of one per bucket.
    """
    original = fn
    original_sig = inspect.signature(fn)
    reduced_params = [
        param
        for name, param in original_sig.parameters.items()
        if name not in computed_args
    ]
    reduced_sig = original_sig.replace(parameters=reduced_params)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = reduced_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for arg_name, compute in computed_args.items():
            bound.arguments[arg_name] = compute(**bound.arguments)
        # Order arguments strictly according to the original signature to support
        # C/C++ extensions and pybind11 kernels that only accept positional args.
        ordered = [bound.arguments[param] for param in original_sig.parameters]
        return original(*ordered)

    wrapper.__signature__ = reduced_sig  # type: ignore[attr-defined]
    wrapper.__annotations__ = {
        key: value
        for key, value in getattr(original, "__annotations__", {}).items()
        if key not in computed_args
    }
    return wrapper


def _register(
    *,
    op_name: str,
    namespace: str,
    op_func: Callable[..., Any],
    mutates_args: list[str],
    fake_impl: Callable[..., Any] | None,
    dispatch_key: str,
) -> None:
    """Define the op and bind its implementations. Idempotent."""
    import torch
    import torch.library

    # Several engines in one process (evaluation harnesses do this)
    # would otherwise collide on the second registration.
    namespace_ops = getattr(torch.ops, namespace, None)
    if namespace_ops is not None and hasattr(namespace_ops, op_name):
        return

    lib = _get_library(namespace)
    schema = torch.library.infer_schema(op_func, mutates_args=mutates_args)

    try:
        lib.define(op_name + schema)
        lib.impl(op_name, op_func, dispatch_key)
        if fake_impl is not None:
            lib._register_fake(op_name, fake_impl)
    except RuntimeError as exc:
        message = str(exc)
        if "Tried to register an operator" in message and "multiple times" in message:
            return
        raise


def custom_op(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    namespace: str = "ayaka",
    mutates_args: list[str] | None = None,
    out_shape: int | str | None = None,
    out_dtype: torch.dtype | None = None,
    fake_impl: Callable[..., Any] | None = None,
    reference: Callable[..., Any] | None = None,
    computed_args: dict[str, Callable[..., Any]] | None = None,
    dispatch_key: str = "CUDA",
) -> Any:
    """Register `fn` as a torch custom op.

    Args:
        fn: The kernel. Its annotations must be schema-inferable
            (``Tensor``, ``int``, ``float``, ``bool``,
            ``Optional[Tensor]``, ``List[int]``, ...).
        name: Op name. Defaults to the function name.
        namespace: torch.ops namespace. Defaults to ``ayaka``.
        mutates_args: Argument names written in place. Getting this wrong
            is not cosmetic -- an unlisted mutation lets the compiler
            reorder reads across the write, so a KV-cache write silently
            lands after the attention that should have seen it.
        out_shape: Position or name of the argument whose shape the
            output mirrors, used to synthesise a fake implementation.
            ``None`` means the op returns nothing.
        out_dtype: Override the fake output dtype, for ops whose output
            dtype differs from the mirrored input (fp8 in, bf16 out).
        fake_impl: A hand-written meta kernel, when `out_shape` cannot
            express the shape. Mutually exclusive with `out_shape`.
        reference: A plain-torch implementation with identical
            semantics. Strongly encouraged: it is what makes
            ``AYAKA_FORCE_REFERENCE_OPS`` and
            `verify_against_reference` work.
        computed_args: ``{argument_name: fn(**other_args) -> value}`` for
            arguments that should be computed inside the op body rather
            than appearing in the schema.
        dispatch_key: Backend to bind to. ``CUDA`` unless you are
            registering a CPU or XPU variant.

    Returns:
        An `OpHandle`, callable like the original function.
    """
    if out_shape is not None and fake_impl is not None:
        raise ValueError("pass either out_shape or fake_impl, not both")

    def decorator(op_func: Callable[..., Any]) -> OpHandle:
        resolved_name = name or op_func.__name__

        registered_func: Callable[..., Any] = op_func
        if computed_args:
            registered_func = _reduce_signature(op_func, computed_args)

        meta = fake_impl or _build_fake_impl(
            registered_func, out_shape, out_dtype, resolved_name
        )

        _register(
            op_name=resolved_name,
            namespace=namespace,
            op_func=registered_func,
            mutates_args=mutates_args or [],
            fake_impl=meta,
            dispatch_key=dispatch_key,
        )

        handle = OpHandle(
            name=resolved_name,
            namespace=namespace,
            kernel=registered_func,
            reference=reference,
            mutates_args=mutates_args or [],
        )
        _REGISTRY[resolved_name] = handle
        return handle

    if fn is not None:
        return decorator(fn)
    return decorator


def wrap_extern_op(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    namespace: str = "ayaka",
    mutates_args: list[str] | None = None,
    out_shape: int | str | None = None,
    out_dtype: torch.dtype | None = None,
    fake_impl: Callable[..., Any] | None = None,
    reference: Callable[..., Any] | None = None,
    computed_args: dict[str, Callable[..., Any]] | None = None,
    dispatch_key: str = "CUDA",
) -> OpHandle:
    """Wrap a third-party kernel so Dynamo treats it as opaque.

    External kernel libraries JIT-compile, read files, and load modules
    at call time. Dynamo cannot trace any of that; without a wrapper it
    either fails to compile or bakes in a first-call artifact. Registered
    as a custom op, the call becomes a single opaque node with a known
    shape.
    """
    return custom_op(
        fn,
        name=name or fn.__name__,
        namespace=namespace,
        mutates_args=mutates_args,
        out_shape=out_shape,
        out_dtype=out_dtype,
        fake_impl=fake_impl,
        reference=reference,
        computed_args=computed_args,
        dispatch_key=dispatch_key,
    )


def registered_ops() -> dict[str, OpHandle]:
    """Every op registered so far. For startup logging and tests."""
    return dict(_REGISTRY)


@contextlib.contextmanager
def force_reference(*names: str) -> Iterator[None]:
    """Route the named ops (or all of them) to their reference.

    ``with force_reference():`` covers everything; naming ops narrows it,
    which is how you bisect a numerical regression down to one kernel
    without restarting the process.
    """
    global _forced_reference
    previous = set(_forced_reference)
    if names:
        _forced_reference |= set(names)
    else:
        _forced_reference |= set(_REGISTRY)
    try:
        yield
    finally:
        _forced_reference = previous


def verify_against_reference(
    op: str | OpHandle,
    *args: Any,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    **kwargs: Any,
) -> None:
    """Run kernel and reference on the same inputs and compare.

    Raises ``AssertionError`` with the worst absolute and relative
    deviation and the index where it occurred, because "tensors differ"
    alone tells you nothing about whether you are chasing an accumulation
    order or an indexing bug.

    Inputs are cloned before each run: an op that mutates its arguments
    would otherwise let the first call poison the second, and the
    comparison would pass while the kernel is wrong.
    """
    import torch

    handle = _REGISTRY[op] if isinstance(op, str) else op
    if handle.reference is None:
        raise ValueError(f"op {handle.name!r} declares no reference implementation")

    def clone(value: Any) -> Any:
        return value.clone() if isinstance(value, torch.Tensor) else value

    kernel_args = [clone(value) for value in args]
    kernel_kwargs = {key: clone(value) for key, value in kwargs.items()}
    ref_args = [clone(value) for value in args]
    ref_kwargs = {key: clone(value) for key, value in kwargs.items()}

    with force_reference(handle.name):
        expected = handle(*ref_args, **ref_kwargs)
    actual = handle(*kernel_args, **kernel_kwargs)

    # If the op mutates its arguments, verify mutated states
    if handle.mutates_args:
        sig = inspect.signature(handle.kernel)
        ref_bound = sig.bind(*ref_args, **ref_kwargs)
        ref_bound.apply_defaults()
        kernel_bound = sig.bind(*kernel_args, **kernel_kwargs)
        kernel_bound.apply_defaults()

        for mutated in handle.mutates_args:
            ref_val = ref_bound.arguments.get(mutated)
            kernel_val = kernel_bound.arguments.get(mutated)
            if isinstance(ref_val, torch.Tensor) and isinstance(kernel_val, torch.Tensor):
                _compare(ref_val, kernel_val, rtol, atol, f"{handle.name}.{mutated}")

    # Compare return values if non-None
    if expected is not None and actual is not None:
        _compare_nested(expected, actual, rtol, atol, handle.name)
    elif (expected is None) ^ (actual is None):
        raise AssertionError(
            f"{handle.name}: return mismatch, reference returned {expected!r} "
            f"vs kernel returned {actual!r}"
        )


def _compare_nested(
    expected: Any,
    actual: Any,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    """Compare tensors or nested sequences of tensors."""
    import torch

    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(expected) != len(actual):
            act_len = len(actual) if isinstance(actual, (tuple, list)) else "N/A"
            raise AssertionError(
                f"{label}: output sequence mismatch, reference {type(expected).__name__} "
                f"len={len(expected)} vs kernel {type(actual).__name__} len={act_len}"
            )
        for i, (exp_item, act_item) in enumerate(zip(expected, actual, strict=True)):
            _compare_nested(exp_item, act_item, rtol, atol, f"{label}[{i}]")
        return

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        _compare(expected, actual, rtol, atol, label)


def _compare(
    expected: torch.Tensor,
    actual: torch.Tensor,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    import torch

    if expected.shape != actual.shape:
        raise AssertionError(
            f"{label}: shape mismatch, reference {tuple(expected.shape)} "
            f"vs kernel {tuple(actual.shape)}"
        )

    expected_f = expected.float()
    actual_f = actual.float()

    nan_mismatch = torch.isnan(expected_f) ^ torch.isnan(actual_f)
    if nan_mismatch.any():
        raise AssertionError(
            f"{label}: NaN in one output but not the other at "
            f"{int(nan_mismatch.sum())} positions"
        )

    finite = torch.isfinite(expected_f) & torch.isfinite(actual_f)
    diff = (expected_f - actual_f).abs()
    diff = torch.where(finite, diff, torch.zeros_like(diff))
    tolerance = atol + rtol * expected_f.abs()
    failures = diff > tolerance

    if failures.any():
        worst = int(diff.argmax())
        raise AssertionError(
            f"{label}: {int(failures.sum())}/{diff.numel()} elements exceed "
            f"tolerance (rtol={rtol}, atol={atol}). "
            f"Worst at flat index {worst}: reference "
            f"{expected_f.flatten()[worst].item():.6g} vs kernel "
            f"{actual_f.flatten()[worst].item():.6g} "
            f"(abs {diff.flatten()[worst].item():.3g})"
        )
