from __future__ import annotations

import threading
from collections.abc import Callable


class CancelledError(RuntimeError):
    """Raised by :meth:`CancellationToken.raise_if_cancelled`."""

class CancellationToken:
    __slots__ = (
        "_callbacks",
        "_cancelled",
        "_children",
        "_lock",
        "_reason",
        "name"
    )

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._cancelled = False
        self._reason = ""
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[str], None]] = []
        self._children: list[CancellationToken] = []

    # read path: no lock
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError(f"{self.name or 'request'} cancelled: {self._reason}")

    # write path: locked, idempotent
    def cancel(self, reason: str = "client disconnect") -> bool:
        """Returns True if this call performed the cancellation, False if the
        token was already cancelled.  The return value is what lets a caller
        distinguish "I cancelled it" from "it was already gone" without a race."""
        with self._lock:
            if self._cancelled:
                return False
            # Order matters: the reason must be visible before the flag, or a
            # reader that sees is_cancelled can read an empty reason.
            self._reason = reason
            self._cancelled = True
            callbacks = tuple(self._callbacks)
            children = tuple(self._children)
            self._callbacks.clear()

        # Outside the lock: a callback is free to touch this token, and a child
        # cancel takes the child's lock, not ours.
        for child in children:
            child.cancel(reason)
        for cb in callbacks:
            try:
                cb(reason)
            except Exception:
                continue
        return True

    def on_cancel(self, callback: Callable[[str], None]) -> None:
        """Register a one-shot callback.  If the token is already cancelled the
        callback runs immediately on the calling thread — otherwise a race
        between registration and cancellation silently drops it."""
        with self._lock:
            if not self._cancelled:
                self._callbacks.append(callback)
                return
            reason = self._reason
        callback(reason)

    def __bool__(self) -> bool:
        """``if token:`` reads as "is it still live", not "is it cancelled" —
        which is the wrong way round often enough to be worth banning."""
        raise TypeError("CancellationToken has no truth value; use `token.is_cancelled` explicitly")

    def __repr__(self) -> str:
        state = f"cancelled({self._reason!r})" if self._cancelled else "live"
        return f"<CancellationToken {self.name or '-'} {state}>"
