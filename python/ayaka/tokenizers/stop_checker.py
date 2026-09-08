"""Stop-string matching for incremental detokenization.

:class:`StopChecker` scans a character stream for any of a set of stop strings
in ``O(n × |stops|)`` time using plain ``str.find``, which is fast enough for
the handful of stop strings that LLM serving typically uses.

Design goals:

* **Zero false positives** — never emit a character that might be the start of
  a stop string until enough lookahead proves it is not.
* **Low latency** — emit as much text as possible per tick; only the minimum
  look-back window (``max_stop_len − 1`` characters) is withheld.
* **Stateless flush** — :meth:`StopChecker.flush` drains the buffer
  unconditionally so callers do not need to distinguish between "done" and
  "EOS" signal paths.

Typical call sequence::

    checker = StopChecker(stop=("</s>", "<|eot_id|>"), include_stop_str_in_output=False)
    for delta in token_stream:
        emit, matched = checker.feed(delta)
        if emit:
            forward_to_client(emit)
        if matched:
            break
    tail = checker.flush()          # release any held look-back characters
    if tail:
        forward_to_client(tail)
"""

from __future__ import annotations


class StopChecker:
    """Stateful stop-string detector for streaming text output.

    The checker consumes text fragments produced by an incremental detokenizer
    and decides how much can be safely forwarded to the client.  A fragment is
    held back only if it could be the start of a stop string; once enough
    context accumulates to confirm that no stop string begins there, the held
    characters are released.

    The look-back window size is ``max(len(s) for s in stops) - 1``.  This is
    the tightest safe bound: any stop string that starts in the already-emitted
    text would have been caught on a previous call.

    Attributes:
        _stops: Non-empty stop strings in the order supplied.
        _keep: Number of characters to retain as a look-back window when no
            match has been found yet.
        _include: When ``True``, the matched stop string is forwarded to the
            client as part of the final :meth:`feed` result.
        _buf: Accumulated but not-yet-emitted characters.
        _done: ``True`` once a stop has been matched or :meth:`flush` has been
            called; subsequent :meth:`feed` calls are no-ops.
    """

    __slots__ = (
        "_stops",
        "_keep",
        "_buf",
        "_include",
        "_done",
    )

    def __init__(self, stop: tuple[str, ...], include_stop_str_in_output: bool) -> None:
        """Initialise the checker.

        Args:
            stop: Candidate stop strings.  Empty strings are silently dropped
                because they would match immediately on every input and stall
                the stream.
            include_stop_str_in_output: When ``True``, the matched stop string
                is forwarded to the client as part of the final delta; when
                ``False`` it is consumed silently.
        """
        self._stops = tuple(s for s in stop if s)
        self._keep = max((len(s) for s in self._stops), default=1) - 1
        self._include = include_stop_str_in_output
        self._buf = ""
        self._done = False

    @property
    def has_stops(self) -> bool:
        """``True`` when at least one non-empty stop string was provided."""
        return bool(self._stops)

    @property
    def pending(self) -> str:
        """Current look-back buffer — characters held, not yet emitted."""
        return self._buf

    def feed(self, delta: str) -> tuple[str, str | None]:
        """Feed a new text fragment and return what can be safely forwarded.

        The return value is a ``(emit, matched)`` pair:

        * ``emit`` — text that is safe to forward to the client now.  May be
          empty when a stop was already detected (``_done``) or when the
          look-back window is still filling up.
        * ``matched`` — the stop string that was detected, or ``None`` if
          generation should continue.

        Once ``matched`` is not ``None`` the checker is considered done and
        all subsequent calls return ``("", None)``.

        Args:
            delta: New text from the detokenizer.  An empty string is a no-op.

        Returns:
            ``(emit, matched_stop_string_or_None)``
        """
        if self._done:
            return "", None   # already stopped — swallow all further input
        if not self._stops:
            return delta, None  # no stops configured — pass through immediately
        if not delta:
            return "", None   # empty delta is a no-op

        # Append incoming text to the look-back buffer before scanning.
        # The buffer may already contain a partial stop-string tail from the
        # previous call, so we must search the combined content.
        self._buf += delta

        # --- Linear scan: find the earliest (leftmost) stop match ---
        # We iterate over all stop strings and keep the one with the smallest
        # starting index so behaviour is deterministic when multiple stops
        # could match at different offsets in the same buffer.
        best_idx = -1
        best_stop: str | None = None
        for s in self._stops:
            i = self._buf.find(s)
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx, best_stop = i, s

        if best_stop is not None:
            # A stop string was matched.
            # include=True  → cut just after the stop string (emit it to client).
            # include=False → cut just before it (swallow the stop string silently).
            cut = best_idx + (len(best_stop) if self._include else 0)
            out = self._buf[:cut]
            self._buf = ""    # discard everything after the stop
            self._done = True
            return out, best_stop

        # --- No stop found: emit what is safe, retain the look-back tail ---
        # We must withhold the last `_keep` characters because they could be
        # the prefix of a stop string whose suffix arrives in the next delta.
        # `_keep = max_stop_len − 1` is the tightest safe bound: a stop string
        # that started entirely in already-emitted text would have been caught.
        if self._keep <= 0:
            # No look-back needed (all stops have length 1) — flush entire buffer.
            out, self._buf = self._buf, ""
            return out, None
        if len(self._buf) <= self._keep:
            # Buffer too small to safely emit anything yet; stall until more arrives.
            return "", None
        # Emit everything except the look-back tail.
        out = self._buf[: -self._keep]
        self._buf = self._buf[-self._keep :]
        return out, None

    def flush(self) -> str:
        """Drain the look-back buffer and mark the checker as done.

        Called at end-of-sequence (EOS or max-token limit) to release any
        characters that were withheld pending a potential stop-string match.
        After ``flush``, :meth:`feed` will always return ``("", None)``.

        Returns:
            Text that was held in the look-back buffer, or an empty string
            when the checker was already done (stop matched or previously
            flushed).
        """
        if self._done:
            return ""
        out, self._buf = self._buf, ""
        self._done = True
        return out
