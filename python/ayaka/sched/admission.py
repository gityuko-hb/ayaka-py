"""Composable admission prechecks.

These helpers are deliberately advisory.  They reduce failed prepare attempts;
physical reservation still happens only in ``StepRuntime.prepare``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ayaka.request.lifecycle import RequestLifecycle
from ayaka.sched.interfaces import AdmissionAdvisor, AdmissionCallback

__all__ = ["AlwaysAdmit", "CallbackAdmissionAdvisor", "CompositeAdmissionAdvisor"]


@dataclass(frozen=True, slots=True)
class AlwaysAdmit:
    """No-op advisor used when physical prepare is the only admission oracle."""

    def can_admit(
        self,
        lifecycle: RequestLifecycle,
        *,
        full_prompt_remaining: int,
        scheduled_prompt_tokens: int,
    ) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class CallbackAdmissionAdvisor:
    """Adapter for a concrete paged-KV capacity callback."""

    callback: AdmissionCallback

    def can_admit(
        self,
        lifecycle: RequestLifecycle,
        *,
        full_prompt_remaining: int,
        scheduled_prompt_tokens: int,
    ) -> bool:
        return bool(self.callback(lifecycle, full_prompt_remaining, scheduled_prompt_tokens))


class CompositeAdmissionAdvisor:
    """Logical AND over multiple independent advisory capacity gates."""

    def __init__(self, advisors: Iterable[AdmissionAdvisor]) -> None:
        self._advisors = tuple(advisors)

    def can_admit(
        self,
        lifecycle: RequestLifecycle,
        *,
        full_prompt_remaining: int,
        scheduled_prompt_tokens: int,
    ) -> bool:
        return all(
            advisor.can_admit(
                lifecycle,
                full_prompt_remaining=full_prompt_remaining,
                scheduled_prompt_tokens=scheduled_prompt_tokens,
            )
            for advisor in self._advisors
        )
