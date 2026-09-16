"""Host memory: policy ceilings and exactly-tiered host sources.

- :mod:`~ayaka.memory.host.host_policy` — computes the pinned-memory ceiling
  from operator settings plus machine facts, and gates individual pin requests.
- :mod:`~ayaka.memory.host.host_source` — host allocation that reports the tier
  it actually produced and never substitutes one for another.

The workspace manager is not here: it is device/host-generic and lives at
:mod:`ayaka.memory.workspace`.
"""
