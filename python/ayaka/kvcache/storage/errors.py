from __future__ import annotations

from ayaka.exceptions import StorageUnavailableError


class KVStorageCompatibilityError(StorageUnavailableError):
    """A storage configuration cannot execute on the target device."""

    def __init__(self, result: object) -> None:
        self.result = result
        issues = getattr(result, "issues", ())
        detail = "; ".join(getattr(issue, "message", str(issue)) for issue in issues)
        super().__init__(f"KV storage is incompatible: {detail}")

class StorageClosedError(RuntimeError):
    """A storage object was used after :meth:`close`.

    Its own type because the alternative -- indexing an emptied tuple -- raises
    ``IndexError`` from a line that mentions neither the close nor the caller's
    mistake, and reads exactly like an out-of-range layer index.
    """

class KVQuantizationError(ValueError):
    """A quantized store was used without, or against, a valid scale.

    Covers three distinct mistakes that all corrupt data silently otherwise:
    writing before calibration, changing a scale after pages were written under
    the previous one, and supplying a non-finite or non-positive scale.
    """


class SlotAddressError(ValueError):
    """A slot or page address is out of range, duplicated, or malformed.

    Duplicates matter on the write path specifically: scattering to the same
    slot twice lets the second write silently overwrite the first, which is a
    correctness bug with no symptom.
    """
