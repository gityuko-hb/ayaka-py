from __future__ import annotations

import re
from decimal import Decimal
from typing import Final

__all__ = [
    "format_human_int",
    "parse_bool",
    "parse_human_int",
]

_TRUE: Final = frozenset({"1", "true", "yes", "y", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "n", "off"})

_SI: Final = {"k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}
_IEC: Final = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}

# Suffix alternatives are ordered longest-first so "Ki" wins over "K".
_SIZE_RE: Final = re.compile(r"(\d+(?:\.\d+)?)\s*(Ki|Mi|Gi|Ti|k|M|G|T)?")


def parse_bool(value: str, *, what: str = "value") -> bool:
    """Parse a string into a boolean.

    Raises `ValueError` with an actionable message on anything else.

    Examples:
        >>> parse_bool("true")
        True
        >>> parse_bool("no")
        False
    """
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(
        f"{what}: cannot parse {value!r} as a boolean. "
        f"Use one of {sorted(_TRUE)} or {sorted(_FALSE)}."
    )


def parse_human_int(value: str, *, what: str = "value") -> int:
    """Parse ``'80Gi'`` / ``'25.6k'`` / ``'4096'`` into an int.

    See the module docstring for the SI/IEC contract. Raises `ValueError`
    with an actionable message on anything else.

    Examples:
        >>> parse_human_int("1k")
        1000
        >>> parse_human_int("1Ki")
        1024
        >>> parse_human_int("25.6k")
        25600
        >>> parse_human_int("4096")
        4096
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{what}: empty string is not an integer")

    match = _SIZE_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{what}: cannot parse {value!r} as an integer. Use a plain "
            f"integer, an SI suffix (1k, 1M, 1G, 1T = powers of 1000), or "
            f"an IEC suffix (1Ki, 1Mi, 1Gi, 1Ti = powers of 1024). "
            f"Suffixes are case-sensitive."
        )

    number, suffix = match.groups()

    if suffix is None:
        if "." in number:
            raise ValueError(
                f"{what}: {value!r} is not an integer. Drop the decimal "
                f"point, or use an SI suffix such as {number}k."
            )
        return int(number)

    if suffix in _IEC:
        if "." in number:
            raise ValueError(
                f"{what}: decimals are not allowed with the IEC suffix "
                f"{suffix!r} because the result would be truncated "
                f"silently. Use an integer such as "
                f"{int(Decimal(number))}{suffix}, or the SI form "
                f"{number}{suffix[0]}."
            )
        return int(number) * _IEC[suffix]

    # Decimal arithmetic, not float: int(0.1 * 10**9) is 99999999 under
    # binary floating point.
    return int(Decimal(number) * _SI[suffix])


def format_human_int(value: int, *, binary: bool = True) -> str:
    """Render an int back into the shortest exact human form.

    Only emits a suffix when the value divides exactly, so the result
    always round-trips through `parse_human_int`. Used in log lines and
    error messages, never on a hot path.

    Examples:
        >>> format_human_int(1000)
        '1k'
        >>> format_human_int(1024)
        '1Ki'
        >>> format_human_int(25600)
        '25.6k'
    """
    table = _IEC if binary else _SI
    for suffix, mult in sorted(table.items(), key=lambda kv: -kv[1]):
        if mult <= abs(value) and value % mult == 0:
            return f"{value // mult}{suffix}"
    return str(value)
