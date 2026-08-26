from __future__ import annotations

import enum

class DType(enum.Enum):
    FP32 = ("fp32", 32, False)
    FP16 = ("fp16", 16, False)
    BF16 = ("bf16", 16, False)
    FP8_E4M3 = ("fp8_e4m3", 8, False)
    FP8_E5M2 = ("fp8_e5m2", 8, False)
    FP4_E2M1 = ("fp4_e2m1", 4, True)
    INT64 = ("int64", 64, False)
    INT32 = ("int32", 32, False)
    INT8 = ("int8", 8, False)
    UINT8 = ("uint8", 8, False)
    INT4 = ("int4", 4, True)
    BOOL = ("bool", 8, False)

    def __init__(self, label: str, bits: int, sub_byte: bool) -> None:
        self.label = label
        self.bits = bits
        self.sub_byte = sub_byte

    @classmethod
    def from_str(cls, name: str) -> DType:
        name = name.lower()
        for member in cls:
            if member.label == name or member.name.lower() == name:
                return member
        raise ValueError(f"Unknown dtype: {name}")

    @property
    def itemsize(self) -> float:
        """Bytes per element.  Fractional for sub-byte types — callers that need
        an exact byte count must go through :meth:`nbytes`."""
        return self.bits / 8.0

    def nbytes(self, numel: int) -> int:
        """Exact storage in bytes for ``numel`` elements of this dtype.

        Sub-byte types are packed 2-per-byte along the last dim; an odd count is
        rounded up to a whole byte (matches the int4 packing in
        ``csrc/quantization``).
        """
        total_bits = numel * self.bits
        return (total_bits + 7) // 8

    def __repr__(self) -> str:
        return f"DType.{self.name}"
    
class DeviceKind(enum.StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    DISK = "disk"
    REMOTE = "remote"
