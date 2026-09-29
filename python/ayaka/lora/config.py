"""Single serving configuration; legacy field names remain public aliases."""

from dataclasses import dataclass

from ayaka.lora.variant import ExecutionVariant
from ayaka.utils.validation import require_int

#: Selectable execution backends. ``triton`` stays opt-in until it is certified.
BACKENDS = ("torch_reference", "triton", "auto")
#: Floating LoRA compute dtypes; ``None`` derives the contract from the base layer.
COMPUTE_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    modules: tuple[str, ...]
    max_rank: int = 16
    max_adapters: int = 4
    memory_bytes: int = 64 << 20
    max_host_adapters: int = 32
    host_memory_bytes: int = 256 << 20
    max_loras_per_batch: int | None = None
    #: Actual checkpoint identity, independent of the API's served model alias.
    base_model_name_or_path: str | None = None
    #: Backend policy: explicit reference, explicit triton, or certified fallback.
    backend: str = "torch_reference"
    #: LoRA compute dtype; None follows the base activation contract.
    compute_dtype: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.compute_dtype is not None and self.compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"compute_dtype must be None or one of {COMPUTE_DTYPES}")
        ExecutionVariant(
            self.modules, self.max_rank, self.max_adapters, self.compute_dtype or "float32"
        )
        require_int(self.memory_bytes, "LoRA memory_bytes", minimum=1)
        require_int(self.host_memory_bytes, "LoRA host_memory_bytes", minimum=1)
        require_int(self.max_host_adapters, "max_host_adapters", minimum=1)
        if self.base_model_name_or_path is not None and (
            not isinstance(self.base_model_name_or_path, str)
            or not self.base_model_name_or_path.strip()
        ):
            raise ValueError("base_model_name_or_path must be nonempty text or None")
        limit = self.max_adapters if self.max_loras_per_batch is None else self.max_loras_per_batch
        require_int(limit, "max_loras_per_batch", minimum=1)
        if limit > self.max_adapters:
            raise ValueError("max_loras_per_batch exceeds device adapter capacity")
        object.__setattr__(self, "max_loras_per_batch", limit)

    @property
    def max_device_adapters(self) -> int:
        return self.max_adapters

    @property
    def max_lora_rank(self) -> int:
        return self.max_rank


__all__ = ["LoRAConfig"]
