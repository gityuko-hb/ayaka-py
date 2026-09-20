"""Vocabulary-parallel embedding and LM head for tensor-parallel inference.

Layout follows the vLLM/SGLang reference (Apache-2.0): the base vocabulary is
padded, sharded across the TP group, and LoRA-style added embeddings
(``org_num_embeddings < num_embeddings``) are placed after the padded base.
Out-of-shard tokens embed as zeros and the partial results are reduced across
the group, so every rank returns the full embedding. The LM head loads the same
sharded layout and exposes :meth:`ParallelLMHead.logits`, which all-gathers the
local shards and reorders them into token-id order; ``forward`` deliberately
refuses to run, matching the reference engines where the sampler consumes head
weights directly.

The fused Triton op (``kernel/triton/embedding.py``) masks, gathers and zeroes
in one pass when the backend is Triton on CUDA with dense weights and
contiguous int32/int64 ids. Everything else uses the eager mask +
``F.embedding`` + all-reduce composition, which is the reference semantics and
the CPU path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self, cast

import torch
import torch.nn.functional as F
from torch import nn

from ayaka.distributed.parallel import ParallelContext, divide
from ayaka.layers._common import LayerBackend, check_dtype, load_kernel
from ayaka.layers._parallel_runtime import prepare_parallel_runtime
from ayaka.layers.base import BaseLayer
from ayaka.layers.linear.weight_loading import WeightLoadError, set_weight_attrs
from ayaka.layers.quantization.base import (
    BaseQuantization,
    QuantizationCapabilities,
    QuantizationTarget,
    QuantizeMethodBase,
)
from ayaka.utils.import_utils import CapabilityError, resolve_qualname
from ayaka.utils.torch_utils import compute_torch_dtypes
from ayaka.utils.validation import require_int

__all__ = [
    "DEFAULT_VOCAB_PADDING_SIZE",
    "EmbeddingMethodBase",
    "ParallelLMHead",
    "UnquantizedEmbeddingMethod",
    "VocabParallelEmbedding",
    "VocabParallelEmbeddingShardIndices",
    "masked_vocab_input",
    "pad_vocab_size",
    "resolve_embedding_quantization_config",
    "vocab_range_from_global_vocab_size",
    "vocab_range_from_per_partition_vocab_size",
]

DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Round ``vocab_size`` up to the next multiple of ``pad_to``."""
    require_int(vocab_size, "vocab_size", minimum=1)
    require_int(pad_to, "pad_to", minimum=1)
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> tuple[int, int]:
    """Half-open global range this rank owns for one per-partition block."""
    require_int(per_partition_vocab_size, "per_partition_vocab_size")
    require_int(rank, "rank")
    require_int(offset, "offset")
    index_f = rank * per_partition_vocab_size
    return index_f + offset, index_f + per_partition_vocab_size + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> tuple[int, int]:
    """Half-open global range this rank owns after an even shard of the vocabulary."""
    per_partition_vocab_size = divide(global_vocab_size, world_size, name="global_vocab_size")
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, offset=offset)


@dataclass(frozen=True, slots=True)
class VocabParallelEmbeddingShardIndices:
    """Global and padded index ranges owned by one shard of a vocab embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self) -> None:
        for name in (
            "padded_org_vocab_start_index",
            "padded_org_vocab_end_index",
            "padded_added_vocab_start_index",
            "padded_added_vocab_end_index",
            "org_vocab_start_index",
            "org_vocab_end_index",
            "added_vocab_start_index",
            "added_vocab_end_index",
        ):
            require_int(getattr(self, name), name)

        if self.padded_org_vocab_start_index > self.padded_org_vocab_end_index:
            raise ValueError("padded org vocab start must not exceed its end")
        if self.padded_added_vocab_start_index > self.padded_added_vocab_end_index:
            raise ValueError("padded added vocab start must not exceed its end")
        if self.org_vocab_start_index > self.org_vocab_end_index:
            raise ValueError("org vocab start must not exceed its end")
        if self.added_vocab_start_index > self.added_vocab_end_index:
            raise ValueError("added vocab start must not exceed its end")
        if self.org_vocab_start_index > self.padded_org_vocab_start_index:
            raise ValueError("org vocab start must not exceed its padded start")
        if self.added_vocab_start_index > self.padded_added_vocab_start_index:
            raise ValueError("added vocab start must not exceed its padded start")
        if self.org_vocab_end_index > self.padded_org_vocab_end_index:
            raise ValueError("org vocab range must lie inside its padded range")
        if self.added_vocab_end_index > self.padded_added_vocab_end_index:
            raise ValueError("added vocab range must lie inside its padded range")
        if self.num_org_elements > self.num_org_elements_padded:
            raise ValueError("org vocab padding must not be negative")
        if self.num_added_elements > self.num_added_elements_padded:
            raise ValueError("added vocab padding must not be negative")


def masked_vocab_input(
    input_: torch.Tensor,
    *,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global token ids onto this rank's padded shard and report misses.

    Returns ``(masked_input, invalid_mask)``: tokens owned by this rank map to
    their local storage row, every other token maps to index 0 and is flagged
    in ``invalid_mask`` so the caller zeroes its embedding row before the group
    reduction.
    """
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    return vocab_mask * (input_ - valid_offset), ~vocab_mask


class EmbeddingMethodBase(QuantizeMethodBase):
    """Embedding specialization of the shared storage/execution contract."""

    supported_targets: ClassVar[frozenset[QuantizationTarget]] = frozenset(
        {QuantizationTarget.EMBEDDING, QuantizationTarget.LM_HEAD}
    )

    def get_capabilities(self) -> QuantizationCapabilities:
        return QuantizationCapabilities(
            supported_devices=frozenset({"cpu", "cuda"}),
            supported_act_dtypes=compute_torch_dtypes(),
            supported_targets=self.supported_targets,
        )


class UnquantizedEmbeddingMethod(EmbeddingMethodBase):
    """Dense ``[partition, embedding_dim]`` storage with gather and local GEMM."""

    def create_weights(self, layer: nn.Module, *weight_args: Any, **attrs: Any) -> None:
        if weight_args:
            raise TypeError("embedding weight dimensions must be passed by name")
        weight = nn.Parameter(
            torch.empty(
                sum(attrs["output_partition_sizes"]),
                attrs["input_size_per_partition"],
                dtype=attrs["params_dtype"],
                device=attrs["device"],
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                key: value
                for key, value in attrs.items()
                if key
                not in {
                    "output_partition_sizes",
                    "input_size_per_partition",
                    "input_size",
                    "output_size",
                    "params_dtype",
                    "device",
                }
            },
        )

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_.long(), cast(torch.Tensor, layer.weight))

    def apply(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(hidden, weight if weight is not None else cast(torch.Tensor, layer.weight))


class _EmbeddingMethodConfig(BaseQuantization):
    def __init__(self, method: EmbeddingMethodBase) -> None:
        self.method = method

    @classmethod
    def get_name(cls) -> str:
        return "embedding_method"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        raise TypeError("construct the method explicitly")

    def get_capabilities(self) -> QuantizationCapabilities:
        return self.method.get_capabilities()

    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase:
        return self.method


def resolve_embedding_quantization_config(
    config: BaseQuantization | EmbeddingMethodBase | str | None,
) -> BaseQuantization | None:
    """Adapt explicit embedding methods to capability selection and finalization."""
    if config is None or isinstance(config, BaseQuantization):
        return config
    if isinstance(config, str):
        resolved = resolve_qualname(config)
        config = resolved() if isinstance(resolved, type) else resolved
    if not isinstance(config, EmbeddingMethodBase):
        raise TypeError("quant_config must be BaseQuantization or EmbeddingMethodBase")
    return _EmbeddingMethodConfig(config)


class VocabParallelEmbedding(BaseLayer):
    """Embedding parallelized over the vocabulary dimension.

    The base vocabulary is padded, sharded, and placed before any added
    (LoRA-style) embeddings and their padding, following the reference layout.
    Inference weights are frozen and loaded through ``weight_loader``; any
    token outside this rank's shard embeds as zero so the group all-reduce
    reconstructs the full embedding. Ids must be ``int32`` or ``int64`` and
    address the global, unpadded vocabulary.
    """

    _target: ClassVar[QuantizationTarget] = QuantizationTarget.EMBEDDING

    weight: nn.Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: BaseQuantization | EmbeddingMethodBase | str | None = None,
        prefix: str = "",
        activation_dtype: torch.dtype | None = None,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        backend: LayerBackend = "triton",
        **runtime: Any,
    ) -> None:
        require_int(num_embeddings, "num_embeddings", minimum=1)
        require_int(embedding_dim, "embedding_dim", minimum=1)
        require_int(padding_size, "padding_size", minimum=1)
        if org_num_embeddings is not None:
            require_int(org_num_embeddings, "org_num_embeddings", minimum=1)
            if org_num_embeddings > num_embeddings:
                raise ValueError("org_num_embeddings must not exceed num_embeddings")
        device, context, runtime = prepare_parallel_runtime(
            device,
            parallel_context,
            runtime,
            disabled=disable_tp,
        )
        super().__init__(
            prefix=prefix,
            quant_config=resolve_embedding_quantization_config(quant_config),
            **runtime,
        )
        self.num_embeddings = num_embeddings
        self.org_vocab_size = (
            org_num_embeddings if org_num_embeddings is not None else num_embeddings
        )
        self.num_added_embeddings = num_embeddings - self.org_vocab_size
        self.embedding_dim = embedding_dim
        self.disable_tp = disable_tp
        self.parallel_context = context
        self.device = device
        self.backend: LayerBackend = backend
        self.params_dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self.activation_dtype = (
            activation_dtype if activation_dtype is not None else self.params_dtype
        )
        if self.quant_config is None:
            check_dtype(self.params_dtype)
            if self.activation_dtype != self.params_dtype:
                raise ValueError(
                    "dense embedding requires matching parameter and activation dtypes"
                )

        self.padding_size = self._resolve_padding_size(padding_size)
        self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size, self.padding_size)
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + self.num_added_embeddings, self.padding_size
        )
        if self.org_vocab_size_padded > self.num_embeddings_padded:
            raise ValueError("padded org vocabulary must fit inside the padded vocabulary")
        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            self.tp_rank,
            self.tp_size,
        )
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size, name="num_embeddings_padded"
        )
        if self.shard_indices.num_elements_padded != self.num_embeddings_per_partition:
            raise ValueError("shard layout does not cover the padded vocabulary exactly")
        self.num_org_embeddings_per_partition = self.shard_indices.num_org_elements
        self.num_added_embeddings_per_partition = self.shard_indices.num_added_elements

        self._fused_op: Callable[..., torch.Tensor] | None = None
        self._create_embedding_weights()
        if not self.weight.is_meta:
            self._select_fused_op()

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Compute padded and unpadded index ranges for one shard."""
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_start, padded_org_end = vocab_range_from_global_vocab_size(
            org_vocab_size_padded, tp_rank, tp_size
        )
        padded_added_start, padded_added_end = vocab_range_from_global_vocab_size(
            num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
        )
        return VocabParallelEmbeddingShardIndices(
            padded_org_start,
            padded_org_end,
            padded_added_start,
            padded_added_end,
            min(padded_org_start, org_vocab_size),
            min(padded_org_end, org_vocab_size),
            min(padded_added_start, vocab_size),
            min(padded_added_end, vocab_size),
        )

    def _resolve_padding_size(self, padding_size: int) -> int:
        # Every padded size is a multiple of padding_size, so making it a
        # multiple of the group size keeps every shard width an integer.
        if self.tp_size > 1 and padding_size % self.tp_size:
            padding_size *= self.tp_size
        return padding_size

    def _create_embedding_weights(self) -> None:
        attrs: dict[str, Any] = {
            "output_partition_sizes": [self.num_embeddings_per_partition],
            "input_size_per_partition": self.embedding_dim,
            "input_size": self.embedding_dim,
            "output_size": self.num_embeddings_padded,
            "params_dtype": self.params_dtype,
            "device": self.device,
            "output_dim": 0,
        }
        if self.quant_config is not None:
            self.create_weights(
                activation_dtype=self.activation_dtype,
                target=self._target,
                **attrs,
            )
            return
        self.quant_method = UnquantizedEmbeddingMethod()
        attrs.setdefault("weight_loader", self.select_weight_loader(self.quant_method))
        self.quant_method.create_weights(self, **attrs)

    def select_weight_loader(self, method: QuantizeMethodBase) -> Callable[..., None]:
        """Embedding storage always shards the vocabulary rows in this layer."""
        del method
        return self.weight_loader

    def _select_fused_op(self) -> None:
        if self.backend != "triton" or self.tp_size == 1 or self.quant_config is not None:
            return
        self._fused_op = load_kernel(self.backend, "embedding", "vocab_parallel_embedding")

    def _fused_available(self, input_: torch.Tensor) -> bool:
        return (
            self._fused_op is not None
            and input_.is_cuda
            and input_.is_contiguous()
            and self.weight.is_cuda
            and self.weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Any = None,
    ) -> None:
        """Copy the shared checkpoint rows this rank owns; zero everything else.

        Shape checks run before any write, so a rejected checkpoint cannot
        leave the parameter partially updated.
        """
        if loaded_shard_id is not None:
            raise WeightLoadError("embedding weights do not accept logical shard IDs")
        output_dim = getattr(parameter, "output_dim", None)
        if output_dim is None:
            if loaded_weight.ndim == 0 and parameter.data.ndim == 1 and parameter.data.numel() == 1:
                loaded_weight = loaded_weight.reshape(1)
            if parameter.data.shape != loaded_weight.shape:
                raise WeightLoadError(
                    f"replicated parameter shape {tuple(parameter.data.shape)} does not match "
                    f"checkpoint {tuple(loaded_weight.shape)}"
                )
            with torch.no_grad():
                parameter.data.copy_(loaded_weight.to(parameter.data))
            return
        require_int(output_dim, "output_dim")
        data = parameter.data
        if parameter.is_meta or loaded_weight.is_meta:
            raise WeightLoadError("materialize meta storage before copying checkpoint weights")
        if output_dim >= data.ndim or loaded_weight.ndim != data.ndim:
            raise WeightLoadError("checkpoint rank and output dimension must match the parameter")

        width = loaded_weight.shape[output_dim]
        indices = self.shard_indices
        packed_dim = getattr(parameter, "packed_dim", None)
        packed_factor = 1
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = getattr(parameter, "packed_factor", 1)
            require_int(packed_factor, "packed_factor", minimum=1)
            if width != self.org_vocab_size // packed_factor:
                raise WeightLoadError(
                    f"packed checkpoint width {width} does not match org vocabulary "
                    f"{self.org_vocab_size} at factor {packed_factor}"
                )
            if width * packed_factor != self.org_vocab_size:
                raise WeightLoadError("packed checkpoint does not cover the org vocabulary")
        elif width not in (self.org_vocab_size, self.num_embeddings):
            raise WeightLoadError(
                f"checkpoint width {width} matches neither org vocabulary "
                f"{self.org_vocab_size} nor full vocabulary {self.num_embeddings}"
            )

        plan: list[tuple[int, int, int]] = []
        base_start = indices.org_vocab_start_index
        base_size = indices.org_vocab_end_index - base_start
        if packed_factor != 1:
            if base_start % packed_factor or base_size % packed_factor:
                raise WeightLoadError("embedding shard is not divisible by the packed factor")
            base_start //= packed_factor
            base_size //= packed_factor
        if base_size:
            plan.append((0, base_start, base_size))
        if width == self.num_embeddings and indices.num_added_elements:
            plan.append(
                (
                    indices.num_org_elements_padded,
                    indices.added_vocab_start_index,
                    indices.num_added_elements,
                )
            )

        for destination_start, source_start, size in plan:
            if destination_start + size > data.shape[output_dim]:
                raise WeightLoadError(
                    f"embedding shard [{destination_start}, {destination_start + size}) exceeds "
                    f"parameter width {data.shape[output_dim]}"
                )
            destination = data.narrow(output_dim, destination_start, size)
            source = loaded_weight.narrow(output_dim, source_start, size)
            if destination.shape != source.shape:
                raise WeightLoadError(
                    f"destination shape {tuple(destination.shape)} does not match "
                    f"source shape {tuple(source.shape)}"
                )
        with torch.no_grad():
            # Zero every row the plan does not write: base padding, the gap
            # before the added block and the trailing padding.
            cursor = 0
            for destination_start, _source_start, size in plan:
                if destination_start > cursor:
                    data.narrow(output_dim, cursor, destination_start - cursor).zero_()
                cursor = destination_start + size
            if cursor < data.shape[output_dim]:
                data.narrow(output_dim, cursor, data.shape[output_dim] - cursor).zero_()
            for destination_start, source_start, size in plan:
                destination = data.narrow(output_dim, destination_start, size)
                source = loaded_weight.narrow(output_dim, source_start, size)
                destination.copy_(source.to(device=data.device, dtype=data.dtype))

    def process_weights_after_loading(self) -> None:
        if self.quant_config is not None:
            super().process_weights_after_loading()
            return
        if any(tensor.is_meta for tensor in (*self.parameters(), *self.buffers())):
            raise RuntimeError("materialize meta weights before finalizing embedding")
        self.runtime_context(self.weight.device, self.weight.dtype, target=self._target)
        if self._fused_op is None:
            self._select_fused_op()

    def _embedding(self, input_: torch.Tensor) -> torch.Tensor:
        if self.quant_config is not None:
            return self.apply_quantization(input_)
        assert self.quant_method is not None
        return self.quant_method.embedding(self, input_)

    def _validate_input(self, input_: torch.Tensor) -> None:
        if not isinstance(input_, torch.Tensor):
            raise TypeError("token ids must be a torch.Tensor")
        if input_.dtype not in (torch.int32, torch.int64):
            raise TypeError("token ids must be int32 or int64")
        if input_.ndim < 1:
            raise ValueError("token ids must have at least one dimension")
        if input_.device != self.weight.device:
            raise ValueError(
                f"token ids on {input_.device} do not match weight {self.weight.device}"
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        self._validate_input(input_)
        if self.tp_size == 1:
            return self._embedding(input_)

        indices = self.shard_indices
        if self._fused_available(input_):
            flat = input_.reshape(-1)
            assert self._fused_op is not None
            output = self.run_kernel(
                self._fused_op,
                flat,
                self.weight,
                indices.org_vocab_start_index,
                indices.org_vocab_end_index,
                indices.num_org_vocab_padding,
                indices.added_vocab_start_index,
                indices.added_vocab_end_index,
            )
            if input_.ndim != 1:
                output = output.view(*input_.shape, self.embedding_dim)
        else:
            masked_input, invalid_mask = masked_vocab_input(
                input_,
                org_vocab_start_index=indices.org_vocab_start_index,
                org_vocab_end_index=indices.org_vocab_end_index,
                num_org_vocab_padding=indices.num_org_vocab_padding,
                added_vocab_start_index=indices.added_vocab_start_index,
                added_vocab_end_index=indices.added_vocab_end_index,
            )
            output = self._embedding(masked_input)
            output.masked_fill_(invalid_mask.unsqueeze(-1), 0)
        return self.parallel_context.all_reduce(output)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"num_embeddings_per_partition={self.num_embeddings_per_partition}, "
            f"embedding_dim={self.embedding_dim}, org_vocab_size={self.org_vocab_size}, "
            f"num_embeddings_padded={self.num_embeddings_padded}, tp_size={self.tp_size}"
        )


class ParallelLMHead(VocabParallelEmbedding):
    """Vocabulary-sharded LM head consumed through :meth:`logits`.

    The weight layout matches :class:`VocabParallelEmbedding`, so weights can
    be tied to an embedding table. ``forward`` raises: logits must go through
    :meth:`logits`, which gathers the local shards and reorders them so the
    final tensor is the full, token-id-ordered vocabulary every sampler
    expects.
    """

    _target: ClassVar[QuantizationTarget] = QuantizationTarget.LM_HEAD

    bias: nn.Parameter | None

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: BaseQuantization | EmbeddingMethodBase | str | None = None,
        prefix: str = "",
        activation_dtype: torch.dtype | None = None,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        backend: LayerBackend = "triton",
        **runtime: Any,
    ) -> None:
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            org_num_embeddings=org_num_embeddings,
            padding_size=padding_size,
            quant_config=quant_config,
            prefix=prefix,
            activation_dtype=activation_dtype,
            disable_tp=disable_tp,
            parallel_context=parallel_context,
            device=device,
            backend=backend,
            **runtime,
        )
        if bias:
            bias_parameter = nn.Parameter(
                torch.zeros(
                    self.num_embeddings_per_partition,
                    dtype=self.params_dtype,
                    device=self.device,
                ),
                requires_grad=False,
            )
            self.register_parameter("bias", bias_parameter)
            set_weight_attrs(
                bias_parameter,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)

    @property
    def requires_gather(self) -> bool:
        """Whether producing full logits needs a collective across the group."""
        return self.tp_size > 1

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        del input_
        raise RuntimeError("LMHead's weights should be used through logits()")

    def _local_logits(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.quant_config is not None:
            return self.apply_quantization(hidden)
        assert self.quant_method is not None
        return self.quant_method.apply(self, hidden, weight)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden rows and return the full, token-id-ordered vocabulary.

        The local shard GEMM is followed by an all-gather and a reorder that
        undoes the padded base/added/padding layout, so the result has exactly
        ``num_embeddings`` columns in global id order on every rank.
        """
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a torch.Tensor")
        if hidden.ndim < 2 or hidden.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"hidden must end in {self.embedding_dim} features with at least two dimensions"
            )
        if hidden.device != self.weight.device:
            raise ValueError(
                f"hidden on {hidden.device} does not match weight {self.weight.device}"
            )
        if self.tp_size == 1:
            # Project only the real vocabulary: padding must not change the
            # GEMM shape or its rounding relative to a dense head.
            logits = self._local_logits(hidden, self.weight[: self.num_embeddings])
            if self.bias is not None:
                logits = logits + self.bias[: self.num_embeddings]
            return logits[..., : self.num_embeddings]
        local = self._local_logits(hidden)
        if self.bias is not None:
            local = local + self.bias
        gathered = self.parallel_context.all_gather_last_dim(local)
        mapping = self.get_sharded_to_full_mapping()
        assert mapping is not None
        order = torch.tensor(mapping, dtype=torch.long, device=gathered.device)
        return gathered.index_select(-1, order)[..., : self.num_embeddings]

    def get_sharded_to_full_mapping(self) -> list[int] | None:
        """Gathered-shard index -> padded global index, or None without sharding."""
        if self.tp_size < 2:
            return None
        base_embeddings: list[int] = []
        added_embeddings: list[int] = []
        padding: list[int] = []
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            base_embeddings.extend(range(range_start, range_start + shard_indices.num_org_elements))
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            if (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                != range_end
            ):
                raise ValueError("shard layout does not cover the padded vocabulary exactly")
        ret = base_embeddings + added_embeddings + padding
        if len(ret) != self.num_embeddings_padded:
            raise ValueError("sharded-to-full mapping does not cover the padded vocabulary")
        return ret

    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> ParallelLMHead:
        """Share the embedding table; a head bias survives because it is not tied."""
        if not isinstance(embed_tokens, VocabParallelEmbedding):
            raise TypeError("embed_tokens must be a VocabParallelEmbedding")
        if self.num_embeddings != embed_tokens.num_embeddings:
            raise ValueError("tied embedding and head must share num_embeddings")
        if self.embedding_dim != embed_tokens.embedding_dim:
            raise ValueError("tied embedding and head must share embedding_dim")
        if self.bias is not None:
            if self.quant_config is not None or embed_tokens.quant_config is not None:
                raise CapabilityError(
                    "quantization.tie_weights",
                    detail="a biased head cannot tie packed storage",
                    remedy="disable the head bias or the packed head method",
                )
            self.weight = embed_tokens.weight
            return self
        assert self.quant_method is not None
        tied = self.quant_method.tie_weights(self, embed_tokens)
        if not isinstance(tied, ParallelLMHead):
            raise TypeError("tie_weights must return the tied LM head")
        return tied

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, bias={self.bias is not None}"
