from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from ayaka.distributed.parallel import ParallelContext, divide
from ayaka.utils.validation import require_int
from ayaka.weights.plan import TensorSlice
from ayaka.weights.spec import ShardKind, ShardSpec


class ProjectionKind(StrEnum):
    COLUMN = "column"
    REPLICATED = "replicated"
    HEAD = "head"
    KV_HEAD = "kv_head"


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    output_size: int
    kind: ProjectionKind = ProjectionKind.COLUMN
    aliases: tuple[str, ...] = ()
    num_heads: int | None = None
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("projection name must not be empty")
        require_int(self.output_size, "output_size", minimum=1)
        object.__setattr__(self, "kind", ProjectionKind(self.kind))
        if self.kind in (ProjectionKind.HEAD, ProjectionKind.KV_HEAD):
            if self.num_heads is None or self.head_dim is None:
                raise ValueError("head projections require num_heads and head_dim")
            require_int(self.num_heads, "num_heads", minimum=1)
            require_int(self.head_dim, "head_dim", minimum=1)
            if self.output_size != self.num_heads * self.head_dim:
                raise ValueError("head projection output_size must equal num_heads * head_dim")

    @classmethod
    def heads(
        cls,
        name: str,
        *,
        num_heads: int,
        head_dim: int,
        aliases: tuple[str, ...] = (),
        replicate: bool = False,
    ) -> ProjectionSpec:
        return cls(
            name=name,
            output_size=num_heads * head_dim,
            kind=ProjectionKind.KV_HEAD if replicate else ProjectionKind.HEAD,
            aliases=aliases,
            num_heads=num_heads,
            head_dim=head_dim,
        )


@dataclass(frozen=True)
class LocalProjection:
    spec: ProjectionSpec
    global_offset: int
    local_offset: int
    local_size: int
    source_offset: int
    source_size: int
    source_head_start: int | None = None
    local_num_heads: int | None = None
    num_head_replicas: int = 1

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def source_slice(self) -> TensorSlice:
        """Logical checkpoint coordinates, before storage packing."""
        start = self.global_offset + self.source_offset
        return TensorSlice(0, start, start + self.source_size)

    @property
    def global_size(self) -> int:
        return self.spec.output_size


class ProjectionLayout:
    """Turns logical projection declarations into one rank-local packed layout."""

    def __init__(
        self,
        specs: list[ProjectionSpec] | tuple[ProjectionSpec, ...],
        parallel_context: ParallelContext,
    ) -> None:
        if not specs:
            raise ValueError("at least one projection is required")
        self.specs = tuple(specs)
        self.parallel_context = parallel_context
        self._parts: dict[str, LocalProjection] = {}
        self._canonical: list[LocalProjection] = []

        global_offset = 0
        local_offset = 0
        for spec in self.specs:
            part = self._build_part(spec, global_offset, local_offset)
            if spec.name in self._parts:
                raise ValueError(f"duplicate projection name {spec.name!r}")
            self._parts[spec.name] = part
            for alias in spec.aliases:
                if alias in self._parts:
                    raise ValueError(f"duplicate projection alias {alias!r}")
                self._parts[alias] = part
            self._canonical.append(part)
            global_offset += spec.output_size
            local_offset += part.local_size

        self.global_output_size = global_offset
        self.local_output_size = local_offset

    def _build_part(
        self,
        spec: ProjectionSpec,
        global_offset: int,
        local_offset: int,
    ) -> LocalProjection:
        rank = self.parallel_context.rank
        world = self.parallel_context.world_size
        if spec.kind is ProjectionKind.REPLICATED:
            return LocalProjection(
                spec=spec,
                global_offset=global_offset,
                local_offset=local_offset,
                local_size=spec.output_size,
                source_offset=0,
                source_size=spec.output_size,
                num_head_replicas=world,
            )

        if spec.kind is ProjectionKind.COLUMN:
            shard = ShardSpec(ShardKind.COLUMN, dim=0, rank=rank, world_size=world)
            local_size = shard.shard_shape((spec.output_size,))[0]
            return LocalProjection(
                spec=spec,
                global_offset=global_offset,
                local_offset=local_offset,
                local_size=local_size,
                source_offset=rank * local_size,
                source_size=local_size,
            )

        assert spec.num_heads is not None and spec.head_dim is not None
        if world <= spec.num_heads:
            local_heads = divide(spec.num_heads, world, name=f"{spec.name}.num_heads")
            head_start = rank * local_heads
            replicas = 1
        else:
            if spec.kind is ProjectionKind.HEAD:
                raise ValueError("query/output heads cannot be replicated across TP ranks")
            replicas = divide(world, spec.num_heads, name="world_size")
            local_heads = 1
            head_start = rank // replicas
        local_size = local_heads * spec.head_dim
        return LocalProjection(
            spec=spec,
            global_offset=global_offset,
            local_offset=local_offset,
            local_size=local_size,
            source_offset=head_start * spec.head_dim,
            source_size=local_size,
            source_head_start=head_start,
            local_num_heads=local_heads,
            num_head_replicas=replicas,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(part.name for part in self._canonical)

    @property
    def parts(self) -> tuple[LocalProjection, ...]:
        return tuple(self._canonical)

    def __getitem__(self, name: str | int) -> LocalProjection:
        if isinstance(name, int):
            require_int(name, "projection index")
            return self._canonical[name]
        try:
            return self._parts[name]
        except KeyError:
            raise KeyError(f"unknown projection {name!r}; expected one of {self.names}") from None

    def split_output(self, output: torch.Tensor) -> dict[str, torch.Tensor]:
        if output.shape[-1] != self.local_output_size:
            raise ValueError(
                f"packed output width must be {self.local_output_size}, got {output.shape[-1]}"
            )
        return {
            part.name: output.narrow(-1, part.local_offset, part.local_size)
            for part in self._canonical
        }

    def extra_repr(self) -> str:
        values = ", ".join(
            f"{part.name}:{part.local_size}/{part.global_size}" for part in self._canonical
        )
        return f"[{values}]"
