"""Adapter location resolution, deliberately independent of residency slots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ayaka.lora.variant import AdapterIdentity
from ayaka.model_loader.source import safe_join
from ayaka.utils.import_utils import import_module


@dataclass(frozen=True, slots=True)
class AdapterRef:
    identity: AdapterIdentity
    source: str
    hub_revision: str | None = None
    local_files_only: bool = True


@dataclass(frozen=True, slots=True)
class ResolvedAdapter:
    identity: AdapterIdentity
    root: Path
    config_path: Path
    weight_files: tuple[Path, ...]
    source_revision: str


class AdapterResolver:
    def resolve(self, ref: AdapterRef) -> ResolvedAdapter:
        if not isinstance(ref.identity, AdapterIdentity) or not ref.source.strip():
            raise ValueError("adapter reference requires immutable identity and source")
        root = Path(ref.source).expanduser()
        revision = ref.identity.revision
        if not root.is_dir():
            if root.is_absolute() or ref.source.startswith((".", "~")):
                raise FileNotFoundError(f"adapter directory does not exist: {root}")
            hub = import_module("huggingface_hub")
            root = Path(
                hub.snapshot_download(
                    repo_id=ref.source,
                    revision=ref.hub_revision or ref.identity.revision,
                    local_files_only=ref.local_files_only,
                    allow_patterns=[
                        "adapter_config.json",
                        "adapter_model*.safetensors",
                        "added_tokens.json",
                    ],
                )
            )
            revision = root.name
        root = root.resolve()
        config = safe_join(root, "adapter_config.json")
        files = tuple(
            safe_join(root, p.name) for p in sorted(root.glob("adapter_model*.safetensors"))
        )
        if not config.is_file() or not files:
            raise FileNotFoundError(
                "adapter requires adapter_config.json and adapter_model*.safetensors"
            )
        if safe_join(root, "added_tokens.json").is_file():
            raise ValueError("added-vocabulary adapters are unsupported in P0")
        return ResolvedAdapter(ref.identity, root, config, files, revision)
