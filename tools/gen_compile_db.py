"""Generate a local clangd compilation database for Ayaka's native sources.

Run from an environment created by ``uv sync``. The database is for editor
navigation; the extension's JIT build remains the authority for compilation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CSRC_ROOT = REPO_ROOT / "python" / "ayaka" / "kernel" / "csrc"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "compile_commands.json"


def _cuda_root() -> Path | None:
    candidates = [
        Path(value)
        for name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT")
        if (value := os.environ.get(name))
    ]
    if nvcc := shutil.which("nvcc"):
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.extend((Path("/usr/local/cuda"), Path("/opt/cuda")))
    return next((root for root in candidates if (root / "include" / "cuda.h").is_file()), None)


def _include_paths() -> list[Path]:
    try:
        import pybind11
        from torch.utils.cpp_extension import include_paths
    except ImportError as exc:
        raise RuntimeError("install the project with uv sync before generating the database") from exc

    paths = [Path(path) for path in include_paths()]
    paths.extend((Path(pybind11.get_include()), CSRC_ROOT))
    if python_include := sysconfig.get_path("include"):
        paths.append(Path(python_include))
    return list(dict.fromkeys(path.resolve() for path in paths if path.is_dir()))


def build_database() -> list[dict[str, object]]:
    compiler = shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("clang++ is required to generate the clangd database")

    sources = sorted((*CSRC_ROOT.rglob("*.cpp"), *CSRC_ROOT.rglob("*.cu")))
    includes = _include_paths()
    cuda_root = _cuda_root()
    if any(source.suffix == ".cu" for source in sources) and cuda_root is None:
        print("CUDA toolkit not found; CUDA editor diagnostics may be incomplete", file=sys.stderr)

    entries: list[dict[str, object]] = []
    for source in sources:
        arguments = [compiler, "-std=c++17"]
        for include in (*includes, source.parent.resolve()):
            arguments.extend(("-I", str(include)))
        if source.parent.name == "ipc":
            arguments.append("-DTORCH_EXTENSION_NAME=ayaka_ipc_ext")
        elif source.parent.name == "gguf":
            arguments.append("-DTORCH_EXTENSION_NAME=ayaka_gguf_ext")
        if source.suffix == ".cu":
            arguments.extend(("-x", "cuda"))
            if cuda_root is not None:
                arguments.extend((f"--cuda-path={cuda_root}", "-I", str(cuda_root / "include")))
        arguments.extend(("-c", str(source)))
        entries.append(
            {"directory": str(REPO_ROOT), "file": str(source), "arguments": arguments}
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    options = parser.parse_args(argv)
    try:
        entries = build_database()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not entries:
        print(f"no native sources found under {CSRC_ROOT}", file=sys.stderr)
        return 1

    contents = json.dumps(entries, indent=2) + "\n"
    if options.stdout:
        print(contents, end="")
    else:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(contents, encoding="utf-8")
        print(f"wrote {len(entries)} entries to {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
