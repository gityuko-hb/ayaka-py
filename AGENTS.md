# Repository Guidelines

## Project Structure & Module Organization

Ayaka is a Python LLM inference engine. Source lives in `python/ayaka/`: `configs/` defines configuration and execution phase policy, `model_loader/` and `weights/` handle checkpoints and weight readiness, `memory/` and `kvcache/` manage storage, while `sched/`, `executor/`, `worker/`, and `runner/` define execution components (phase adapters, graph programs, PDMux lanes). `speculative/` owns scheduler-aware transactional speculative decoding and is deliberately its own plane because it cuts across `sched/`, `kvcache/`, `runner/`, and `sampling/`. `device/` owns hardware qualification and profiles, `device_comm/` owns collective policy dispatch and the verified-P2P probe, `sampling/` owns samplers, and `attention/` owns the attention backend contract and scoped forward context. GPU operations live in `kernel/triton/`; shared helpers belong in `utils/`. Tests live in `tests/`, and contribution templates live in `.github/`. The former `execution/` package has been dissolved into these owners; import dependencies flow `runtime/worker → runner → configs/device/model_loader/attention`.

Two kernel subtrees in `kernel/triton/` have no runtime consumer and should not be assumed wired: `moe/` (a complete CUDA→Triton port) and `lora/` (BGMV/SGMV shrink/expand, orphaned after the `lora/` package was removed). Conversely, `kernel/triton/comm/` and `p2p_signal.py` are live — they are imported by `device_comm/`.

## Build, Test, and Development Commands

Use Python 3.13 and uv from the repository root. Prefix shell commands with `rtk`; use `rtk proxy` for passthrough, following `RTK.md`.

- `rtk proxy uv sync --group dev --extra cpu`: install the package, development tools, and CPU PyTorch. Substitute `--extra cuda` for GPU development; these extras are mutually exclusive.
- `rtk proxy uv build`: build the setuptools source distribution and wheel.
- `rtk proxy uv run --no-sync pytest -q`: run the local test suite.
- `rtk proxy uv run --no-sync ruff check .`: lint code and imports.
- `rtk proxy uv run --no-sync ruff format --check .`: check formatting; omit `--check` to format.
- `rtk proxy uv run --no-sync pyright .`: check types.

Run sync before the `--no-sync` commands. The `ayaka` CLI is implemented: `python -m ayaka serve` (see `python/ayaka/commands/cli.py`). Use library modules and tests for local development.

## Coding Style & Naming Conventions

Use four-space indentation, UTF-8, LF endings, and Ruff's 100-character line limit. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Add type annotations and document public behavior, validation, and exceptions. Reuse existing validation and optional-import helpers in `utils/`.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions; Hypothesis is available for property tests. Focus a run by appending a path, such as `tests/test_configs.py`. Tests have a configured 30-second timeout; reserve `slow` for real OS-process cases. No numerical coverage threshold is configured. Cover changed behavior and failure paths, and distinguish CPU fallback checks from real GPU validation.

`tests/*` is currently ignored: explicitly verify intended test files are tracked before submitting.

## Commit & Pull Request Guidelines

Follow recent emoji-prefixed Conventional Commits, such as `✨ feat(kernel): add activation backend`. Keep commits focused. Complete `.github/PULL_REQUEST_TEMPLATE.md` with rationale, related issues, breaking changes, and reproducible validation commands and results. Update affected documentation; include screenshots when useful.
