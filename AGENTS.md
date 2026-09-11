# Repository Guidelines

## Project Structure & Module Organization

Ayaka is a Python LLM inference engine. Source lives in `python/ayaka/`: `configs/` defines configuration, `model_loader/` and `weights/` handle checkpoints, `memory/` and `kvcache/` manage storage, and `sched/`, `executor/`, and `attention/` define execution components. GPU operations live in `kernel/triton/`; shared helpers belong in `utils/`. Tests live in `tests/`, and contribution templates live in `.github/`.

## Build, Test, and Development Commands

Use Python 3.13 and uv from the repository root. Prefix shell commands with `rtk`; use `rtk proxy` for passthrough, following `RTK.md`.

- `rtk proxy uv sync --group dev --extra cpu`: install the package, development tools, and CPU PyTorch. Substitute `--extra cuda` for GPU development; these extras are mutually exclusive.
- `rtk proxy uv build`: build the setuptools source distribution and wheel.
- `rtk proxy uv run --no-sync pytest -q`: run the local test suite.
- `rtk proxy uv run --no-sync ruff check .`: lint code and imports.
- `rtk proxy uv run --no-sync ruff format --check .`: check formatting; omit `--check` to format.
- `rtk proxy uv run --no-sync pyright .`: check types.

Run sync before the `--no-sync` commands. The declared `ayaka` CLI target is not implemented; use library modules and tests for local development.

## Coding Style & Naming Conventions

Use four-space indentation, UTF-8, LF endings, and Ruff's 100-character line limit. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Add type annotations and document public behavior, validation, and exceptions. Reuse existing validation and optional-import helpers in `utils/`.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions; Hypothesis is available for property tests. Focus a run by appending a path, such as `tests/test_configs.py`. Tests have a configured 30-second timeout; reserve `slow` for real OS-process cases. No numerical coverage threshold is configured. Cover changed behavior and failure paths, and distinguish CPU fallback checks from real GPU validation.

`tests/*` is currently ignored: explicitly verify intended test files are tracked before submitting.

## Commit & Pull Request Guidelines

Follow recent emoji-prefixed Conventional Commits, such as `✨ feat(kernel): add activation backend`. Keep commits focused. Complete `.github/PULL_REQUEST_TEMPLATE.md` with rationale, related issues, breaking changes, and reproducible validation commands and results. Update affected documentation; include screenshots when useful.
