# T01 Agent Prompt: Core Contracts And Worker

Read `00_PROJECT_CONSTITUTION.md`, `01_TASK_BREAKDOWN.md`, and `03_GLOBAL_ACCEPTANCE.md` first. No upstream handoff is required.

## Objective

Implement schema-v1 algorithm/parameter specs, run config/events/results, run storage/history, diagnostics redaction, and a standalone worker dispatcher. Preserve all existing CLI behavior.

## Allowed Write Boundary

- `spimaging/appcore/**`
- `spimaging/worker.py`
- `tests/test_appcore.py`, `tests/test_worker.py`
- T01 handoff and smoke files

## Forbidden Changes

- Do not edit `pyproject.toml`, README, existing algorithm modules, GUI modules, packaging, public assets, or user dirty files.

## Required Deliverables

- Validated, JSON-roundtrippable `RunConfig v1`, `WorkerEvent v1`, `ResultManifest v1`.
- Complete simulation/reconstruction `AlgorithmSpec` registries with conditional parameters and quick/standard presets.
- Atomic run storage, JSONL events, rebuildable SQLite history, result/path validation and diagnostic redaction.
- Worker CLI with fake/no-op workflow for tests and adapters for the existing CLI modules.
- Updated T01 handoff and smoke result.

## Smoke Test

```powershell
conda run -n spimaging python -m pytest -q tests/test_appcore.py tests/test_worker.py
```
