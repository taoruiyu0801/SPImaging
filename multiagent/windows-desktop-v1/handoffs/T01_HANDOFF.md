# T01 Handoff: Core Contracts And Worker

## Status
- State: Complete
- Agent session: root/T01
- Last updated: 2026-08-23

## What Changed
- Added schema-v1 run config, algorithm/parameter registries, structured worker events, atomic run/result storage, rebuildable SQLite history, diagnostics redaction/export, cancellation primitives, Windows Job Object support, and the isolated worker CLI.
- Added a source-runnable `noop` workflow so the desktop/controller contract can be tested without Torch, GUI, or network.
- Added adapters for generate, inspect, supervised/self-supervised train, predict, evaluate, full pipeline and quick demo workflows.

## Files Touched
- `spimaging/appcore/**` - shared contracts and services.
- `spimaging/worker.py` - JSONL worker entry and existing CLI adapters.
- `tests/test_appcore.py`, `tests/test_worker.py` - contract/storage/history/worker tests.

## Smoke Test Result
- Command/check: `conda run -n spimaging python -m pytest -q tests/test_appcore.py tests/test_worker.py`
- Result: Passed
- Evidence: `14 passed in 0.70s` before the final Windows handle prototype/strict boolean refinements; integrator should rerun.

## Decisions Made
- Config, events and results use explicit schema version 1 and reject unknown contract fields.
- Simulation and reconstruction registries remain separate; model forms expose only architecture parameters that currently affect each implementation.
- Child algorithms may emit `SPIMAGING_EVENT <json>` only when `SPIMAGING_STRUCTURED_EVENTS=1`; the worker wraps them with run ID, sequence and timestamp.
- Worker cancellation is requested through stdin JSON or `SPIMAGING_CANCEL_FILE`; process trees receive a 10-second cooperative grace period.

## Open Issues For Downstream Context
- T05 owns final CLI names/semantics for resume and generation continuation; T_FINAL must align worker flags with the T05 handoff.
- PySide6 dependency and public console entry points remain T_FINAL-owned root-manifest changes.
- Source quick demo requires T04 `public_demo` assets; until then it intentionally reports a repair/select-data error.

## Boundary Notes
- Stayed within write boundary: Yes

## Suggested Next Checks
- T02 should consume `RunConfig`, algorithm registries, `WorkerEvent` and `ResultManifest` directly.
- T_FINAL should run a real full-pipeline worker after T04/T05 integration and validate Windows Job Object cancellation.
