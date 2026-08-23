# T05 Agent Prompt: Algorithm Security And Recovery

Read the constitution, task breakdown, global acceptance, and T01 handoff when available. If T01 is still running, implement only backward-compatible hooks and record requested contract integration.

## Objective

Harden user-controlled file loading, add explicit device selection, cooperative cancellation/resume inputs and machine-readable training/generation progress while preserving CLI flags/default output behavior.

## Allowed Write Boundary

- `spimaging/cli.py`
- `spimaging/generation/**`
- `spimaging/supervised_training/**`
- `spimaging/self_supervised_training/**`
- `spimaging/testing/**`
- `spimaging/training_common/**`
- `tests/test_security_resume.py`, `tests/test_algorithm_events.py`, and precise updates to existing algorithm tests
- T05 handoff and smoke files

## Forbidden Changes

- Do not edit appcore, worker, desktop, packaging, licenses, root manifests, or existing Tkinter GUI.

## Required Deliverables

- `allow_pickle=False`, archive validation and `weights_only=True` checkpoint loading.
- Auto/CUDA/CPU device selection with GPU index and explainable fallback.
- Cancellation hooks, resume metadata/state validation, training history JSONL/CSV and structured callbacks usable by T01.
- Generation partial manifest/resume capability without publishing incomplete data as success.

## Smoke Test

```powershell
conda run -n spimaging python -m pytest -q tests/test_security_resume.py tests/test_algorithm_events.py
```
