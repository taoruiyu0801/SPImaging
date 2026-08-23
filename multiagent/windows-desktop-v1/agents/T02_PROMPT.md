# T02 Agent Prompt: PySide6 Desktop Workbench

Read the constitution, task breakdown, global acceptance, `handoffs/T01_HANDOFF.md`, and `handoffs/T04_HANDOFF.md` before changes.

## Objective

Implement the Chinese-first PySide6 workbench consuming T01 contracts without duplicating algorithm logic.

## Allowed Write Boundary

- `spimaging/desktop/**`
- `tests/test_desktop.py`, `tests/test_desktop_models.py`
- T02 handoff and smoke files

## Forbidden Changes

- Do not edit appcore contracts, algorithms, packaging, project dependency files, existing Tkinter GUI, or public assets.

## Required Deliverables

- Home, Experiment, Run, Results, History, and Settings pages.
- Dynamic basic/advanced forms, presets, device selection, QProcess worker controller, progress/events, cancel/close handling.
- Result gallery supporting 1–12 samples, labeled/unlabeled states, metrics/artifacts, history reopen and configuration reuse.
- Translation-ready strings and offscreen tests.

## Smoke Test

```powershell
$env:QT_QPA_PLATFORM='offscreen'; conda run -n spimaging python -m pytest -q tests/test_desktop.py tests/test_desktop_models.py
```
