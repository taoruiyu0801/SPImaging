# Task Breakdown

| ID | Agent Role | Objective | Write Boundary | Depends On | Feeds Into | Smoke Test |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | Core contract owner | Config/spec/event/result/history/worker foundation | `spimaging/appcore/**`, `spimaging/worker.py`, owned tests | None | T02, T05, T_FINAL | appcore + worker pytest |
| T02 | Desktop engineer | PySide6 workbench, dynamic forms, run/results/history UI | `spimaging/desktop/**`, desktop tests | T01, T04 | T06, T_FINAL | offscreen desktop pytest |
| T03 | Release engineer | Bootstrap, runtime provisioning, updater, Inno and CI | `launcher/**`, `packaging/**`, `.github/workflows/**`, launcher tests | None | T06, T_FINAL | launcher pytest + dry-run manifest |
| T04 | Public asset/compliance owner | Deterministic synthetic demo, checkpoint recipe, licenses and provenance | public asset/script/license paths | None | T02, T06, T_FINAL | deterministic asset pytest |
| T05 | Algorithm hardening engineer | Safe loading, device choice, cancellation/resume hooks and structured training artifacts | existing algorithm modules + owned tests | T01 contract guidance | T06, T_FINAL | security/resume pytest |
| T06 | QA/security reviewer | Cross-review integrated implementation and release risks | review report + own handoff/smoke only | T02, T03, T04, T05 | T_FINAL | evidence-backed report |
| T_FINAL | Integrator | Resolve glue, dependencies, docs, build/test/package and final report | shared integration files and narrow fixes | All | Release candidate | full validation |

## Task Notes

### T01

- Define stable schema v1 contracts, algorithm/parameter specs, run directory layout, JSONL event writer, history index, diagnostics redaction, and worker dispatch.
- The worker must have a testable no-Torch fake workflow and real command adapters without importing PySide6.

### T02

- Build Home, Experiment, Run, Results, History, and Settings pages.
- UI must remain importable with a clear dependency error when PySide6 is absent and testable with `QT_QPA_PLATFORM=offscreen`.

### T03

- Build a safe downloader/installer library before creating GUI polish.
- Release actions must not publish automatically without explicit workflow dispatch/tag conditions and required hashes.

### T04

- Do not copy NYUv2-derived arrays. Generate deterministic geometry/albedo/depth/SPAD content from code.
- A small Simple3D checkpoint may be generated locally; if time/resources prevent committing it, provide the deterministic recipe and a manifest with a clear release blocker.

### T05

- Preserve CLI flags and outputs.
- Remove unsafe pickle loading, use `weights_only=True`, add device selection and cooperative hooks, and persist machine-readable metrics/checkpoint state.

### T06

- Review contracts, unsafe file handling, subprocess lifetime, updater extraction, license assets, and UI workflow. Do not implement fixes.
