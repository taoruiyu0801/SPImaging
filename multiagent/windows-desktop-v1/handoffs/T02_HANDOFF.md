# T02 Handoff: PySide6 Desktop Workbench

## Status
- State: Complete
- Agent session: `/root/t02_desktop`
- Last updated: 2026-08-23

## What Changed
- Added a Chinese-first six-page PySide6 workbench: Home, Experiment, Run,
  Results, History, and Settings.
- Added registry-driven simulation/reconstruction parameter forms with separate
  selectors, supervised/self-supervised mode switching, quick/standard/custom
  presets, conditional visibility, collapsible advanced parameters, validation,
  and a 1--12 sample gallery selector.
- Added a `QProcess` worker controller that writes schema-v1 `run.json`, invokes
  `python.exe -m spimaging.worker`, strictly parses `WorkerEvent` JSONL, owns a
  Windows Job Object, requests cooperative cancellation through both stdin and
  `cancel.request`, and hard-stops after the required 10-second grace period.
  `pythonw.exe` is normalized to sibling `python.exe`; Job Object failures
  degrade with a visible warning.
- Added progress projection for stages/batches/epochs/samples and live loss/MAE
  curves, bounded logs, terminal-state handling, and safe close-while-running.
- Added public-demo manifest discovery, safe sample inspection through T05's
  pickle-free NPZ loader, a navigable sample list, RGB/count/depth/histogram and
  simulation-layer views, plus labeled/unlabeled result gallery rendering.
- Added validated result/artifact loading, supervised-metric hiding for
  unlabeled data, config reuse, compatible checkpoint resume preparation,
  result export, history recovery/rebuild, settings persistence, CUDA/CPU and
  available-memory preflight, runtime repair affordance, and user-initiated
  redacted diagnostic export.
- Added the stable `spimaging.desktop.__main__:main` and
  `python -m spimaging.desktop` entry points with explicit optional-dependency
  errors, `--version`, and an offscreen `--smoke-test` health check. Visible
  text routes through Qt translation helpers; Chinese is the beta source locale.

## Files Touched
- `spimaging/desktop/__init__.py`
- `spimaging/desktop/__main__.py`
- `spimaging/desktop/application.py`
- `spimaging/desktop/controller.py`
- `spimaging/desktop/dependency.py`
- `spimaging/desktop/i18n.py`
- `spimaging/desktop/models.py`
- `spimaging/desktop/pages.py`
- `spimaging/desktop/style.py`
- `spimaging/desktop/widgets.py`
- `spimaging/desktop/window.py`
- `tests/test_desktop.py`
- `tests/test_desktop_models.py`
- This handoff and `smoke-tests/T02_SMOKE_TEST.md`

## Smoke Test Result
- Command: `$env:QT_QPA_PLATFORM='offscreen'; conda run -n spimaging python -m pytest -q tests/test_desktop.py tests/test_desktop_models.py`
- Result: PASS, final run `20 passed in 0.75s`.
- Source entry smoke: `$env:QT_QPA_PLATFORM='offscreen'; conda run -n spimaging python -m spimaging.desktop --smoke-test`
- Result: PASS, exit code `0`.
- Version entry smoke: `conda run -n spimaging python -m spimaging.desktop --version`
- Result: PASS, `SPImaging 0.2.0-beta.1`.

## Decisions Made
- Desktop business state is Qt-free in `models.py`; optional PySide6 is checked
  only when a GUI/controller module is used. The package itself stays importable
  and calling the UI yields a clear dependency error when PySide6 is absent.
- Prediction exposes no reconstruction selector: the worker derives its method
  from checkpoint metadata. Only Simple3D is marked bundled; other algorithms
  explicitly require training or an imported compatible checkpoint.
- Gallery arrays are loaded lazily through T05 safe archive inspection and
  `allow_pickle=False` loading. Artifact paths use T01 safe relative paths.
- Resume creates a new run identity/directory, retains the checkpoint, and
  increases target epochs by one before returning to the editable Experiment
  page. Fingerprint compatibility enforcement remains in T05.
- Charts and array rendering use QPainter/QImage, adding no plotting dependency.

## Open Issues For Downstream Context
- T_FINAL owns adding PySide6 and the desktop entry point to `pyproject.toml`
  and packaged manifests. Recommended callable:
  `spimaging.desktop.__main__:main`.
- Runtime repair is an affordance in source mode. T_FINAL/T03 should wire the
  installed button to the launcher's repair action.
- The translation loader searches `spimaging/desktop/translations`, but this
  Chinese-first beta intentionally ships no English `.qm`.
- The GUI consumes T05's `spimaging.training_common.security` and `device`
  module names; T_FINAL should retain them or make a narrow import adaptation.

## Boundary Notes
- Stayed within write boundary: Yes.
- No dependency manifest, appcore/worker/algorithm file, Tkinter GUI, public
  asset, packaging file, user dirty file, staging area, or Git commit was
  changed by T02.

## Suggested Next Checks
- T_FINAL should add/install the PySide6 dependency and `spimaging-desktop`
  windowed entry, then run `python -m spimaging.desktop --smoke-test` under both
  source Python and the packed private `pythonw.exe` runtime.
- Run one full synthetic public-demo prediction/evaluation workflow and open its
  gallery from the installed build.
