# T02 Smoke Test: PySide6 Desktop Workbench

## Test Surface
Offscreen window construction, page navigation, dynamic forms, worker event handling and gallery models.

## Procedure
1. Set `QT_QPA_PLATFORM=offscreen`.
2. Run desktop-owned pytest files.

## Expected Result
- All tests pass without displaying a window or starting real training.

## Actual Result
- PASS on 2026-08-23.
- Desktop pytest completed with `20 passed in 0.75s`.
- `python -m spimaging.desktop --smoke-test` completed offscreen with exit code
  `0`.
- `python -m spimaging.desktop --version` printed
  `SPImaging 0.2.0-beta.1` and exited `0`.
