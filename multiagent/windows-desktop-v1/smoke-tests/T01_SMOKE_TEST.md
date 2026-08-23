# T01 Smoke Test: Core Contracts And Worker

## Test Surface
Schema/config roundtrip, algorithm registry, event writer, history and fake worker.

## Procedure
1. Run `conda run -n spimaging python -m pytest -q tests/test_appcore.py tests/test_worker.py`.

## Expected Result
- All tests pass without PySide6 or network.

## Actual Result
- Not run yet.
