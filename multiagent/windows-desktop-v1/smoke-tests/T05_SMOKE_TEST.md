# T05 Smoke Test: Algorithm Security And Recovery

## Test Surface
Safe NPZ/checkpoint loading, device fallback, structured progress, cancel and resume.

## Procedure
1. Run T05-owned tests.
2. Run affected legacy algorithm tests.

## Expected Result
- Security/recovery tests and affected legacy tests pass.

## Actual Result
- PASS on 2026-08-23.
- Required command: `15 passed in 2.34s`.
- Affected legacy training/prediction/evaluation/four-generation-model command: `14 passed in 24.09s`.
- T01 appcore/worker compatibility: `14 passed in 0.55s`.
- Owned-path compileall: passed.
- Additional CLI/error contract run: `16 passed, 85 subtests passed`; one expected integration-owned failure remains because the generated Day 13-14 parameter tables do not yet include the new device/resume flags.
