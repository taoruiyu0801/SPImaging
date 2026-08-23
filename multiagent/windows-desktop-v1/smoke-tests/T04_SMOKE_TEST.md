# T04 Smoke Test: Synthetic Public Demo And Compliance

## Test Surface
Deterministic synthetic data, provenance hashes, checkpoint recipe and licenses.

## Procedure
1. Run `conda run -n spimaging python -m pytest -q tests/test_public_demo.py`.

## Expected Result
- Four valid samples and manifest hashes reproduce with no external data.

## Actual Result
- PASS on 2026-08-23 (Asia/Shanghai).
- Command: `conda run -n spimaging python -m pytest -q tests/test_public_demo.py`
- Output: `8 passed in 6.16s`.
- Confirmed four pickle-free `uint16` count cubes with shape
  `(1024, 64, 64)`, byte-for-byte sample regeneration, manifest/index/source
  hashes, safe `weights_only=True` checkpoint loading, strict Simple3D state
  compatibility, byte-for-byte checkpoint regeneration in the locked runtime,
  CC0 dedication, Apache-2.0 license, notices, and source SBOM.
