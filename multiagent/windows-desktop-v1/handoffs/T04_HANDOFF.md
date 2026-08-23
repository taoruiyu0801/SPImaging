# T04 Handoff: Synthetic Public Demo And Compliance

## Status
- State: Complete

## What Changed
- Added a no-input, deterministic analytic generator for four `64 x 64 x 1024`
  SPAD NPZ samples. All RGB/depth/albedo/count content comes from source-code
  geometry and fixed PCG64 seeds; sample archives use fixed ZIP metadata and
  pickle-free dtypes.
- Generated four CC0 samples, `index.csv`, and a versioned portable manifest
  containing source provenance, parameters, array schemas, file sizes, SHA-256
  values, generator hash, and checkpoint training-input hashes.
- Generated an 11,029-byte Simple3D checkpoint from only those four samples.
  The deterministic CPU recipe uses 2 base channels, temporal downsample 64,
  spatial stride 2, 8 epochs, and 32 fixed-order optimizer steps. It loads with
  `weights_only=True` and reproduces byte-for-byte in the locked T04 runtime.
- Added Apache-2.0 project licensing, a NOTICE, CC0 dedication/full legal text,
  third-party inventory, and a source-level SBOM with final-release refresh
  requirements.
- Added deterministic reproduction, safe-schema, provenance, contamination,
  checkpoint compatibility, and license tests.

## Files Touched
- `scripts/generate_synthetic_demo.py`
- `public_demo/**`
- `LICENSE`
- `NOTICE`
- `THIRD_PARTY_LICENSES.md`
- `SBOM.md`
- `tests/test_public_demo.py`
- `multiagent/windows-desktop-v1/handoffs/T04_HANDOFF.md`
- `multiagent/windows-desktop-v1/smoke-tests/T04_SMOKE_TEST.md`

## Smoke Test Result
- Result: PASS
- Command: `conda run -n spimaging python -m pytest -q tests/test_public_demo.py`
- Output: `8 passed in 6.16s`

## Decisions Made
- Counts are stored as `uint16`, which is accepted by the existing loaders and
  reduces the four public samples to about 465 KiB while retaining exact photon
  counts. Training/inference converts them to float32 as before.
- Each sample uses the existing public schema (`counts`, `depth_m`, `rgb`,
  `albedo`, `intensity`, `xhat`, `x`, physical scalars) plus synthetic
  provenance/license scalars. `surface_model` remains `single` for CLI/UI
  compatibility; `source_mode` is `synthetic`.
- The committed checkpoint is deliberately tiny and demonstrational. Its
  manifest makes no real-data accuracy claim and records every training input.
- `manifest.json` paths are relative to `public_demo`; the verifier rejects
  path escape, source-hash mismatch, object arrays, schema/dtype mismatch, and
  asset/checkpoint hash mismatch.
- Exact sample bytes are stable by construction. Exact checkpoint bytes are
  promised only for the locked runtime recorded by the release build because
  numerical kernels can differ between PyTorch versions.

## Open Issues For Downstream Context
- T02 should discover the quick-demo dataset/checkpoint through
  `public_demo/manifest.json`, not through either legacy demo directory.
- T_FINAL/T03 must explicitly exclude the existing `example_data` and
  `demo_checkpoint` trees from public installers and include `public_demo`,
  `LICENSE`, `NOTICE`, and the third-party notices.
- `SBOM.md` is a source-level inventory. Before publishing, T_FINAL/T03 must
  add exact CPU/CUDA archive versions, hashes, transitive/native components,
  NVIDIA terms, and the license files from the actual locked packages.
- The synthetic checkpoint is for workflow visualization only; it is not a
  scientific benchmark or a general-purpose real-data model.

## Boundary Notes
- Stayed within write boundary: Yes.
- No file under `example_data`, `demo_checkpoint`, algorithm/GUI/package
  manifests, or the user's pre-existing dirty paths was copied, modified,
  staged, restored, or committed.

## Suggested Next Checks
- Run `python scripts/generate_synthetic_demo.py --verify-only` in the final
  locked CPU runtime and each release builder.
- Run one public checkpoint prediction/evaluation flow from the desktop worker.
- Inspect the final installer file list and fail if either legacy demo path is
  present.
- Refresh the release SBOM and bundled licenses from the exact resolved
  CPU/CUDA environment manifests.
