# T05 Handoff: Algorithm Security And Recovery

## Status
- State: Complete
- Agent session: root/t05_hardening
- Last updated: 2026-08-23

## What Changed
- Added fail-closed NPZ inspection/loading: pickle is disabled, NPY headers and field names are inspected before allocation, object arrays/path traversal/duplicate fields are rejected, archive/member/compression limits are enforced, SPAD fields/shapes/finiteness are validated, and loading requires 2x currently available physical-memory headroom.
- Replaced inference checkpoint loading with `torch.load(..., weights_only=True)`, a 2 GiB file cap, and mapping/schema/model-parameter validation. Newly saved checkpoints are atomically replaced and remain weights-only loadable.
- Added explicit `auto`/`cuda`/`cpu` device selection with GPU index, CUDA allocation self-test, CPU fallback reason, and worker environment defaults from `SPIMAGING_DEVICE`/`SPIMAGING_GPU_INDEX`. Existing `get_torch_device(prefer_gpu=...)` callers remain compatible.
- Added optional structured callbacks and the T01 child protocol: only when `SPIMAGING_STRUCTURED_EVENTS=1`, algorithms print exact-prefix `SPIMAGING_EVENT <json>`. Training emits device/warning/stage/batch/epoch/artifact/completed/cancelled information; generation emits stage/sample/device/completed/cancelled information.
- Added cooperative cancellation through an injected callback or `SPIMAGING_CANCEL_FILE`. Supervised and self-supervised cancellation saves model, optimizer, content-based dataset fingerprint, architecture/preprocessing signature, RNG state, epoch/batch position, and global step to `cancelled.pt` before exiting with status 130.
- Added `--resume_checkpoint` for both training CLIs. Resume uses safe loading, requires identical dataset content and algorithm/network/preprocessing/optimizer signature, allows the epoch target to stay equal or increase but not decrease, restores model/optimizer/RNG/global-step/batch state, and uses deterministic per-epoch sample ordering.
- Added atomic per-epoch `last.pt`/`best.pt`, per-batch events, and append-only `training_history.jsonl` plus `training_history.csv`.
- Added generation `--resume` and persistent sibling partial directories named `.<output-name>.spimaging-partial`. Every completed sample is SHA-256 recorded with its index row and NumPy RNG state; resume validates configuration, source content, sample hashes, and never publishes the incomplete manifest as success. `--overwrite` replaces only recognized owned partial files and refuses unknown entries.
- Prediction now accepts safe unlabeled NPZ input; target/error fields remain conditional. Predict/evaluate/generate/train CLIs accept shared device flags without changing their old defaults.

## Files Touched
- `spimaging/cli.py`
- `spimaging/generation/models.py`, `spimaging/generation/pipeline.py`, `spimaging/generation/recovery.py`
- `spimaging/supervised_training/train.py`, `spimaging/self_supervised_training/train.py`
- `spimaging/testing/browse.py`, `spimaging/testing/evaluate.py`, `spimaging/testing/predict.py`, `spimaging/testing/verify.py`
- `spimaging/training_common/dataset.py`, `device.py`, `events.py`, `recovery.py`, `security.py`, `utils.py`
- `tests/test_security_resume.py`, `tests/test_algorithm_events.py`
- `tests/test_training_integration.py` (one precise change: load produced checkpoints with `weights_only=True`)
- This handoff and `smoke-tests/T05_SMOKE_TEST.md`

## Smoke Test Result
- Command: `conda run -n spimaging python -m pytest -q tests/test_security_resume.py tests/test_algorithm_events.py`
- Result: Passed, `15 passed in 2.34s`.
- Affected legacy algorithms: `tests/test_training_integration.py tests/test_prediction_evaluation_integration.py tests/test_generation_models.py` passed, `14 passed in 24.09s`.
- T01 contract regression: `tests/test_appcore.py tests/test_worker.py` passed, `14 passed in 0.55s`.
- Compile check: owned Python paths and tests passed `python -m compileall -q`.

## Decisions Made
- NPZ defaults are 128 members, 512 MiB per member, 1 GiB total expansion, compression ratio <= 2000, at most five dimensions, and 2x free-memory headroom. Callers can pass narrower `ArchiveLimits`.
- Explicit CUDA is also fail-safe: missing/out-of-range/broken CUDA falls back to CPU and reports why rather than crashing startup.
- Training resume compatibility intentionally includes batch size, optimizer settings, seed, validation split, architecture, preprocessing, and loss settings; compute device and output path may change.
- Resume position is an epoch plus the next training batch. Validation is allowed to finish before honoring a late cancellation so a resumed run never applies the same completed training batch twice.
- Generation incomplete state stays outside the final output directory. Success first writes a `status=complete` generation manifest, then publishes staged files; only then is the empty partial directory removed.

## Open Issues For Downstream Context
- T_FINAL must align T01 worker generation flags: first run with no partial passes neither flag; `generation.resume=true` plus an existing sibling partial passes `--resume`; explicit replacement passes `--overwrite`; a completed nonempty output must fail unless the user explicitly selects replacement. Never use `--overwrite` as a resume synonym.
- T_FINAL must regenerate `record_of_SPI/Day_13-14/参数说明表.md` and `.csv` because the files are outside T05's boundary. The relevant legacy run was otherwise clean: `16 passed, 85 subtests passed`, with only the stale generated tables failing.
- The integrated branch now reports package version `0.2.0-beta.1`, while `tests/test_smoke.py::test_package_import` still expects `0.1.0`; T_FINAL must update that owned version assertion. The other selected smoke/demo/GUI/error tests passed (`30 passed`).
- CUDA selection/fallback logic is unit-covered for CPU/unavailable CUDA, but NVIDIA hardware, CUDA OOM, and Windows Job Object cancellation remain T06/T_FINAL machine-level acceptance work.
- Existing old checkpoints remain safe for prediction when weights-only compatible, but they cannot be used for training resume because they lack schema-v1 resume metadata; the error is explicit.

## Boundary Notes
- Stayed within write boundary: Yes.
- Did not edit or stage appcore, worker, desktop, launcher/packaging, root manifests, Tkinter GUI, T04 assets, or pre-existing user dirty files. Other agents' concurrent files were visible but left untouched. No git add/commit was run.

## Suggested Next Checks
- Update the worker generation adapter using `partial_directory_for(output_dir)` and the flag semantics above, then run a worker generation, cancel after one sample, and resume the same run.
- Regenerate the CLI parameter tables and update the package-version assertion, then run full pytest.
- On a Windows NVIDIA host, verify requested physical GPU mapping through `CUDA_VISIBLE_DEVICES`, CUDA self-test/fallback reason, cooperative cancel within the 10-second window, and checkpoint resume after restart.
