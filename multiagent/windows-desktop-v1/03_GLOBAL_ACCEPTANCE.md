# Global Acceptance

## Functional Acceptance

- The source desktop exposes quick demo, local data import/generation, sample inspection, supervised/self-supervised training configuration, prediction/evaluation, results gallery, history, settings, cancellation, and compatible resume controls.
- Simulation and reconstruction algorithms are separate selectors; prediction derives its algorithm from checkpoint metadata.
- A completed run has a versioned config, event log, full log, metrics/artifacts, and result manifest.
- Default gallery count is 4 and configurable from 1 through 12; unlabeled inputs hide target/error metrics.
- Auto device mode prefers NVIDIA CUDA and falls back to CPU with a user-visible reason.

## Technical Acceptance

- No `np.load(..., allow_pickle=True)` remains on user-controlled paths.
- Checkpoint loads use `weights_only=True` and validate mapping/schema fields.
- Runtime/download archives are hash-checked, safely extracted into staging, health-checked, and atomically activated.
- Existing CLI commands and tests remain compatible.
- No user-owned dirty workspace change is staged in implementation commits.
- Public release bundles contain only synthetic demo assets and required third-party/license notices.

## Packaging Acceptance

- A reproducible build path exists for launcher, CPU/CUDA runtime archives, app asset, and Inno Setup.
- `SPImaging-Setup.exe` is produced when Inno Setup is available; otherwise the exact missing external prerequisite and successful earlier build artifacts are reported.
- Signing is conditional on provided secrets/certificate and unsigned outputs are labeled beta.
- A clean Windows user can install without admin, without Python/Conda, and without PATH changes.

## Integration Acceptance

- All Txx handoffs are present and all smoke tests have an actual result.
- No task writes outside its boundary without explanation.
- Broad pytest, compile, source desktop smoke, worker smoke, and packaging dry-runs pass.

## Final Report Required

The final integration agent reports completed deliverables, integration fixes, exact tests/results, generated distributables, residual external blockers, and recommended follow-ups in `FINAL_STATUS_REPORT.md`.
