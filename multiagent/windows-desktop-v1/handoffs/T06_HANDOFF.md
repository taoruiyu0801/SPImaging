# T06 Handoff: Cross-Review And QA

## Status

- State: Complete
- Verdict: **NO-GO for public distribution; GO for continued source/internal testing**
- Reviewed snapshot: `codex/windows-desktop-v1`, HEAD `2c03d78` plus current integration worktree
- Report: `multiagent/windows-desktop-v1/reviews/T06_QA_REPORT.md`

## What Changed

- Completed the read-only cross-review of contracts, GUI/worker lifecycle, loading safety, cancellation/recovery, downloader/extraction/update behavior, release workflows, public assets, licenses, and acceptance coverage.
- Classified verified P0-P3 findings with exact source locations, evidence/reproduction, and release-blocking status.
- Added a global acceptance matrix and separated source-test success from external Windows/NVIDIA/signing/compliance work.
- Incorporated real CPU train -> predict -> four-sample evaluate evidence for PRSNet, PENonLocal, STIN, and SPISR. All 12 subprocess stages exited `0`.
- Recorded the final green regression baseline and preview binary hashes.

## Files Touched

- `multiagent/windows-desktop-v1/reviews/T06_QA_REPORT.md`
- `multiagent/windows-desktop-v1/handoffs/T06_HANDOFF.md`
- `multiagent/windows-desktop-v1/smoke-tests/T06_SMOKE_TEST.md`

No product, test, release, source-documentation, user-data, staging-area, or Git state was modified by T06.

## Smoke Test Result

- Full suite: `159 passed, 104 subtests passed in 46.28s`.
- Desktop/model target: `24 passed in 0.81s`.
- Launcher/release target: `29 passed, 12 subtests passed in 0.43s`.
- Desktop offscreen smoke, public-demo verification, compileall, frozen launcher dry-run, and `git diff --check`: exit `0`.
- Staged content: none.
- Full details and commands are in `smoke-tests/T06_SMOKE_TEST.md`.

## Decisions Made

- Green source tests are necessary but not sufficient for a public installer/updater.
- Manual four-model E2E success proves the current implementations can complete; the open defect is the absence of committed automated all-five-model E2E coverage.
- Recently fixed findings are listed separately and excluded from the open issue count.
- Preview `SPImaging.exe` and `SPImaging-Setup-unsigned-beta.exe` are not treated as final release artifacts because CPU/CUDA/app ZIPs, resolved locks, final manifest, final SBOM, and external acceptance are absent.
- No screenshots were taken, per user instruction.

## Open Issues For Downstream Context

Release-blocking P1 themes:

1. Manifest publisher policy is controlled by the unauthenticated manifest itself.
2. Workflow values are interpolated directly into signing-runner PowerShell.
3. Publish artifact layout and Inno installation are incomplete on the publish runner.
4. Runtime/app `activate_many()` transaction exists but is not called; app has no health check and build tooling still shares one asset version.
5. Launcher/update has no single-instance or active-task exclusion lock.
6. GitHub `latest` does not discover prereleases and unsigned beta has no public workflow path.
7. Generated output publication is file-by-file after removing the resumable marker.
8. GUI quick Simple3D keeps heavy architecture defaults instead of demo-light values.
9. Automated five-model train/predict/evaluate coverage is missing.
10. Real runtime/app release assets, final SBOM/licenses, aligned checkpoint runtime, and external Windows/NVIDIA acceptance are missing.

See the QA report for P2 isolation/downloader/archive/resume/privacy/i18n/disk-full details.

## Boundary Notes

- Stayed within write boundary: Yes.
- No `git add`, commit, restore, checkout, reset, release upload, installer execution, screenshot, or remote mutation was performed.
- Existing user/agent dirty files and untracked `packaging/out` artifacts were preserved.

## Suggested Next Checks

1. Close P1-01 through P1-06 before exposing updater/release credentials.
2. Wire staged runtime/app changes through one `activate_many()` transaction with a combined desktop health check.
3. Add atomic generation directory publication and crash-injection recovery tests.
4. Add a CPU-minimal parametrized five-model end-to-end regression.
5. Build and verify the real release assets, then run the clean Windows and NVIDIA external acceptance checklist from the QA report.
