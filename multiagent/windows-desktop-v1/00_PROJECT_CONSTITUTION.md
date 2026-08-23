# Project Constitution

## Mission

Deliver SPImaging `0.2.0-beta.1` as a Chinese-first Windows 10/11 x64 desktop workbench with a PySide6 UI, isolated worker processes, structured experiment records, safe cancellation/recovery, synthetic public demo assets, and a per-user installer whose launcher provisions a versioned private CPU/CUDA runtime from GitHub Releases.

## Non-Goals

- Do not remove or break the current CLI commands or Tkinter demo.
- Do not redistribute the existing NYUv2-derived `example_data` or checkpoint in public release assets.
- Do not download NYUv2/Middlebury data, install GPU drivers, add AMD acceleration, telemetry, background execution after GUI exit, or require administrator access.
- Do not promise a signed production release before an Authenticode certificate is available.

## Source Of Truth

- User-approved implementation plan in the Codex task that created this workpack.
- Existing CLI behavior and validation in `spimaging/` and current test suite.
- Public release source: `https://github.com/ewellchen/SPImaging` GitHub Releases.
- Baseline commit: `4b03dc4`; implementation branch: `codex/windows-desktop-v1`.

## Technical Decisions

- Desktop: PySide6, with the existing Tkinter UI retained.
- Contracts: versioned JSON `RunConfig`, `WorkerEvent`, `ResultManifest`, and `ReleaseManifest` using standard-library dataclasses and explicit validation.
- Execution: one worker process per run, JSONL events, Windows Job Object ownership, cooperative cancellation followed by a 10-second hard-stop window.
- Storage: versioned JSON files in each run directory plus SQLite as a rebuildable history index.
- Runtime: locked CPU and CUDA conda-pack archives; no dependency solver on the user's computer.
- Installer: per-user Inno Setup plus a small bootstrap launcher; runtime/app releases are hash-verified and atomically activated.
- Tests: `conda run -n spimaging python -m pytest -q` and component smoke tests documented per task.

## Global Write Rules

- Agents may only edit files inside their assigned write boundary.
- Agents must not stage, commit, restore, or alter the user's pre-existing dirty files unless the task explicitly owns a precise hunk.
- `pyproject.toml`, environment files, README, package version, and final integration glue are owned by T_FINAL.
- Shared contracts are owned by T01. Downstream tasks consume them and record requested changes rather than rewriting them.
- Destructive commands, broad rewrites, remote publishing, certificate creation, and release uploads are forbidden.
- Every task must update its handoff and smoke-test result before finishing.

## Conflict Resolution

1. Follow this constitution.
2. Follow the owning task prompt.
3. Follow upstream handoff decisions.
4. Record unresolved issues instead of editing outside the boundary.

## Required Finish State

- Existing public CLIs remain compatible and the full pytest suite passes.
- All Txx handoffs and smoke results are complete.
- Desktop and worker can run from source even if a distributable runtime cannot be downloaded during CI.
- Installer/launcher build inputs are reproducible and fail closed when release assets or hashes are missing.
- Public assets are synthetic, deterministic, documented, and separately licensed for redistribution.
