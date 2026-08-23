# T06 QA Report: Windows Desktop Public Beta

## Review Snapshot

- Date: 2026-08-24 (Asia/Shanghai)
- Branch: `codex/windows-desktop-v1`
- Baseline: `4b03dc4`
- Reviewed HEAD: `2c03d78` plus the current integration worktree
- Scope: contracts, GUI/worker lifecycle, unsafe file handling, generation/training recovery, launcher/update/extraction, packaging, public assets, compliance, and test coverage
- Review mode: read-only. No product/test files were edited, staged, committed, published, installed, or deleted by T06.
- Screenshots: intentionally not taken, per user instruction.

## Executive Verdict

**Public distribution verdict: NO-GO.**

The source workbench is substantially functional and the final local regression run is green: `159 passed, 104 subtests passed`. The PySide6 source desktop, worker protocol, safe NPZ/checkpoint loading, synthetic demo, cancellation/recovery foundations, launcher preview, and unsigned installer preview are all real rather than placeholders.

The current candidate is still not safe to publish as the promised public beta. The release manifest has no trust root independent of the manifest itself; the signing workflow contains a PowerShell input-injection boundary and cannot complete reliably on its publish runner; launcher updates have no cross-process lock; runtime and app activation is not yet wired as one transaction; the beta discovery/publish channel is inconsistent; generation publication can become half-published and non-resumable; and required clean-Windows/NVIDIA/release-compliance acceptance has not been performed.

This is a **GO for continued source integration and internal testing**, not a GO for uploading `SPImaging-Setup.exe` to the public release channel.

## Severity Summary

| Severity | Open | Meaning |
| --- | ---: | --- |
| P0 | 0 | No verified immediate arbitrary-code execution or unrecoverable user-data-loss path in the tested source workflows. |
| P1 | 10 | Public-beta blockers or explicit plan/acceptance failures. |
| P2 | 9 | Important reliability, isolation, privacy, and completeness defects. |
| P3 | 1 | Documentation/release-polish follow-up. |

## P1 Findings: Release Blocking

### P1-01: Release manifest has no independent trust root

- **Location:** `launcher/update.py:90-112`, `launcher/manifest.py:96-135`, `launcher/manifest.py:283-313`, `launcher/signing.py:56-96`
- **Confidence:** 10/10
- **Evidence:** `ManifestCache.resolve()` accepts downloaded JSON after schema validation only. The same JSON supplies `unsigned_beta`, signature requirement, signer thumbprint, detached signature URL, and asset hashes. `WindowsSignatureVerifier` compares the artifact signer to that manifest-provided thumbprint.
- **Reproduction:** replace the remote manifest and assets together. A beta manifest can set `unsigned_beta=true` and `kind=none`; a nominally signed manifest can name the attacker's certificate thumbprint. There is no launcher-embedded publisher key/thumbprint or separately authenticated manifest signature to reject the replacement.
- **Impact:** artifact hash/signature verification does not authenticate the publisher if the control manifest is replaced. Manifest downgrade is also not prevented.
- **Blocking:** **Yes.** Pin a publisher policy in the launcher or verify a detached/signed manifest against a launcher-pinned trust root, then enforce monotonic release/channel rules.

### P1-02: Workflow inputs cross directly into PowerShell source

- **Location:** `.github/workflows/windows-release-candidate.yml:33-43`, `:57-61`, `:85-90`, `:161-180`, `:181-196`
- **Confidence:** 10/10
- **Evidence:** `${{ inputs.version }}`, `${{ github.ref }}`, and `${{ github.repository }}` are interpolated inside `run: |` PowerShell. The request gate checks string equality but does not validate strict SemVer before interpolation.
- **Reproduction:** dispatch from a crafted matching tag/input containing PowerShell metacharacters permitted by Git ref syntax. Expression substitution occurs before PowerShell parses the script.
- **Impact:** commands can execute on a signing runner that holds the PFX password, publisher certificate material, thumbprint, and release write token.
- **Blocking:** **Yes.** Pass values only through `env:`, validate strict SemVer/tag equality in code, and never splice GitHub expressions into executable PowerShell source.

### P1-03: Publish runner is structurally incomplete

- **Location:** `.github/workflows/windows-release-candidate.yml:91-98`, `:119-123`, `:130-180`
- **Confidence:** 9/10
- **Evidence:** the runtime artifact includes both `packaging/runtime/locks/*` and `packaging/out/spimaging-runtime-*`, whose common artifact root is `packaging`; downloading it to `packaging/out` can produce `packaging/out/out/...`, while publish commands search only the top-level `$env:OUT_DIR`. Separately, Inno Setup is installed only in `installer-preview`; the fresh `publish` runner calls `Build-Installer.ps1` without installing Inno.
- **Reproduction:** execute the publish job on a clean GitHub Windows runner. `Build-Installer.ps1` cannot find `ISCC.exe`; runtime ZIP discovery/signing may also miss the nested artifact paths.
- **Impact:** the signed publish job cannot reliably produce the final manifest and installer.
- **Blocking:** **Yes.** Normalize downloaded artifact layout explicitly and install/pin Inno Setup in the publish job itself.

### P1-04: Runtime/app activation is not yet a wired transaction

- **Location:** `launcher/bootstrap.py:80-140`, `launcher/activation.py:153-293`, `packaging/scripts/build_release_manifest.py:57-109`, `:156-175`
- **Confidence:** 10/10
- **Evidence:** `ActivationManager.activate_many()` now exists, but no production caller uses it. `Provisioner.provision()` still activates runtime first and app second through separate `install_asset()` calls. The app asset has `health_check=None`. The manifest contract now permits independently versioned runtimes, but the build script still assigns `args.version` to every asset.
- **Reproduction:** install an update whose runtime passes and whose app extraction/required-path check fails. Runtime state has already changed; the previous app stays active. There is no combined desktop smoke or automatic multi-component rollback.
- **Impact:** mixed runtime/app versions can become authoritative; app-only updates still cannot be produced by the current manifest builder without re-versioning runtimes.
- **Blocking:** **Yes.** Stage all changed components, run runtime relocation plus a combined desktop health check, then publish one state update through `activate_many()`; teach the manifest builder separate runtime versions.

### P1-05: No launcher single-instance/install lock

- **Location:** `launcher/app.py:158-220`, `launcher/activation.py:108-130`; repository-wide search of `launcher/*.py`
- **Confidence:** 10/10
- **Evidence:** launcher startup proceeds directly into manifest resolution/provisioning. Activation state uses atomic file replacement but there is no named mutex, file lock, or other inter-process exclusion around downloads, staging, activation, repair, or launch. Repository search found no production lock implementation.
- **Reproduction:** start two launcher instances, or start repair/update while another instance has launched the GUI/worker. Both may mutate cache, staging, release directories, and `activation-state.json` concurrently.
- **Impact:** concurrent update/repair can race, and the requirement “do not install updates while a task is running” is unenforced.
- **Blocking:** **Yes.** Add a per-install-root Windows mutex/file lock and a GUI/worker activity lease that blocks update/repair while work is active.

### P1-06: The intended beta release channel is not reachable

- **Location:** `launcher/app.py:27-30`, `.github/workflows/windows-release-candidate.yml:130-180`
- **Confidence:** 10/10
- **Evidence:** the launcher default is `releases/latest/download/...`; GitHub's `latest` release resolution excludes prereleases. The only GitHub publish job requires signing secrets, while the approved plan requires an unsigned public beta before a certificate is acquired.
- **Reproduction:** publish only `0.2.0-beta.1` as a prerelease and start a fresh launcher. The default URL does not discover it. Attempting the workflow's public upload without PFX secrets fails closed.
- **Impact:** the unsigned beta can exist as an Actions/installer preview but cannot use the designed update/distribution path.
- **Blocking:** **Yes.** Define a beta manifest URL/channel and an explicitly approved unsigned-beta publish path that cannot publish a stable filename or stable channel.

### P1-07: Generated dataset publication can be half-complete and non-resumable

- **Location:** `spimaging/generation/recovery.py:142-152`, `spimaging/generation/pipeline.py:215-235`, `:683-688`
- **Confidence:** 10/10
- **Evidence:** `session.complete()` writes the complete manifest and deletes the resumable partial marker before publication. `publish_generated_output()` then moves files individually into the final directory.
- **Reproduction:** terminate the process after one or more `source.replace(...)` calls but before all files are moved. The final output is partial and the sibling staging directory no longer has an `incomplete` manifest accepted by `--resume`.
- **Impact:** crash/cancellation recovery no longer satisfies “continue only unfinished samples,” and overwrite mode can expose a mixed old/new dataset.
- **Blocking:** **Yes.** Publish a complete directory through one atomic directory switch, or keep an authoritative recoverable journal until the final swap commits.

### P1-08: The GUI quick supervised preset is not the demo-light configuration

- **Location:** `spimaging/desktop/models.py:280-300`, `spimaging/appcore/specs.py:304-312`, `:391-404`, `public_demo/manifest.json:6-20`
- **Confidence:** 9/10
- **Evidence:** a new experiment defaults to `training_preset="quick"`; quick changes epochs/max samples but leaves Simple3D architecture defaults at `base_channels=8` and `temporal_downsample=1`. The deterministic demo uses `base_channels=2` and `temporal_downsample=64`.
- **Reproduction:** open New Experiment, leave Quick/Simple3D selected, and inspect the resolved config. It is materially heavier than the known demo-light path.
- **Impact:** first-run CPU training can be much slower and more memory intensive than promised.
- **Blocking:** **Yes for plan completion.** Add algorithm-specific quick overrides matching the current demo-light settings while keeping standard preset CLI defaults.

### P1-09: Five-model end-to-end coverage is manual, not automated

- **Location:** `tests/test_training_integration.py:83-135`, `tests/test_algorithm_events.py:113-148`, `tests/test_prediction_evaluation_integration.py:40-136`
- **Confidence:** 10/10
- **Evidence:** Simple3D has real training coverage and prediction/evaluation coverage. PRSNet, PENonLocal, and STIN only run a forward pass in the committed tests. SPISR's committed test cancels before completing training. T06 manually proved the missing four real chains, but no regression test preserves that proof.
- **Reproduction:** inspect the listed tests; none parametrizes all five algorithms through completed train -> predict -> four-sample evaluate.
- **Impact:** future model-specific CLI/checkpoint/data-shape regressions can pass CI despite violating an explicit acceptance requirement.
- **Blocking:** **Yes.** Add a CPU-minimal parametrized end-to-end regression for all five reconstruction models.

### P1-10: Final release/compliance inputs do not yet exist

- **Location:** `packaging/runtime/environment-*.in.yml`, `SBOM.md:13-17`, `THIRD_PARTY_LICENSES.md:41-45`, `packaging/inno/SPImaging.iss:48-50`, `public_demo/manifest.json:19`
- **Confidence:** 10/10
- **Evidence:** the worktree has an unsigned launcher and installer preview, but no actual CPU/CUDA conda-pack ZIP, app ZIP, resolved runtime locks, or final release manifest. `SBOM.md` explicitly remains source-level. Inno installs only `LICENSE` and `NOTICE`, not `THIRD_PARTY_LICENSES.md`. The demo checkpoint records Torch `2.11.0+cu130`, while runtime inputs pin PyTorch `2.5.1`.
- **Reproduction:** enumerate `packaging/out` and `packaging/runtime`: only preview EXEs and input environment YAMLs are present; no release ZIP/final manifest exists.
- **Impact:** reproducibility, native/transitive license inventory, runtime/checkpoint compatibility, and publish manifest verification cannot be claimed.
- **Blocking:** **Yes.** Build the real locked artifacts, regenerate/verify the demo checkpoint in both locked runtimes or align the pin, generate the final SBOM/license bundle, and inspect the final installer/app file lists.

## P2 Findings: Important Follow-Ups

| ID | Location | Finding and evidence | Blocking |
| --- | --- | --- | --- |
| P2-01 | `launcher/update.py:100-112`, `launcher/app.py:154-155` | Failed online checks do not call `record_check()`, so offline startup retries every time. `_is_update()` is inequality-only, so older manifests are treated as updates; there is no release-notes field/order enforcement. | Before public beta updater use |
| P2-02 | `launcher/bootstrap.py:151-165` | The private interpreter inherits `PYTHONHOME`, Conda variables, and the prior `PYTHONPATH`; the launcher appends the prior path after its app path. External Python state can break startup or shadow private-runtime imports. | Before clean/user-Python acceptance |
| P2-03 | `launcher/download.py:49-65`, `:88-147` | Only the final redirect URL is checked for HTTPS. An invalid complete `.part` starts a `Range: bytes=<size>-` request and can loop on HTTP 416 instead of discarding/restarting. | Before real network acceptance |
| P2-04 | `launcher/archive.py:16-32` | Windows reserved-name checks omit `CONIN$`, `CONOUT$`, and Unicode superscript COM/LPT aliases. | Before hostile-archive claim |
| P2-05 | `spimaging/appcore/specs.py:220-246` | Translucent-layer X/Y slopes and sinusoidal amplitude have no `visible_when`; parameters remain visible when the selected front type makes them ineffective. | No; UI correctness |
| P2-06 | `spimaging/supervised_training/train.py:424-440` | Resume signature omits `early_stopping_patience` and `early_stopping_min_delta`, despite the requirement that only target epochs may change. | Before strict-resume claim |
| P2-07 | `spimaging/appcore/diagnostics.py:16-42`, `:70-112` | Redaction covers home/AppData, `C:\Users\...`, and the explicit run directory, but arbitrary personal paths on other roots/volumes embedded in `run.json` can remain. A settings-only export also lacks launcher/update state. | Before asking users to share bundles |
| P2-08 | `spimaging/desktop/models.py:41-66` | Workflow/status/device labels are hard-coded Chinese constants outside Qt translation resources. English addition would require business-model edits. | No; violates translation organization goal |
| P2-09 | `spimaging/worker.py:646-656` | Failure/cancellation cleanup calls `_collect_results()` and persistence without a secondary guard. Under disk exhaustion, cleanup can raise before the authoritative terminal manifest/event is written. | Before disk-full acceptance |

## P3 Finding

### P3-01: Release handoffs are stale relative to the final integration worktree

- **Location:** `multiagent/windows-desktop-v1/handoffs/T03_HANDOFF.md`, `multiagent/windows-desktop-v1/smoke-tests/T03_SMOKE_TEST.md`
- **Confidence:** 10/10
- **Evidence:** T03 says no real launcher/installer was generated, while current untracked preview outputs include `packaging/out/launcher/SPImaging.exe` and `packaging/out/SPImaging-Setup-unsigned-beta.exe`.
- **Impact:** final reporting can understate available preview artifacts or confuse them with reproducible final release inputs.
- **Blocking:** No. T_FINAL should report the current preview hashes and explicitly distinguish them from locked runtime/app/final-manifest deliverables.

## Manual Reconstruction-Model End-to-End Evidence

T06 ran real CPU training, one-sample prediction, and four-sample evaluation against `public_demo/dataset` for the four models not covered by a committed full chain.

| Model | Train | Predict | Evaluate | Prediction | Evaluation |
| --- | ---: | ---: | ---: | --- | --- |
| PRSNet | 4.44 s | 4.70 s | 4.50 s | `(64, 64)`, finite | `n_samples=4` |
| PENonLocal | 4.53 s | 4.35 s | 4.61 s | `(64, 64)`, finite | `n_samples=4` |
| STIN | 6.12 s | 4.48 s | 5.78 s | `(64, 64)`, finite | `n_samples=4` |
| SPISR | 4.84 s | 4.81 s | 4.51 s | `(64, 64)`, finite | `n_samples=4` |

All 12 subprocesses exited `0`. Temporary evidence is under `C:\Users\32499\AppData\Local\Temp\SPImaging-T06-model-e2e-20260824`; it is not a release artifact. This changes the conclusion from “model implementation failure” to “missing automated regression coverage.”

## Global Acceptance Matrix

| Acceptance area | Status | Evidence / remaining gap |
| --- | --- | --- |
| PySide6 source workbench pages and navigation | PASS (source) | Offscreen desktop tests `24 passed`; source smoke exits `0`. No screenshot-based visual review was performed by request. |
| Simulation/reconstruction selectors and effective forms | PARTIAL | Separate registries and four generation models are tested; translucent conditional visibility remains incomplete. |
| RunConfig/events/result manifest/history | PASS (source) | Full suite passes; authoritative cancellation/interruption and incremental event/history fixes are covered. |
| Gallery 1-12, default 4, unlabeled behavior | PASS | Desktop/model tests pass, including labeled/unlabeled gallery cases. |
| Auto CUDA preference and CPU fallback | PARTIAL | Unit/source checks pass; real NVIDIA/CUDA and OOM acceptance remain external. |
| Safe NPZ/checkpoint loading | PASS (tested surface) | Pickle-free NPZ and `weights_only=True` security/recovery tests pass. |
| Cancellation and resume | PARTIAL | Cooperative/forced status, checkpoint resume, generated-data reuse, generation-partial GUI resume, and incremental evaluation are tested; atomic generation publication and disk-full terminal persistence remain open. |
| Five reconstruction models | PARTIAL | Manual real E2E passes for missing four; committed automated all-five train/predict/evaluate matrix is absent. |
| Four simulation models | PASS | Existing generation-model end-to-end tests pass inside the green full suite. |
| Launcher download/extraction/update | FAIL | Unit tests pass, but trust root, single-instance lock, transaction wiring, beta channel, and network edge cases block release. |
| Reproducible release build | PARTIAL | Launcher and unsigned installer previews exist; CPU/CUDA/app archives, final manifest, resolved locks, and final SBOM do not. |
| Public synthetic assets | PASS (source) | Deterministic verifier passes; four CC0 samples and Simple3D checkpoint are present. Locked-runtime checkpoint reproducibility remains unresolved because Torch versions differ. |
| Licenses/SBOM in final public bundle | FAIL/PENDING | Source notices exist; final runtime/native inventory is absent and Inno does not install `THIRD_PARTY_LICENSES.md`. |
| CLI compatibility and broad regression | PASS | `159 passed, 104 subtests passed in 46.28s`. |
| No staged user dirty changes | PASS | `git diff --cached --name-only` is empty. Existing dirty/untracked worktree content was preserved. |
| Standard-user clean Windows install, no Python/Conda, spaces/Chinese path | NOT TESTED | Requires clean Windows VM or classmate machine. |
| NVIDIA hardware, SmartScreen, proxy, offline restart, update rollback, uninstall preservation | NOT TESTED | Requires external Windows/NVIDIA/signing/network acceptance. |

## Findings Verified As Fixed And Excluded From The Open List

- `evaluation.dataset_dir` is honored by the worker.
- Forced cancellation and startup recovery persist authoritative terminal result-manifest/history status.
- Full-pipeline prediction/evaluation toggles are honored.
- Registry/CLI numeric bounds and odd-kernel validation are aligned.
- CUDA resume loads RNG tensors on CPU before restoring state.
- Matplotlib implementation caches are excluded from result manifests.
- `pythonw.exe` desktop launch can still create a worker with functional QProcess JSONL pipes on this Windows host.
- Generated-data training resume now reuses the original fingerprinted dataset; generation-only partial runs resume in the original run directory.
- Evaluation metrics/progress are atomically persisted after every completed sample/model, including failure state.
- RunConfig integer fields now reject floats, strings, and booleans instead of coercing them.
- ZIP inspection checks actual expansion size and free space on the staging/install volume before extraction.
- Volume scattering nullable medium defaults and fog/water front-boost visibility are aligned with CLI behavior.
- SPISR standard preset uses self-supervised CLI `weight_decay=1e-6`.
- The temporary gallery-fixture conflict introduced by full-pipeline checkpoint preflight is fixed; final full pytest is green.

## Exact Test Evidence

| Command / check | Result |
| --- | --- |
| `$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; ...python.exe -m pytest -q` | `159 passed, 104 subtests passed in 46.28s` |
| `...python.exe -m pytest -q tests/test_desktop.py tests/test_desktop_models.py` | `24 passed in 0.81s` |
| `...python.exe -m pytest -q tests/test_launcher.py tests/test_release_manifest.py` | `29 passed, 12 subtests passed in 0.43s` |
| `...python.exe -m spimaging.desktop --smoke-test` with offscreen Qt | Exit `0` |
| `...python.exe scripts/generate_synthetic_demo.py --verify-only` | Verified `public_demo` assets |
| `...python.exe -m compileall -q launcher packaging/scripts spimaging tests` | Exit `0` |
| Frozen `packaging/out/launcher/SPImaging.exe --manifest-file ... --dry-run --runtime auto` | Exit `0`, no console output |
| `git diff --check` | Exit `0`; CRLF conversion warnings only |
| `git diff --cached --name-only` | Empty; no staged content |

Preview artifact snapshot:

- `packaging/out/launcher/SPImaging.exe`: 11,463,524 bytes, SHA-256 `41909236DE41F96281A63D6E9D653AB64781F766FD3075606572280111DE0BAB`
- `packaging/out/SPImaging-Setup-unsigned-beta.exe`: 13,232,571 bytes, SHA-256 `0A7CE52B130DE8AFBB5B4C5F85DFD0275AA2844943792F63C2C200B2E8D88C9B`
- These are unsigned preview artifacts, not a complete public release; no CPU/CUDA/app release ZIP or final release manifest is present.

## Required External Acceptance Before Public Upload

1. Build real CPU/CUDA conda-pack archives, app ZIP, split parts/signatures, resolved locks, final release manifest, final SBOM, and full runtime/native license inventory.
2. Fix P1-01 through P1-06, then exercise publish from a protected release environment using non-production test credentials before using the real certificate.
3. On a clean Windows 10/11 x64 standard-user VM with no Python/Conda: install from a Chinese/space-containing path, complete first online provisioning, reboot offline, run the synthetic demo, repair, update, roll back, and uninstall while retaining user results.
4. On an NVIDIA Windows host: verify driver selection, CUDA health, GPU index mapping, forced CUDA failure/CPU fallback reason, CUDA OOM behavior, cancellation of the entire Job Object tree, and checkpoint resume.
5. Test system proxy, interrupted/resumed downloads, HTTP 416 recovery, hash/signature failure, insufficient cache/install disk, SmartScreen messaging, and concurrent-launch/update exclusion.
6. Add and run the automated five-model end-to-end matrix and a crash-injection test around generation publication.

## Final Recommendation

Keep the branch as an internal/source beta candidate, land the verified integration fixes, and do not publish the installer until every P1 item is closed and the external acceptance matrix has recorded artifacts, machine details, commands, and results. The current preview EXEs are suitable for controlled testing by the classmate only when clearly labeled unsigned and accompanied by the known limitations; they are not yet a public-release candidate.
