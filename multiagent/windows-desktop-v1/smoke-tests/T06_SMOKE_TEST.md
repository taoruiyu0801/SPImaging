# T06 Smoke Test: Cross-Review And QA

## Test Surface

- Integrated source regression and CLI compatibility.
- PySide6 desktop/model contracts and offscreen startup.
- Launcher/release manifest unit behavior and frozen launcher dry-run.
- Deterministic public demo verification.
- Python syntax/bytecode compilation.
- Manual CPU training, prediction, and evaluation for reconstruction models missing a committed full chain.
- Git whitespace/staging preservation.

No screenshots were taken, per user instruction.

## Procedure And Actual Results

### 1. Full regression

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:MPLBACKEND='Agg'
& 'C:\Users\32499\anaconda3\envs\spimaging\python.exe' -m pytest -q
```

Result: **PASS**, `159 passed, 104 subtests passed in 46.28s`.

An immediately preceding run exposed three gallery-fixture failures after the new full-pipeline checkpoint preflight. The integration owner corrected the fixture to disable unrelated prediction/evaluation stages. T06 reran the entire suite after that change; the result above is the authoritative final run.

### 2. Desktop/model target

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:MPLBACKEND='Agg'
& 'C:\Users\32499\anaconda3\envs\spimaging\python.exe' -m pytest -q tests/test_desktop.py tests/test_desktop_models.py
```

Result: **PASS**, `24 passed in 0.81s`.

### 3. Launcher/release target

```powershell
& 'C:\Users\32499\anaconda3\envs\spimaging\python.exe' -m pytest -q tests/test_launcher.py tests/test_release_manifest.py
```

Result: **PASS**, `29 passed, 12 subtests passed in 0.43s`.

### 4. Source desktop smoke

```powershell
$env:QT_QPA_PLATFORM='offscreen'
& 'C:\Users\32499\anaconda3\envs\spimaging\python.exe' -m spimaging.desktop --smoke-test
```

Result: **PASS**, exit `0`, no window and no output.

### 5. Public synthetic asset verification

```powershell
& 'C:\Users\32499\anaconda3\envs\spimaging\python.exe' scripts/generate_synthetic_demo.py --verify-only
```

Result: **PASS**, `Verified public demo assets: C:\Users\32499\Desktop\SPImaging\public_demo`.

### 6. Compile check

```powershell
& 'C:\Users\32499\anaconda3\envs\spimaging\python.exe' -m compileall -q launcher packaging/scripts spimaging tests
```

Result: **PASS**, exit `0`, no output.

### 7. Frozen unsigned launcher dry-run

```powershell
& '.\packaging\out\launcher\SPImaging.exe' `
  --manifest-file '.\packaging\manifests\release-manifest.unsigned-beta.example.json' `
  --dry-run --runtime auto
```

Result: **PASS**, exit `0`, no console output.

### 8. Manual model end-to-end

Using `public_demo/dataset`, T06 ran one-epoch CPU training, single-sample prediction, and four-sample evaluation for each missing model.

| Model | Train | Predict | Evaluate | Result |
| --- | ---: | ---: | ---: | --- |
| PRSNet | 4.44 s | 4.70 s | 4.50 s | Exit `0`; `(64,64)` finite prediction; `n_samples=4` |
| PENonLocal | 4.53 s | 4.35 s | 4.61 s | Exit `0`; `(64,64)` finite prediction; `n_samples=4` |
| STIN | 6.12 s | 4.48 s | 5.78 s | Exit `0`; `(64,64)` finite prediction; `n_samples=4` |
| SPISR | 4.84 s | 4.81 s | 4.51 s | Exit `0`; `(64,64)` finite prediction; `n_samples=4` |

Temporary outputs: `C:\Users\32499\AppData\Local\Temp\SPImaging-T06-model-e2e-20260824`.

This evidence passes the current implementation smoke but does not satisfy the explicit requirement for committed automated all-five-model end-to-end coverage.

### 9. Diff/staging checks

```powershell
git diff --check
git diff --cached --name-only
```

Result: **PASS**. `git diff --check` exits `0` with CRLF conversion warnings only. The cached diff is empty; no content is staged.

## Preview Artifact Snapshot

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `packaging/out/launcher/SPImaging.exe` | 11,463,524 | `41909236DE41F96281A63D6E9D653AB64781F766FD3075606572280111DE0BAB` |
| `packaging/out/SPImaging-Setup-unsigned-beta.exe` | 13,232,571 | `0A7CE52B130DE8AFBB5B4C5F85DFD0275AA2844943792F63C2C200B2E8D88C9B` |

These are unsigned preview files. No CPU/CUDA/app release ZIP, resolved lock set, or final release manifest was present.

## Expected Result

- Source tests and smoke commands pass.
- QA findings distinguish source correctness, fixed findings, open product defects, and external acceptance gaps.
- T06 leaves product/test code and staged Git content untouched.

## Final Result

- Source/integration smoke: **PASS**.
- Public distribution acceptance: **FAIL / NO-GO** due to the P1 findings in `reviews/T06_QA_REPORT.md`.
- Write-boundary compliance: **PASS**.
