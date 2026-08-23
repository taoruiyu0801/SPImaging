# T04 Agent Prompt: Synthetic Public Demo And Compliance

Read the constitution, task breakdown, and global acceptance first. No upstream handoff is required.

## Objective

Create deterministic, redistributable synthetic demo inputs/outputs, a reproducible lightweight checkpoint recipe, provenance/hash manifests, Apache-2.0 project licensing, CC0 asset licensing, notices and third-party inventory.

## Allowed Write Boundary

- `scripts/generate_synthetic_demo.py`
- `public_demo/**`
- `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, `SBOM.md`
- `tests/test_public_demo.py`
- T04 handoff and smoke files

## Forbidden Changes

- Do not copy or modify `example_data`, `demo_checkpoint`, algorithms, GUI, package manifests, or user dirty files.

## Required Deliverables

- Four deterministic synthetic 64x64/1024-bin NPZ samples with no external image data.
- Reproduction script, CC0 notice, source/parameter/hash manifest and validation test.
- Small Simple3D checkpoint generated only from those assets, or an explicit blocker plus deterministic build recipe.
- Apache-2.0 and third-party compliance files.

## Smoke Test

```powershell
conda run -n spimaging python -m pytest -q tests/test_public_demo.py
```
