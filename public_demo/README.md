# SPImaging Public Synthetic Demo

This directory is the only demo dataset/checkpoint set intended for the
`0.2.0-beta.1` public Windows bundle. It contains no NYUv2, Middlebury, or other
external image data.

## Contents

- `dataset/`: four deterministic `64 x 64 x 1024` SPAD NPZ samples. RGB,
  metric depth, albedo, intensity, and photon counts all originate from
  analytic geometry in `scripts/generate_synthetic_demo.py`.
- `checkpoint/simple3d_synthetic.pt`: a two-channel Simple3D checkpoint trained
  only on centered, strided views of those four samples. It is a workflow demo,
  not a claim of general reconstruction accuracy.
- `manifest.json`: portable provenance, generation parameters, array schemas,
  training-input hashes, file sizes, and SHA-256 hashes.
- `CC0_NOTICE.md` and `CC0-1.0.txt`: the asset dedication and complete CC0 legal
  text.

## Reproduce and verify

From the repository root, with the locked SPImaging training environment:

```powershell
python scripts/generate_synthetic_demo.py --force
python scripts/generate_synthetic_demo.py --verify-only
```

To reproduce only the dataset without importing PyTorch:

```powershell
python scripts/generate_synthetic_demo.py `
  --output-root path/to/empty/public_demo `
  --samples-only
```

The sample archives use sorted NPY members, fixed ZIP metadata, explicit
non-object dtypes, PCG64 seeds, and `allow_pickle=False` validation. Exact
checkpoint bytes are reproducible in the locked release Python/NumPy/PyTorch
runtime recorded in the manifest; other framework builds can differ in
floating-point details while following the same deterministic recipe.

## Licensing

The files listed as assets in `manifest.json` are released under CC0-1.0. The
generator source remains under the repository's Apache-2.0 license. No
trademark or patent rights are granted by CC0.
