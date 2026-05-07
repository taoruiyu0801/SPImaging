"""Evaluate one or more reconstruction checkpoints on generated samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SPAD reconstruction checkpoints.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path. Can be passed multiple times.")
    parser.add_argument("--label", action="append", default=None, help="Model label for each checkpoint.")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing sample_*.npz files.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics and figures.")
    parser.add_argument("--figure_index", type=int, default=0, help="Sample index used for the comparison figure.")
    return parser.parse_args()


def list_samples(dataset_dir: Path):
    files = sorted(dataset_dir.glob("sample_*.npz"))
    if not files:
        files = sorted(dataset_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz samples found in {dataset_dir}")
    return files


def compute_metrics(pred, target):
    err = pred - target
    abs_err = np.abs(err)
    return {
        "mae_m": float(np.mean(abs_err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "abs_rel": float(np.mean(abs_err / np.maximum(target, 1e-6))),
    }


def predict_one(checkpoint, sample, device):
    import torch
    import torch.nn.functional as F

    from spimaging.testing.predict import expected_depth_from_logits, load_model, match_input_time_bins
    from spimaging.training_common.dataset import SPADHistogramDataset, SPISRSelfSupervisedDataset

    model, train_args, method_family = load_model(checkpoint, device)
    temporal_downsample = int(train_args.get("temporal_downsample", 1))
    if method_family == "self_supervised_spisr":
        dataset = SPISRSelfSupervisedDataset(
            [sample],
            temporal_downsample=temporal_downsample,
            spatial_downsample=int(train_args.get("spatial_downsample", 1)),
            normalize=not bool(train_args.get("no_normalize", False)),
        )
        x_key = "measurement"
    else:
        dataset = SPADHistogramDataset(
            [sample],
            temporal_downsample=temporal_downsample,
            target_sigma_bins=float(train_args.get("target_sigma_bins", 2.0)),
            target_source=str(train_args.get("target_source", "depth")),
            use_log_counts=not bool(train_args.get("no_log_counts", False)),
        )
        x_key = "input"

    item = dataset[0]
    x = item[x_key].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if method_family != "self_supervised_spisr":
            logits = match_input_time_bins(logits, x)
        bin_size = item.get("bin_size", torch.tensor(80e-12, dtype=torch.float32))
        effective_downsample = temporal_downsample
        if method_family == "self_supervised_spisr":
            effective_downsample = temporal_downsample / float(train_args.get("time_scale", 2))
        pred = expected_depth_from_logits(logits, bin_size, effective_downsample)

    pred = pred.squeeze(0).cpu().numpy().astype(np.float32)
    raw = np.load(sample, allow_pickle=True)
    target = raw["depth_m"].astype(np.float32)
    if pred.shape != target.shape:
        target_tensor = torch.from_numpy(target[None, None, ...]).float()
        target = (
            F.interpolate(target_tensor, size=pred.shape, mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
            .astype(np.float32)
        )
    return pred, target


def save_comparison_figure(output_path, sample_path, target, predictions):
    os.environ.setdefault("MPLCONFIGDIR", str((output_path.parent / ".matplotlib-cache").resolve()))
    import matplotlib.pyplot as plt

    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4.2 * (n_models + 1), 7.2))
    axes = np.atleast_2d(axes)

    im = axes[0, 0].imshow(target, cmap="viridis")
    axes[0, 0].set_title("Target depth (m)")
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, Path(sample_path).name, ha="center", va="center")

    for col, (label, pred) in enumerate(predictions, start=1):
        im = axes[0, col].imshow(pred, cmap="viridis", vmin=float(target.min()), vmax=float(target.max()))
        axes[0, col].set_title(f"{label} prediction")
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        err = np.abs(pred - target)
        im = axes[1, col].imshow(err, cmap="magma")
        axes[1, col].set_title(f"{label} abs error (m)")
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    labels = args.label or [Path(p).parent.name for p in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        raise ValueError("--label must be passed the same number of times as --checkpoint")

    import torch

    from spimaging.training_common.device import get_torch_device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = list_samples(Path(args.dataset_dir))
    device = get_torch_device()

    rows = []
    summary = {}
    figure_predictions = []
    figure_target = None
    figure_sample = samples[min(max(args.figure_index, 0), len(samples) - 1)]

    for label, checkpoint in zip(labels, args.checkpoint):
        per_model = []
        for sample in samples:
            pred, target = predict_one(checkpoint, sample, device)
            metrics = compute_metrics(pred, target)
            row = {"model": label, "sample": sample.name, **metrics}
            rows.append(row)
            per_model.append(metrics)
            if sample == figure_sample:
                figure_predictions.append((label, pred))
                figure_target = target

        summary[label] = {
            "n_samples": len(per_model),
            "mae_m": float(np.mean([m["mae_m"] for m in per_model])),
            "rmse_m": float(np.mean([m["rmse_m"] for m in per_model])),
            "abs_rel": float(np.mean([m["abs_rel"] for m in per_model])),
        }

    with open(output_dir / "metrics_per_sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "sample", "mae_m", "rmse_m", "abs_rel"])
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if figure_target is not None:
        save_comparison_figure(output_dir / "comparison.png", figure_sample, figure_target, figure_predictions)

    print(f"Device: {device}")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics and figure to: {output_dir}")


if __name__ == "__main__":
    main()
