"""Evaluate one or more reconstruction checkpoints on generated samples."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import pickle
import zipfile

import numpy as np

from spimaging.appcore.storage import atomic_write_text
from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    add_device_arguments,
    create_output_directory,
    nonnegative_int,
    require_directory,
    require_file,
    validate_npz_archive,
    validate_output_directory,
)


_METRIC_FIELDS = ["model", "sample", "mae_m", "rmse_m", "abs_rel"]


def _model_summary(metrics):
    return {
        "n_samples": len(metrics),
        "mae_m": float(np.mean([item["mae_m"] for item in metrics])),
        "rmse_m": float(np.mean([item["rmse_m"] for item in metrics])),
        "abs_rel": float(np.mean([item["abs_rel"] for item in metrics])),
    }


def _persist_progress(output_dir, rows, summary, *, status, error=None):
    """Atomically retain every completed evaluation result.

    The regular CSV/summary paths always contain only fully computed rows.  A
    separate progress record distinguishes partial/failed output from a
    successful comparison so callers never need to infer completion from file
    presence alone.
    """

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_METRIC_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(output_dir / "metrics_per_sample.csv", buffer.getvalue())
    atomic_write_text(
        output_dir / "metrics_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    progress = {
        "schema_version": 1,
        "status": status,
        "completed_rows": len(rows),
        "completed_models": sum(
            1 for item in summary.values() if item.get("complete", False)
        ),
    }
    if error:
        progress["error"] = str(error)
    atomic_write_text(
        output_dir / "evaluation_progress.json",
        json.dumps(progress, indent=2, ensure_ascii=False) + "\n",
    )


def build_parser():
    parser = ArgumentParser(
        prog="spad-evaluate",
        description="Evaluate one or more SPAD reconstruction checkpoints.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Path to a trained .pt or .pth checkpoint; repeat for multiple models.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Display label for a checkpoint; when used, repeat once per --checkpoint.",
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Directory containing generated sample_*.npz files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for metric files and the comparison figure.",
    )
    parser.add_argument(
        "--figure_index",
        type=nonnegative_int,
        default=0,
        metavar="INDEX",
        help="Zero-based sample index used for the comparison figure (must exist).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    add_device_arguments(parser)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


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
            include_metadata=True,
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
    if "depth_m" not in item:
        raise ValueError(f"evaluation sample has no depth_m target: {sample}")
    target = item["depth_m"].squeeze(0).cpu().numpy().astype(np.float32)
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
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
    )
    try:
        plt.savefig(temporary, dpi=200, bbox_inches="tight")
        os.replace(temporary, output_path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    labels = args.label or [Path(p).parent.name for p in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        parser.error("--label must be passed the same number of times as --checkpoint")

    dataset_dir = require_directory(parser, args.dataset_dir, "--dataset_dir")
    checkpoints = [
        require_file(
            parser,
            value,
            "--checkpoint",
            suffixes=(".pt", ".pth"),
        )
        for value in args.checkpoint
    ]
    try:
        samples = list_samples(dataset_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.figure_index >= len(samples):
        parser.error(
            f"--figure_index {args.figure_index} is out of range for "
            f"{len(samples)} sample(s); expected 0 to {len(samples) - 1}"
        )
    for sample in samples:
        validate_npz_archive(
            parser,
            sample,
            "--dataset_dir sample",
            required_keys=("counts", "depth_m"),
        )

    output_dir = validate_output_directory(
        parser,
        args.output_dir,
        overwrite=args.overwrite,
        option="--output_dir",
    )
    from spimaging.testing.predict import load_model
    from spimaging.training_common.device import get_torch_device

    selection = get_torch_device(
        mode=args.device,
        gpu_index=args.gpu_index,
        return_selection=True,
    )
    device = selection.device
    rows = []
    summary = {}
    figure_predictions = []
    figure_target = None
    figure_sample = samples[args.figure_index]
    output_prepared = False

    try:
        for label, checkpoint in zip(labels, checkpoints):
            try:
                model, _, _ = load_model(checkpoint, device)
            except (
                EOFError,
                IndexError,
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                pickle.UnpicklingError,
            ) as exc:
                raise ValueError(f"cannot load --checkpoint {checkpoint}: {exc}") from exc
            del model
            if not output_prepared:
                create_output_directory(parser, output_dir, option="--output_dir")
                _persist_progress(output_dir, rows, summary, status="running")
                output_prepared = True
            per_model = []
            for sample in samples:
                pred, target = predict_one(checkpoint, sample, device)
                metrics = compute_metrics(pred, target)
                row = {"model": label, "sample": sample.name, **metrics}
                rows.append(row)
                per_model.append(metrics)
                summary[label] = {**_model_summary(per_model), "complete": False}
                _persist_progress(output_dir, rows, summary, status="running")
                if sample == figure_sample:
                    figure_predictions.append((label, pred))
                    figure_target = target

            summary[label] = {**_model_summary(per_model), "complete": True}
            _persist_progress(output_dir, rows, summary, status="running")
    except (
        EOFError,
        IndexError,
        KeyError,
        MemoryError,
        OSError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
    ) as exc:
        if output_prepared:
            _persist_progress(output_dir, rows, summary, status="failed", error=exc)
        parser.error(f"cannot evaluate the supplied checkpoint or dataset: {exc}")

    if figure_target is not None:
        try:
            save_comparison_figure(
                output_dir / "comparison.png",
                figure_sample,
                figure_target,
                figure_predictions,
            )
        except (MemoryError, OSError, RuntimeError, ValueError) as exc:
            _persist_progress(output_dir, rows, summary, status="failed", error=exc)
            parser.error(f"cannot save evaluation comparison: {exc}")
    _persist_progress(output_dir, rows, summary, status="complete")

    print(f"Device: {device}")
    if selection.fallback:
        print(f"Device selection: {selection.reason}")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics and figure to: {output_dir}")


if __name__ == "__main__":
    main()
