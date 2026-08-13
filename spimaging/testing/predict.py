"""Run inference with a trained 3D SPAD histogram network."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import pickle
import zipfile

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    create_output_parent,
    require_file,
    validate_npz_archive,
    validate_output_file,
)


def build_parser():
    parser = ArgumentParser(
        prog="spad-predict",
        description="Predict depth from one generated SPAD .npz sample.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained .pt or .pth checkpoint.",
    )
    parser.add_argument(
        "--sample_file",
        "--sample",
        dest="sample_file",
        required=True,
        help="Path to one generated .npz sample; --sample is a compatibility alias.",
    )
    parser.add_argument(
        "--output_npz",
        default="outputs/prediction.npz",
        help="Path for the compressed .npz prediction output.",
    )
    parser.add_argument(
        "--output_fig",
        default=None,
        help="Optional path for a PNG, JPEG, PDF, or SVG comparison figure.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def load_model(checkpoint, device):
    import torch

    from spimaging.training_common.networks import build_model, build_self_supervised_model, canonical_model_name

    ckpt = torch.load(checkpoint, map_location=device)
    if not isinstance(ckpt, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, Mapping):
        raise ValueError("checkpoint field 'args' must be a mapping")
    if "model_state" not in ckpt or not isinstance(ckpt["model_state"], Mapping):
        raise ValueError("checkpoint field 'model_state' is missing or invalid")
    method_family = ckpt.get("method_family", "supervised")
    model_name = canonical_model_name(ckpt.get("model_name", train_args.get("model", "simple3d")))
    base_channels = int(train_args.get("base_channels", 8))
    num_blocks = int(train_args.get("num_blocks", 10))
    if method_family == "self_supervised_spisr":
        model = build_self_supervised_model(
            model_name,
            in_channels=1,
            base_channels=base_channels,
            num_blocks=num_blocks,
            time_scale=int(train_args.get("time_scale", 2)),
            spatial_scale=int(train_args.get("spatial_scale", 2)),
        ).to(device)
    else:
        model = build_model(
            model_name,
            in_channels=1,
            base_channels=base_channels,
            num_blocks=num_blocks,
        ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, train_args, method_family


def expected_depth_from_logits(logits, bin_size, temporal_downsample):
    import torch

    from spimaging.training_common.dataset import LIGHT_SPEED_M_PER_S

    probs = torch.softmax(logits.squeeze(1), dim=1)
    t = probs.shape[1]
    bins = torch.arange(t, device=probs.device, dtype=probs.dtype).view(1, t, 1, 1)
    expected_bin = (probs * bins).sum(dim=1)
    effective_bin_size = float(bin_size) * float(temporal_downsample)
    return expected_bin * effective_bin_size * LIGHT_SPEED_M_PER_S / 2.0


def match_input_time_bins(logits, inputs):
    import torch.nn.functional as F

    if logits.shape[2:] == inputs.shape[2:]:
        return logits
    return F.interpolate(logits, size=inputs.shape[2:], mode="trilinear", align_corners=False)


def main():
    parser = build_parser()
    args = parser.parse_args()

    checkpoint = require_file(
        parser,
        args.checkpoint,
        "--checkpoint",
        suffixes=(".pt", ".pth"),
    )
    sample_file = require_file(
        parser,
        args.sample_file,
        "--sample_file",
        suffixes=(".npz",),
    )
    validate_npz_archive(parser, sample_file, "--sample_file", required_keys=("counts",))
    output_npz = validate_output_file(
        parser,
        args.output_npz,
        overwrite=args.overwrite,
        option="--output_npz",
        suffixes=(".npz",),
    )
    if output_npz.resolve() == sample_file.resolve():
        parser.error("--output_npz must not overwrite --sample_file")
    output_fig = None
    if args.output_fig is not None:
        output_fig = validate_output_file(
            parser,
            args.output_fig,
            overwrite=args.overwrite,
            option="--output_fig",
            suffixes=(".png", ".jpg", ".jpeg", ".pdf", ".svg"),
        )

    import numpy as np
    import torch
    import torch.nn.functional as F

    from spimaging.training_common.dataset import SPADHistogramDataset, SPISRSelfSupervisedDataset
    from spimaging.training_common.device import get_torch_device

    device = get_torch_device()
    try:
        model, train_args, method_family = load_model(checkpoint, device)
    except (EOFError, IndexError, KeyError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        parser.error(f"cannot load --checkpoint {checkpoint}: {exc}")

    temporal_downsample = int(train_args.get("temporal_downsample", 1))
    if method_family == "self_supervised_spisr":
        dataset = SPISRSelfSupervisedDataset(
            [sample_file],
            temporal_downsample=temporal_downsample,
            spatial_downsample=int(train_args.get("spatial_downsample", 1)),
            normalize=not bool(train_args.get("no_normalize", False)),
        )
    else:
        dataset = SPADHistogramDataset(
            [sample_file],
            temporal_downsample=temporal_downsample,
            target_sigma_bins=float(train_args.get("target_sigma_bins", 2.0)),
            target_source=str(train_args.get("target_source", "depth")),
            use_log_counts=not bool(train_args.get("no_log_counts", False)),
        )
    try:
        item = dataset[0]
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(f"cannot read --sample_file {sample_file}: {exc}")
    x_key = "measurement" if method_family == "self_supervised_spisr" else "input"
    x = item[x_key].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        if method_family != "self_supervised_spisr":
            logits = match_input_time_bins(logits, x)
        bin_size = item.get("bin_size", torch.tensor(80e-12, dtype=torch.float32))
        effective_temporal_downsample = temporal_downsample
        if method_family == "self_supervised_spisr":
            effective_temporal_downsample = temporal_downsample / float(train_args.get("time_scale", 2))
        pred_depth_m = expected_depth_from_logits(logits, bin_size, effective_temporal_downsample)

    pred_depth_m = pred_depth_m.squeeze(0).cpu().numpy().astype(np.float32)
    try:
        raw = np.load(sample_file, allow_pickle=True)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(f"cannot read --sample_file {sample_file}: {exc}")
    target_depth_m = raw["depth_m"].astype(np.float32) if "depth_m" in raw else None

    create_output_parent(parser, output_npz, option="--output_npz")
    save_dict = {"pred_depth_m": pred_depth_m, "sample": str(sample_file), "method_family": method_family}
    if method_family == "self_supervised_spisr":
        save_dict["pred_cube"] = logits.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    if target_depth_m is not None:
        if pred_depth_m.shape != target_depth_m.shape:
            target_tensor = torch.from_numpy(target_depth_m[None, None, ...]).float()
            target_depth_m = (
                F.interpolate(target_tensor, size=pred_depth_m.shape, mode="bilinear", align_corners=False)
                .squeeze()
                .numpy()
                .astype(np.float32)
            )
        save_dict["target_depth_m"] = target_depth_m
        save_dict["abs_error_m"] = np.abs(pred_depth_m - target_depth_m)
    np.savez_compressed(output_npz, **save_dict)
    print(f"Saved prediction to: {output_npz}")

    if output_fig is not None:
        import os

        create_output_parent(parser, output_fig, option="--output_fig")
        os.environ.setdefault("MPLCONFIGDIR", str((output_fig.parent / ".matplotlib-cache").resolve()))
        import matplotlib.pyplot as plt

        ncols = 3 if target_depth_m is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
        axes = np.atleast_1d(axes)
        im = axes[0].imshow(pred_depth_m, cmap="viridis")
        axes[0].set_title("Predicted depth (m)")
        axes[0].axis("off")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        if target_depth_m is not None:
            im = axes[1].imshow(target_depth_m, cmap="viridis")
            axes[1].set_title("Target depth (m)")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            im = axes[2].imshow(np.abs(pred_depth_m - target_depth_m), cmap="magma")
            axes[2].set_title("Absolute error (m)")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(output_fig, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to: {output_fig}")


if __name__ == "__main__":
    main()
