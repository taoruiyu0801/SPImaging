"""Run inference with a trained 3D SPAD histogram network."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict depth from one generated SPAD .npz sample.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--sample", required=True, help="Path to a generated sample_*.npz file.")
    parser.add_argument("--output_npz", default="outputs/prediction.npz", help="Where to save predicted depth.")
    parser.add_argument("--output_fig", default=None, help="Optional PNG path for visual comparison.")
    return parser.parse_args()


def load_model(checkpoint, device):
    import torch

    from spimaging.training_common.networks import build_model, build_self_supervised_model, canonical_model_name

    ckpt = torch.load(checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
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
    args = parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F

    from spimaging.training_common.dataset import SPADHistogramDataset, SPISRSelfSupervisedDataset
    from spimaging.training_common.device import get_torch_device

    device = get_torch_device()
    model, train_args, method_family = load_model(args.checkpoint, device)

    temporal_downsample = int(train_args.get("temporal_downsample", 1))
    if method_family == "self_supervised_spisr":
        dataset = SPISRSelfSupervisedDataset(
            [args.sample],
            temporal_downsample=temporal_downsample,
            spatial_downsample=int(train_args.get("spatial_downsample", 1)),
            normalize=not bool(train_args.get("no_normalize", False)),
        )
    else:
        dataset = SPADHistogramDataset(
            [args.sample],
            temporal_downsample=temporal_downsample,
            target_sigma_bins=float(train_args.get("target_sigma_bins", 2.0)),
            target_source=str(train_args.get("target_source", "depth")),
            use_log_counts=not bool(train_args.get("no_log_counts", False)),
        )
    item = dataset[0]
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
    raw = np.load(args.sample, allow_pickle=True)
    target_depth_m = raw["depth_m"].astype(np.float32) if "depth_m" in raw else None

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"pred_depth_m": pred_depth_m, "sample": str(args.sample), "method_family": method_family}
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

    if args.output_fig is not None:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str((Path(args.output_fig).parent / ".matplotlib-cache").resolve()))
        import matplotlib.pyplot as plt

        output_fig = Path(args.output_fig)
        output_fig.parent.mkdir(parents=True, exist_ok=True)
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
