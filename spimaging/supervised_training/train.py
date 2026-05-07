"""Train a 3D CNN for SPAD single-photon depth reconstruction."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

MODEL_CHOICES = ("simple3d", "prsnet", "penonlocal", "stin")


def apply_kaiming_initialization(model):
    import torch.nn as nn

    initialized = 0
    layer_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
    )

    for module in model.modules():
        if isinstance(module, layer_types) and module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            initialized += 1

    return initialized


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D CNN with KL loss for SPAD histogram depth reconstruction.")
    parser.add_argument("--dataset_dir", action="append", required=True, help="Directory containing generated .npz samples. Can be passed multiple times.")
    parser.add_argument("--output_dir", default="outputs/train_spad_3dcnn", help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="simple3d", help="3D network architecture.")
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, default=10, help="Number of residual/non-local blocks for larger models.")
    parser.add_argument("--temporal_downsample", type=int, default=1, help="Sum neighboring time bins before feeding the 3D CNN.")
    parser.add_argument("--target_sigma_bins", type=float, default=2.0, help="Gaussian target width in downsampled time bins.")
    parser.add_argument("--target_source", choices=["depth", "clean"], default="depth", help="Use depth-derived Gaussian targets or transient_clean distributions when available.")
    parser.add_argument("--no_log_counts", action="store_true", help="Disable log1p compression of counts before normalization.")
    parser.add_argument("--tv_weight", type=float, default=0.0, help="Weight for TV loss on the predicted depth map.")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Stop after N validation epochs without MAE improvement.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Minimum validation MAE improvement in meters.")
    return parser.parse_args()


def expected_depth_from_logits(logits, bin_size, temporal_downsample):
    import torch

    from spimaging.training_common.dataset import LIGHT_SPEED_M_PER_S

    probs = torch.softmax(logits.squeeze(1), dim=1)
    t = probs.shape[1]
    bins = torch.arange(t, device=probs.device, dtype=probs.dtype).view(1, t, 1, 1)
    expected_bin = (probs * bins).sum(dim=1, keepdim=True)
    effective_bin_size = bin_size.view(-1, 1, 1, 1) * float(temporal_downsample)
    return expected_bin * effective_bin_size * LIGHT_SPEED_M_PER_S / 2.0


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    temporal_downsample,
    tv_weight=0.0,
    train=True,
    epoch=0,
    global_step=0,
):
    import torch
    from tqdm import tqdm

    from spimaging.training_common.losses import depth_tv_loss, temporal_kl_loss
    from spimaging.training_common.utils import match_distribution_shape

    model.train(train)
    total_loss = 0.0
    total_kl = 0.0
    total_tv = 0.0
    total_mae_m = 0.0
    n_items = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="train" if train else "val", leave=False), start=1):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        depth_m = batch["depth_m"].to(device)
        bin_size = batch["bin_size"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(inputs)
            logits = match_distribution_shape(logits, targets)
            kl = temporal_kl_loss(logits, targets)
            pred_depth_m_for_loss = expected_depth_from_logits(logits, bin_size, temporal_downsample)
            tv = depth_tv_loss(pred_depth_m_for_loss)
            loss = kl + float(tv_weight) * tv

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        pred_depth_m = expected_depth_from_logits(logits.detach(), bin_size, temporal_downsample)
        mae_m = torch.mean(torch.abs(pred_depth_m - depth_m))

        if train:
            global_step += 1
            tqdm.write(
                f"{global_step:8d} {epoch:6d} {batch_idx:5d}/{len(loader):5d} "
                f"{kl.item():12.6f} {tv.item():12.6f}"
            )

        batch_size = inputs.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_kl += float(kl.item()) * batch_size
        total_tv += float(tv.item()) * batch_size
        total_mae_m += float(mae_m.item()) * batch_size
        n_items += batch_size

    denom = max(n_items, 1)
    return {
        "loss": total_loss / denom,
        "kl": total_kl / denom,
        "tv": total_tv / denom,
        "mae_m": total_mae_m / denom,
        "global_step": global_step,
    }


def main():
    args = parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from spimaging.training_common.dataset import SPADHistogramDataset, list_sample_files
    from spimaging.training_common.device import get_torch_device
    from spimaging.training_common.networks import build_model
    from spimaging.training_common.utils import save_training_checkpoint, split_indices

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    files = list_sample_files(args.dataset_dir)
    if args.max_samples is not None:
        files = files[: args.max_samples]

    train_idx, val_idx = split_indices(len(files), args.val_fraction, args.seed)
    dataset = SPADHistogramDataset(
        files,
        temporal_downsample=args.temporal_downsample,
        target_sigma_bins=args.target_sigma_bins,
        target_source=args.target_source,
        use_log_counts=not args.no_log_counts,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_set is not None
        else None
    )

    device = get_torch_device()
    model = build_model(
        args.model,
        in_channels=dataset.in_channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    n_kaiming_layers = apply_kaiming_initialization(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Initialization: Kaiming normal applied to {n_kaiming_layers} trainable conv/linear layers")
    print("Method family: supervised")
    print(f"Samples: train={len(train_set)}, val={len(val_set) if val_set is not None else 0}")
    print(f"Input shape per sample: (1,T,H,W), target shape: (T,H,W)")
    print(f"Loss: KL(target_time_distribution || predicted_time_distribution) + {args.tv_weight:g} * TV(predicted_depth)")
    print("Step logs are emitted after each optimizer update.")
    print(f"{'step':>8} {'epoch':>6} {'batch':>11} {'KL':>12} {'TV':>12}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.temporal_downsample,
            args.tv_weight,
            train=True,
            epoch=epoch,
            global_step=global_step,
        )
        global_step = train_metrics["global_step"]

        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                args.temporal_downsample,
                args.tv_weight,
                train=False,
                epoch=epoch,
                global_step=global_step,
            )
        else:
            val_metrics = train_metrics

        print("-" * 55)
        print(f"Epoch {epoch:03d} metrics")
        print(
            f"  train | loss {train_metrics['loss']:.6f} | "
            f"KL {train_metrics['kl']:.6f} | TV {train_metrics['tv']:.6f} | "
            f"MAE {train_metrics['mae_m']:.6f} m"
        )
        print(
            f"  val   | loss {val_metrics['loss']:.6f} | "
            f"KL {val_metrics['kl']:.6f} | TV {val_metrics['tv']:.6f} | "
            f"MAE {val_metrics['mae_m']:.6f} m"
        )
        print("-" * 55)

        save_training_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            args,
            model_name=args.model,
            method_family="supervised",
            best_metric=best_val_mae,
            best_metric_name="best_val_mae",
        )
        improved = val_metrics["mae_m"] < best_val_mae - float(args.early_stopping_min_delta)
        if improved:
            best_val_mae = val_metrics["mae_m"]
            epochs_without_improvement = 0
            save_training_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                args,
                model_name=args.model,
                method_family="supervised",
                best_metric=best_val_mae,
                best_metric_name="best_val_mae",
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after {epoch} epochs: "
                f"no validation MAE improvement >= {args.early_stopping_min_delta:g} m "
                f"for {epochs_without_improvement} epochs."
            )
            break

    print(f"Done. Best validation MAE: {best_val_mae:.4f} m")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
