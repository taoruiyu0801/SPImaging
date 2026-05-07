"""Self-supervised SPISR training with PUKL and equivariance losses."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

MODEL_CHOICES = ("spisr",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a self-supervised SPISR model with PUKL and equivariance losses."
    )
    parser.add_argument("--dataset_dir", action="append", required=True, help="Directory containing generated .npz samples.")
    parser.add_argument("--output_dir", default="outputs/train_spisr_selfsup", help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="spisr")
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--time_scale", type=int, default=2, help="Longitudinal super-resolution factor.")
    parser.add_argument("--spatial_scale", type=int, default=2, help="Lateral super-resolution factor.")
    parser.add_argument("--temporal_downsample", type=int, default=8, help="Create LR inputs by summing time bins.")
    parser.add_argument("--spatial_downsample", type=int, default=2, help="Create LR inputs by summing spatial blocks.")
    parser.add_argument("--gamma", type=float, default=0.005, help="Poisson noise parameter in the PUKL estimator.")
    parser.add_argument("--tau", type=float, default=1e-3, help="Monte Carlo finite-difference scale for PUKL.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Equivariance loss weight.")
    parser.add_argument("--poisson_scale", type=float, default=1.0, help="Scale used before Poisson corruption in equivariance learning.")
    parser.add_argument("--max_shift", type=int, default=8, help="Maximum random longitudinal shift on the HR cube.")
    parser.add_argument("--no_normalize", action="store_true", help="Disable per-sample max normalization of LR counts.")
    return parser.parse_args()


def downsample_operator(hr_cube, lr_shape):
    import torch.nn.functional as F

    return F.interpolate(hr_cube, size=lr_shape, mode="trilinear", align_corners=False)


def random_equivariant_transform(cube, max_shift):
    import torch

    k = int(torch.randint(0, 4, (), device=cube.device).item())
    shift_limit = max(int(max_shift), 0)
    if shift_limit > 0:
        shift = int(torch.randint(-shift_limit, shift_limit + 1, (), device=cube.device).item())
    else:
        shift = 0
    transformed = torch.rot90(cube, k=k, dims=(-2, -1))
    transformed = torch.roll(transformed, shifts=shift, dims=2)
    return transformed


def poisson_corrupt(cube, scale):
    import torch

    scale = max(float(scale), 1e-8)
    rates = torch.clamp(cube / scale, min=0.0)
    return torch.poisson(rates) * scale


def self_supervised_losses(model, measurement, args):
    import torch
    import torch.nn.functional as F

    from spimaging.training_common.losses import positive_kl_loss, pukl_loss

    lr_shape = measurement.shape[2:]

    hr_1 = model(measurement)
    lr_1 = downsample_operator(hr_1, lr_shape)

    bernoulli = torch.empty_like(measurement).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    perturbed = torch.clamp(measurement + float(args.tau) * bernoulli, min=0.0)
    lr_perturbed = downsample_operator(model(perturbed), lr_shape)
    risk_estimate = lr_1 + (float(args.gamma) / float(args.tau)) * bernoulli * measurement * (lr_perturbed - lr_1)
    loss_pukl = pukl_loss(measurement, risk_estimate)

    hr_2 = random_equivariant_transform(hr_1.detach(), args.max_shift)
    lr_2 = downsample_operator(hr_2, lr_shape)
    lr_2_poisson = poisson_corrupt(lr_2, args.poisson_scale)
    hr_3 = model(lr_2_poisson)
    if hr_3.shape[2:] != hr_2.shape[2:]:
        hr_3 = F.interpolate(hr_3, size=hr_2.shape[2:], mode="trilinear", align_corners=False)
    loss_equivariance = positive_kl_loss(hr_3, hr_2)

    loss = loss_pukl + float(args.alpha) * loss_equivariance
    return loss, {
        "loss": loss.detach(),
        "pukl": loss_pukl.detach(),
        "equivariance": loss_equivariance.detach(),
    }


def run_epoch(model, loader, optimizer, device, args, train=True):
    import torch
    from tqdm import tqdm

    model.train(train)
    totals = {"loss": 0.0, "pukl": 0.0, "equivariance": 0.0}
    n_items = 0

    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        measurement = batch["measurement"].to(device)
        with torch.set_grad_enabled(train):
            loss, metrics = self_supervised_losses(model, measurement, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = measurement.shape[0]
        for key in totals:
            totals[key] += float(metrics[key].item()) * batch_size
        n_items += batch_size

    denom = max(n_items, 1)
    return {key: value / denom for key, value in totals.items()}


def main():
    args = parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from spimaging.training_common.dataset import SPISRSelfSupervisedDataset, list_sample_files
    from spimaging.training_common.device import get_torch_device
    from spimaging.training_common.networks import build_self_supervised_model
    from spimaging.training_common.utils import save_training_checkpoint, split_indices

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    files = list_sample_files(args.dataset_dir)
    if args.max_samples is not None:
        files = files[: args.max_samples]

    train_idx, val_idx = split_indices(len(files), args.val_fraction, args.seed)
    dataset = SPISRSelfSupervisedDataset(
        files,
        temporal_downsample=args.temporal_downsample,
        spatial_downsample=args.spatial_downsample,
        normalize=not args.no_normalize,
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
    model = build_self_supervised_model(
        args.model,
        in_channels=dataset.in_channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        time_scale=args.time_scale,
        spatial_scale=args.spatial_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print("Method family: self-supervised SPISR")
    print(f"Samples: train={len(train_set)}, val={len(val_set) if val_set is not None else 0}")
    print("Loss: PUKL + alpha * equivariance KL")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False)
        else:
            val_metrics = train_metrics

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.5f} PUKL {train_metrics['pukl']:.5f} E {train_metrics['equivariance']:.5f} | "
            f"val loss {val_metrics['loss']:.5f} PUKL {val_metrics['pukl']:.5f} E {val_metrics['equivariance']:.5f}"
        )

        save_training_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            args,
            model_name=args.model,
            method_family="self_supervised_spisr",
            best_metric=best_val_loss,
            best_metric_name="best_val_loss",
        )
        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_training_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                args,
                model_name=args.model,
                method_family="self_supervised_spisr",
                best_metric=best_val_loss,
                best_metric_name="best_val_loss",
            )

    print(f"Done. Best validation self-supervised loss: {best_val_loss:.5f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
