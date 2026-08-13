"""Self-supervised SPISR training with PUKL and equivariance losses."""

from __future__ import annotations

import argparse
import random

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    create_output_directory,
    fraction,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    random_seed,
    require_dataset_path,
    validate_npz_archive,
    validate_output_directory,
)

MODEL_CHOICES = ("spisr",)


def build_parser():
    parser = ArgumentParser(
        prog="spad-train-selfsup",
        description="Train a self-supervised SPISR model with PUKL and equivariance losses.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        action="append",
        required=True,
        metavar="PATH",
        help="Existing dataset directory or .npz sample file; repeat to combine multiple inputs.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/train_spisr_selfsup",
        metavar="DIR",
        help="Directory in which checkpoints are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument("--epochs", type=positive_int, default=20, help="Number of training epochs (>= 1).")
    parser.add_argument("--batch_size", type=positive_int, default=1, help="Samples per optimizer step (>= 1).")
    parser.add_argument("--lr", type=positive_float, default=1e-3, help="AdamW learning rate (> 0).")
    parser.add_argument(
        "--weight_decay",
        type=nonnegative_float,
        default=1e-6,
        help="AdamW weight-decay coefficient (>= 0).",
    )
    parser.add_argument(
        "--val_fraction",
        type=fraction,
        default=0.2,
        help="Fraction of samples reserved for validation (0 <= value < 1).",
    )
    parser.add_argument(
        "--seed",
        type=random_seed,
        default=0,
        help="Random seed between 0 and 4294967295.",
    )
    parser.add_argument(
        "--num_workers",
        type=nonnegative_int,
        default=0,
        help="Number of DataLoader worker processes (>= 0).",
    )
    parser.add_argument(
        "--max_samples",
        type=positive_int,
        default=None,
        metavar="N",
        help="Optional positive cap on the number of samples, useful for quick tests.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="spisr",
        help="Self-supervised reconstruction architecture.",
    )
    parser.add_argument(
        "--base_channels",
        type=positive_int,
        default=16,
        help="Number of base feature channels in the network (>= 1).",
    )
    parser.add_argument(
        "--num_blocks",
        type=positive_int,
        default=4,
        help="Number of residual processing blocks (>= 1).",
    )
    parser.add_argument(
        "--time_scale",
        type=positive_int,
        default=2,
        help="Longitudinal super-resolution factor (>= 1).",
    )
    parser.add_argument(
        "--spatial_scale",
        type=positive_int,
        default=2,
        help="Lateral super-resolution factor (>= 1).",
    )
    parser.add_argument(
        "--temporal_downsample",
        type=positive_int,
        default=8,
        help="Factor for summing time bins when creating LR inputs (>= 1).",
    )
    parser.add_argument(
        "--spatial_downsample",
        type=positive_int,
        default=2,
        help="Factor for summing spatial blocks when creating LR inputs (>= 1).",
    )
    parser.add_argument(
        "--gamma",
        type=positive_float,
        default=0.005,
        help="Poisson-noise parameter in the PUKL estimator (> 0).",
    )
    parser.add_argument(
        "--tau",
        type=positive_float,
        default=1e-3,
        help="Monte Carlo finite-difference scale for PUKL (> 0).",
    )
    parser.add_argument(
        "--alpha",
        type=nonnegative_float,
        default=1.0,
        help="Equivariance-loss weight (>= 0).",
    )
    parser.add_argument(
        "--poisson_scale",
        type=positive_float,
        default=1.0,
        help="Scale applied before Poisson corruption in equivariance learning (> 0).",
    )
    parser.add_argument(
        "--max_shift",
        type=nonnegative_int,
        default=8,
        help="Maximum absolute random longitudinal shift on the HR cube (>= 0).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable per-sample maximum normalization of LR counts.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    for value in args.dataset_dir:
        require_dataset_path(parser, value, "--dataset_dir")
    validate_output_directory(
        parser,
        args.output_dir,
        overwrite=args.overwrite,
        option="--output_dir",
    )
    return args


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

    try:
        files = list_sample_files(args.dataset_dir)
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))
    if args.max_samples is not None:
        files = files[: args.max_samples]
    input_parser = build_parser()
    for sample_file in files:
        validate_npz_archive(
            input_parser,
            sample_file,
            "--dataset_dir sample",
            required_keys=("counts",),
        )

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

    output_parser = build_parser()
    output_dir = validate_output_directory(
        output_parser,
        args.output_dir,
        overwrite=args.overwrite,
        option="--output_dir",
    )
    output_ready = False
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

        if not output_ready:
            create_output_directory(output_parser, output_dir, option="--output_dir")
            output_ready = True

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
