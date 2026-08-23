"""Self-supervised SPISR training with PUKL and equivariance losses."""

from __future__ import annotations

import argparse
import random

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    add_device_arguments,
    create_output_directory,
    fraction,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    random_seed,
    require_dataset_path,
    require_file,
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
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        metavar="FILE",
        help="Resume a compatible interrupted run from a safe .pt/.pth checkpoint.",
    )
    add_device_arguments(parser)
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
    if args.resume_checkpoint is not None:
        require_file(
            parser,
            args.resume_checkpoint,
            "--resume_checkpoint",
            suffixes=(".pt", ".pth"),
        )
    validate_output_directory(
        parser,
        args.output_dir,
        overwrite=args.overwrite or args.resume_checkpoint is not None,
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


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    train=True,
    *,
    epoch=0,
    global_step=0,
    start_batch=0,
    event_callback=None,
    cancel_check=None,
):
    import torch
    from tqdm import tqdm
    from spimaging.training_common.events import emit_event, raise_if_cancelled

    model.train(train)
    totals = {"loss": 0.0, "pukl": 0.0, "equivariance": 0.0}
    n_items = 0

    for batch_idx, batch in enumerate(
        tqdm(loader, desc="train" if train else "val", leave=False),
        start=1,
    ):
        if batch_idx <= int(start_batch):
            continue
        if train:
            raise_if_cancelled(
                cancel_check,
                phase="train",
                epoch=epoch,
                next_batch=batch_idx - 1,
                global_step=global_step,
            )
        measurement = batch["measurement"].to(device)
        with torch.set_grad_enabled(train):
            loss, metrics = self_supervised_losses(model, measurement, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1

        batch_size = measurement.shape[0]
        for key in totals:
            totals[key] += float(metrics[key].item()) * batch_size
        n_items += batch_size

        emit_event(
            "batch",
            callback=event_callback,
            phase="train" if train else "validation",
            epoch=int(epoch),
            batch=int(batch_idx),
            batches=int(len(loader)),
            global_step=int(global_step),
            loss=float(metrics["loss"].item()),
            pukl=float(metrics["pukl"].item()),
            equivariance=float(metrics["equivariance"].item()),
        )
        if train and batch_idx < len(loader):
            raise_if_cancelled(
                cancel_check,
                phase="train",
                epoch=epoch,
                next_batch=batch_idx,
                global_step=global_step,
            )

    denom = max(n_items, 1)
    result = {key: value / denom for key, value in totals.items()}
    result["global_step"] = global_step
    return result


def main(argv=None, *, event_callback=None, cancel_check=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from spimaging.training_common.dataset import SPISRSelfSupervisedDataset, list_sample_files
    from spimaging.training_common.device import get_torch_device
    from spimaging.training_common.events import CancellationRequested, emit_event, raise_if_cancelled
    from spimaging.training_common.networks import build_self_supervised_model
    from spimaging.training_common.recovery import (
        IncompatibleResumeError,
        append_training_history,
        build_resume_metadata,
        build_resume_signature,
        dataset_fingerprint,
        load_and_validate_resume,
        restore_checkpoint_state,
    )
    from spimaging.training_common.security import UnsafeArchiveError, UnsafeCheckpointError, load_spad_sample
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
        try:
            sample = load_spad_sample(sample_file, required_keys=("counts",))
            counts_shape = sample["counts"].shape
            if len(counts_shape) != 3:
                raise UnsafeArchiveError("field 'counts' must have shape (T,H,W) for training")
            if args.temporal_downsample > counts_shape[0]:
                raise UnsafeArchiveError(
                    f"--temporal_downsample {args.temporal_downsample} exceeds T={counts_shape[0]}"
                )
            if args.spatial_downsample > min(counts_shape[-2:]):
                raise UnsafeArchiveError(
                    f"--spatial_downsample {args.spatial_downsample} exceeds spatial shape {counts_shape[-2:]}"
                )
        except UnsafeArchiveError as exc:
            input_parser.error(f"cannot safely use --dataset_dir sample {sample_file}: {exc}")

    train_idx, val_idx = split_indices(len(files), args.val_fraction, args.seed)
    dataset = SPISRSelfSupervisedDataset(
        files,
        temporal_downsample=args.temporal_downsample,
        spatial_downsample=args.spatial_downsample,
        normalize=not args.no_normalize,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else None
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_set is not None
        else None
    )

    selection = get_torch_device(
        mode=args.device,
        gpu_index=args.gpu_index,
        return_selection=True,
    )
    device = selection.device
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
        overwrite=args.overwrite or args.resume_checkpoint is not None,
        option="--output_dir",
    )
    output_ready = False
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1
    start_batch = 0
    current_epoch = 1

    dataset_hash = dataset_fingerprint(files)
    signature = build_resume_signature(
        args,
        (
            "model",
            "base_channels",
            "num_blocks",
            "batch_size",
            "lr",
            "weight_decay",
            "val_fraction",
            "seed",
            "time_scale",
            "spatial_scale",
            "temporal_downsample",
            "spatial_downsample",
            "gamma",
            "tau",
            "alpha",
            "poisson_scale",
            "max_shift",
            "no_normalize",
        ),
    )
    if args.resume_checkpoint is not None:
        try:
            checkpoint = load_and_validate_resume(
                args.resume_checkpoint,
                map_location=device,
                dataset_hash=dataset_hash,
                signature=signature,
                requested_epochs=args.epochs,
            )
            metadata = restore_checkpoint_state(model, optimizer, checkpoint)
        except (IncompatibleResumeError, UnsafeCheckpointError) as exc:
            output_parser.error(f"cannot resume --resume_checkpoint {args.resume_checkpoint}: {exc}")
        start_epoch = int(metadata["next_epoch"])
        start_batch = int(metadata["next_batch"])
        global_step = int(metadata["global_step"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        if start_epoch > args.epochs:
            output_parser.error(
                "--resume_checkpoint has already reached --epochs; increase --epochs to continue"
            )

    emit_event(
        "device",
        callback=event_callback,
        requested=selection.requested,
        selected=str(device),
        gpu_index=selection.gpu_index,
        fallback=selection.fallback,
        reason=selection.reason,
    )
    if selection.fallback:
        emit_event("warning", callback=event_callback, code="device_fallback", message=selection.reason)

    print(f"Device: {device}")
    if selection.fallback:
        print(f"Device selection: {selection.reason}")
    print(f"Model: {args.model}")
    print("Method family: self-supervised SPISR")
    print(f"Samples: train={len(train_set)}, val={len(val_set) if val_set is not None else 0}")
    print("Loss: PUKL + alpha * equivariance KL")

    emit_event(
        "stage",
        callback=event_callback,
        stage="training",
        status="started",
        epochs=args.epochs,
        start_epoch=start_epoch,
        resumed=args.resume_checkpoint is not None,
    )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            generator = torch.Generator()
            generator.manual_seed(args.seed + epoch)
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                generator=generator,
            )
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args,
                train=True,
                epoch=epoch,
                global_step=global_step,
                start_batch=start_batch if epoch == start_epoch else 0,
                event_callback=event_callback,
                cancel_check=cancel_check,
            )
            global_step = int(train_metrics["global_step"])
            if val_loader is not None:
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    device,
                    args,
                    train=False,
                    epoch=epoch,
                    global_step=global_step,
                    event_callback=event_callback,
                )
            else:
                val_metrics = train_metrics
            raise_if_cancelled(
                cancel_check,
                phase="train",
                epoch=epoch + 1,
                next_batch=0,
                global_step=global_step,
            )

            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.5f} PUKL {train_metrics['pukl']:.5f} E {train_metrics['equivariance']:.5f} | "
                f"val loss {val_metrics['loss']:.5f} PUKL {val_metrics['pukl']:.5f} E {val_metrics['equivariance']:.5f}"
            )

            if not output_ready:
                create_output_directory(output_parser, output_dir, option="--output_dir")
                output_ready = True

            improved = val_metrics["loss"] <= best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
            resume = build_resume_metadata(
                dataset_hash=dataset_hash,
                signature=signature,
                target_epochs=args.epochs,
                next_epoch=epoch + 1,
                next_batch=0,
                global_step=global_step,
            )
            checkpoint_metrics = {
                "train": {key: value for key, value in train_metrics.items() if key != "global_step"},
                "validation": {key: value for key, value in val_metrics.items() if key != "global_step"},
            }
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
                resume=resume,
                metrics=checkpoint_metrics,
            )
            emit_event(
                "artifact",
                callback=event_callback,
                kind="checkpoint",
                path=str(output_dir / "last.pt"),
                epoch=epoch,
            )
            if improved:
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
                    resume=resume,
                    metrics=checkpoint_metrics,
                    status="best",
                )
            history_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_pukl": train_metrics["pukl"],
                "train_equivariance": train_metrics["equivariance"],
                "val_loss": val_metrics["loss"],
                "val_pukl": val_metrics["pukl"],
                "val_equivariance": val_metrics["equivariance"],
                "global_step": global_step,
            }
            append_training_history(output_dir, history_row)
            emit_event("epoch", callback=event_callback, **history_row)
    except CancellationRequested as exc:
        if not output_ready:
            create_output_directory(output_parser, output_dir, option="--output_dir")
            output_ready = True
        cancelled_epoch = int(exc.epoch if exc.epoch is not None else current_epoch)
        resume = build_resume_metadata(
            dataset_hash=dataset_hash,
            signature=signature,
            target_epochs=args.epochs,
            next_epoch=cancelled_epoch,
            next_batch=exc.next_batch,
            global_step=exc.global_step,
            phase=exc.phase or "train",
        )
        cancelled_path = output_dir / "cancelled.pt"
        save_training_checkpoint(
            cancelled_path,
            model,
            optimizer,
            max(cancelled_epoch - 1, 0),
            args,
            model_name=args.model,
            method_family="self_supervised_spisr",
            best_metric=best_val_loss,
            best_metric_name="best_val_loss",
            resume=resume,
            status="cancelled",
        )
        emit_event(
            "cancelled",
            callback=event_callback,
            phase=exc.phase or "train",
            checkpoint=str(cancelled_path),
            next_epoch=cancelled_epoch,
            next_batch=exc.next_batch,
            global_step=exc.global_step,
        )
        print(f"Cancelled safely. Resume checkpoint saved to: {cancelled_path}")
        raise SystemExit(130) from None

    print(f"Done. Best validation self-supervised loss: {best_val_loss:.5f}")
    print(f"Checkpoints saved to: {output_dir}")
    emit_event(
        "completed",
        callback=event_callback,
        best_val_loss=best_val_loss,
        output_dir=str(output_dir),
        global_step=global_step,
    )


if __name__ == "__main__":
    main()
