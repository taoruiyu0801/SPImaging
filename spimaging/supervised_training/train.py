"""Train a 3D CNN for SPAD single-photon depth reconstruction."""

from __future__ import annotations

import argparse
import random

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    add_device_arguments,
    create_output_directory,
    fraction,
    model_base_channels,
    model_num_blocks,
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

MODEL_CHOICES = ("simple3d", "prsnet", "penonlocal", "stin")
SUPERVISED_RESUME_SIGNATURE_FIELDS = (
    "model",
    "base_channels",
    "num_blocks",
    "batch_size",
    "lr",
    "weight_decay",
    "val_fraction",
    "seed",
    "temporal_downsample",
    "target_sigma_bins",
    "target_source",
    "no_log_counts",
    "tv_weight",
    "early_stopping_patience",
    "early_stopping_min_delta",
)


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


def build_parser():
    parser = ArgumentParser(
        prog="spad-train",
        description="Train a 3D CNN with KL loss for SPAD histogram depth reconstruction.",
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
        default="outputs/train_spad_3dcnn",
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
        default=1e-5,
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
        default="simple3d",
        help="3D network architecture.",
    )
    parser.add_argument(
        "--base_channels",
        type=model_base_channels,
        default=8,
        help="Number of base feature channels in the network (>= 1).",
    )
    parser.add_argument(
        "--num_blocks",
        type=model_num_blocks,
        default=10,
        help="Number of residual/non-local blocks for larger models (>= 1).",
    )
    parser.add_argument(
        "--temporal_downsample",
        type=positive_int,
        default=1,
        help="Factor for summing neighboring time bins before the 3D CNN (>= 1).",
    )
    parser.add_argument(
        "--target_sigma_bins",
        type=positive_float,
        default=2.0,
        help="Gaussian target width in downsampled time bins (> 0).",
    )
    parser.add_argument(
        "--target_source",
        choices=["depth", "clean"],
        default="depth",
        help="Training-target source: depth-derived Gaussian or transient_clean when available.",
    )
    parser.add_argument(
        "--no_log_counts",
        action="store_true",
        help="Disable log1p compression of counts before normalization.",
    )
    parser.add_argument(
        "--tv_weight",
        type=nonnegative_float,
        default=0.0,
        help="Weight for TV loss on the predicted depth map (>= 0).",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=positive_int,
        default=None,
        metavar="N",
        help="Stop after N validation epochs without sufficient MAE improvement; disabled when omitted.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=nonnegative_float,
        default=1e-4,
        help="Minimum validation-MAE improvement in meters (>= 0).",
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
    start_batch=0,
    event_callback=None,
    cancel_check=None,
):
    import torch
    from tqdm import tqdm

    from spimaging.training_common.losses import depth_tv_loss, temporal_kl_loss
    from spimaging.training_common.events import (
        emit_event,
        raise_if_cancelled,
        structured_events_enabled,
    )
    from spimaging.training_common.utils import match_distribution_shape

    model.train(train)
    total_loss = 0.0
    total_kl = 0.0
    total_tv = 0.0
    total_mae_m = 0.0
    n_items = 0

    for batch_idx, batch in enumerate(
        tqdm(
            loader,
            desc="train" if train else "val",
            leave=False,
            disable=structured_events_enabled(),
        ),
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

        emit_event(
            "batch",
            callback=event_callback,
            phase="train" if train else "validation",
            epoch=int(epoch),
            batch=int(batch_idx),
            batches=int(len(loader)),
            global_step=int(global_step),
            loss=float(loss.item()),
            kl=float(kl.item()),
            tv=float(tv.item()),
            mae_m=float(mae_m.item()),
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
    return {
        "loss": total_loss / denom,
        "kl": total_kl / denom,
        "tv": total_tv / denom,
        "mae_m": total_mae_m / denom,
        "global_step": global_step,
    }


def main(argv=None, *, event_callback=None, cancel_check=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from spimaging.training_common.dataset import SPADHistogramDataset, list_sample_files
    from spimaging.training_common.device import get_torch_device
    from spimaging.training_common.events import CancellationRequested, emit_event, raise_if_cancelled
    from spimaging.training_common.networks import build_model
    from spimaging.training_common.recovery import (
        IncompatibleResumeError,
        append_training_history,
        build_resume_metadata,
        build_resume_signature,
        dataset_fingerprint,
        load_and_validate_resume,
        restore_checkpoint_state,
    )
    from spimaging.training_common.security import UnsafeCheckpointError
    from spimaging.training_common.security import UnsafeArchiveError, load_spad_sample
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
            required_keys=("counts", "depth_m"),
        )
        try:
            sample = load_spad_sample(sample_file, required_keys=("counts", "depth_m"))
            counts_shape = sample["counts"].shape
            if len(counts_shape) != 3:
                raise UnsafeArchiveError("field 'counts' must have shape (T,H,W) for training")
            if args.temporal_downsample > counts_shape[0]:
                raise UnsafeArchiveError(
                    f"--temporal_downsample {args.temporal_downsample} exceeds T={counts_shape[0]}"
                )
        except UnsafeArchiveError as exc:
            input_parser.error(f"cannot safely use --dataset_dir sample {sample_file}: {exc}")

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
    model = build_model(
        args.model,
        in_channels=dataset.in_channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    n_kaiming_layers = apply_kaiming_initialization(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_parser = build_parser()
    output_dir = validate_output_directory(
        output_parser,
        args.output_dir,
        overwrite=args.overwrite or args.resume_checkpoint is not None,
        option="--output_dir",
    )
    output_ready = False
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    start_epoch = 1
    start_batch = 0
    current_epoch = 1

    dataset_hash = dataset_fingerprint(files)
    signature = build_resume_signature(
        args,
        SUPERVISED_RESUME_SIGNATURE_FIELDS,
    )

    if args.resume_checkpoint is not None:
        try:
            checkpoint = load_and_validate_resume(
                args.resume_checkpoint,
                # RNG snapshots must remain CPU tensors; load_state_dict moves
                # model and optimizer tensors to the selected device safely.
                map_location="cpu",
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
        best_val_mae = float(checkpoint.get("best_val_mae", float("inf")))
        epochs_without_improvement = int(metadata.get("epochs_without_improvement", 0))
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
    print(f"Initialization: Kaiming normal applied to {n_kaiming_layers} trainable conv/linear layers")
    print("Method family: supervised")
    print(f"Samples: train={len(train_set)}, val={len(val_set) if val_set is not None else 0}")
    print(f"Input shape per sample: (1,T,H,W), target shape: (T,H,W)")
    print(f"Loss: KL(target_time_distribution || predicted_time_distribution) + {args.tv_weight:g} * TV(predicted_depth)")
    print("Step logs are emitted after each optimizer update.")
    print(f"{'step':>8} {'epoch':>6} {'batch':>11} {'KL':>12} {'TV':>12}")
    print("-" * 55)

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
            epoch_start_batch = start_batch if epoch == start_epoch else 0
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
                start_batch=epoch_start_batch,
                event_callback=event_callback,
                cancel_check=cancel_check,
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

            if not output_ready:
                create_output_directory(output_parser, output_dir, option="--output_dir")
                output_ready = True

            improved = val_metrics["mae_m"] < best_val_mae - float(args.early_stopping_min_delta)
            if improved:
                best_val_mae = val_metrics["mae_m"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            resume = build_resume_metadata(
                dataset_hash=dataset_hash,
                signature=signature,
                target_epochs=args.epochs,
                next_epoch=epoch + 1,
                next_batch=0,
                global_step=global_step,
            )
            resume["epochs_without_improvement"] = epochs_without_improvement
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
                method_family="supervised",
                best_metric=best_val_mae,
                best_metric_name="best_val_mae",
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
                    method_family="supervised",
                    best_metric=best_val_mae,
                    best_metric_name="best_val_mae",
                    resume=resume,
                    metrics=checkpoint_metrics,
                    status="best",
                )

            history_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_kl": train_metrics["kl"],
                "train_tv": train_metrics["tv"],
                "train_mae_m": train_metrics["mae_m"],
                "val_loss": val_metrics["loss"],
                "val_kl": val_metrics["kl"],
                "val_tv": val_metrics["tv"],
                "val_mae_m": val_metrics["mae_m"],
                "global_step": global_step,
            }
            append_training_history(output_dir, history_row)
            emit_event("epoch", callback=event_callback, **history_row)

            if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping after {epoch} epochs: "
                    f"no validation MAE improvement >= {args.early_stopping_min_delta:g} m "
                    f"for {epochs_without_improvement} epochs."
                )
                break
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
        resume["epochs_without_improvement"] = epochs_without_improvement
        cancelled_path = output_dir / "cancelled.pt"
        save_training_checkpoint(
            cancelled_path,
            model,
            optimizer,
            max(cancelled_epoch - 1, 0),
            args,
            model_name=args.model,
            method_family="supervised",
            best_metric=best_val_mae,
            best_metric_name="best_val_mae",
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

    print(f"Done. Best validation MAE: {best_val_mae:.4f} m")
    print(f"Checkpoints saved to: {output_dir}")
    emit_event(
        "completed",
        callback=event_callback,
        best_val_mae=best_val_mae,
        output_dir=str(output_dir),
        global_step=global_step,
    )


if __name__ == "__main__":
    main()
