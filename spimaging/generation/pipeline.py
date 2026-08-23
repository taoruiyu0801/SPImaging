import argparse
from pathlib import Path
import csv

import numpy as np
from tqdm import tqdm

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    add_device_arguments,
    create_output_directory,
    finite_float,
    nonnegative_float,
    nonnegative_int,
    parameter_index,
    positive_float,
    positive_int,
    positive_odd_int,
    positive_unit_interval,
    require_directory,
    require_file,
    unit_interval,
    validate_output_directory,
)
from spimaging.generation.datasets import (
    load_middlebury_pair,
    load_nyuv2_labeled,
    load_raw_pair,
    scan_middlebury_pairs,
    scan_nyuv2_raw_pairs,
)
from spimaging.generation.models import (
    build_deepinverse_signal,
    build_neighborhood_mixed_measurement,
    build_translucent_layer_measurement,
    build_volume_scattering_measurement,
    simulate_with_deepinverse,
)
from spimaging.generation.processing import (
    get_simulation_parameters,
    resize_rgb_depth,
    rgb_to_albedo_and_intensity,
)


def build_parser():
    parser = ArgumentParser(
        prog="spad-generate",
        description="Generate a SPAD-style dataset from NYUv2 or Middlebury. Supports single-surface, neighborhood-mixing multi-surface, translucent-layer, and volume-scattering simulation.",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="labeled",
        choices=["labeled", "raw", "middlebury"],
        help="Dataset mode: labeled (.mat), raw (scene folders), or middlebury.",
    )

    # -------------------------
    # Surface / measurement model
    # -------------------------
    parser.add_argument(
        "--surface_model",
        type=str,
        default="single",
        choices=["single", "neighborhood_mix", "translucent_layer", "volume_scattering"],
        help="single: original single-surface generation; neighborhood_mix: multi-return simulation by spatial neighborhood mixing; translucent_layer: add a semi-transparent front layer; volume_scattering: medium scattering such as fog or water.",
    )

    # neighborhood_mix parameters
    parser.add_argument("--mix_kernel_size", type=positive_odd_int, default=5, help="Positive odd kernel size for neighborhood mixing, e.g. 3, 5, 7.")
    parser.add_argument("--mix_sigma_xy", type=positive_float, default=1.0, help="Positive spatial Gaussian sigma in pixels for neighborhood mixing.")
    parser.add_argument("--mix_time_sigma_bins", type=positive_float, default=2.0, help="Positive temporal Gaussian sigma in bins for neighborhood mixing.")

    # translucent_layer parameters
    parser.add_argument("--translucent_front_type", type=str, default="flat", choices=["flat", "sloped", "sinusoidal"], help="Front semi-transparent layer geometry.")
    parser.add_argument("--translucent_front_depth", type=positive_float, default=1.0, help="Positive base front-layer depth in meters.")
    parser.add_argument("--translucent_front_depth_x_slope", type=finite_float, default=0.0, help="Finite depth slope along x in meters across normalized image width.")
    parser.add_argument("--translucent_front_depth_y_slope", type=finite_float, default=0.0, help="Finite depth slope along y in meters across normalized image height.")
    parser.add_argument("--translucent_front_depth_amplitude", type=nonnegative_float, default=0.1, help="Nonnegative amplitude in meters for sinusoidal depth variation.")
    parser.add_argument("--translucent_front_signal_ratio", type=nonnegative_float, default=0.25, help="Nonnegative front-layer reflection strength relative to the back scene.")
    parser.add_argument("--translucent_transmission", type=unit_interval, default=0.6, help="Back-scene transmission factor between 0 and 1.")
    parser.add_argument("--translucent_time_sigma_bins", type=positive_float, default=2.0, help="Positive temporal Gaussian sigma in bins for the translucent-layer mode.")

    # volume_scattering parameters
    parser.add_argument("--volume_medium_type", type=str, default="fog", choices=["fog", "water"], help="Scattering medium type.")
    parser.add_argument("--volume_extinction_coeff", type=nonnegative_float, default=None, help="Optional nonnegative extinction coefficient in 1/m; omit to use the medium default.")
    parser.add_argument("--volume_backscatter_ratio", type=nonnegative_float, default=None, help="Optional nonnegative integrated-backscatter strength; omit to use the medium default.")
    parser.add_argument("--volume_scatter_depth_fraction", type=positive_unit_interval, default=0.9, help="Target-depth fraction used for scattering accumulation, in (0, 1].")
    parser.add_argument("--volume_num_steps", type=positive_int, default=64, help="Positive number of range samples used to approximate path scattering.")
    parser.add_argument("--volume_time_sigma_bins", type=positive_float, default=2.0, help="Positive temporal Gaussian sigma in bins for volume scattering.")
    parser.add_argument("--volume_range_weight_power", type=nonnegative_float, default=1.0, help="Nonnegative depth-weight exponent for scatter accumulation.")
    parser.add_argument("--volume_water_front_boost", type=nonnegative_float, default=1.5, help="Nonnegative near-range scattering multiplier for water.")
    parser.add_argument("--volume_fog_front_boost", type=nonnegative_float, default=1.0, help="Nonnegative near-range scattering multiplier for fog.")

    parser.add_argument(
        "--save_clean_transient",
        action="store_true",
        help="Save the clean transient before Poisson sampling in multi-surface modes.",
    )

    # -------------------------
    # NYUv2 labeled
    # -------------------------
    parser.add_argument(
        "--nyu_mat",
        type=str,
        default="NYUv2/nyu_depth_v2_labeled.mat",
        help="Path to nyu_depth_v2_labeled.mat (used when --dataset_mode labeled).",
    )

    # -------------------------
    # NYUv2 raw
    # -------------------------
    parser.add_argument(
        "--raw_root",
        type=str,
        default="NYUv2/raw",
        help="Root directory of NYUv2 raw dataset (used when --dataset_mode raw).",
    )
    parser.add_argument("--raw_max_time_diff", type=nonnegative_float, default=0.05, help="Nonnegative maximum timestamp difference in seconds for RGB/depth matching.")
    parser.add_argument("--raw_stride", type=positive_int, default=1, help="Process every k-th matched raw frame, where k is positive.")
    parser.add_argument("--drop_first", type=nonnegative_int, default=5, help="Nonnegative number of matched frames to drop from each sequence start.")
    parser.add_argument("--drop_last", type=nonnegative_int, default=5, help="Nonnegative number of matched frames to drop from each sequence end.")

    # -------------------------
    # Middlebury
    # -------------------------
    parser.add_argument(
        "--middlebury_root",
        type=str,
        default="middlebury/raw",
        help="Root directory of the Middlebury dataset (used when --dataset_mode middlebury).",
    )
    parser.add_argument("--middlebury_depth_min", type=nonnegative_float, default=0.5, help="Nonnegative minimum pseudo depth in meters.")
    parser.add_argument("--middlebury_depth_max", type=positive_float, default=5.0, help="Positive maximum pseudo depth in meters; must exceed the minimum.")

    # -------------------------
    # Output / simulation
    # -------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/spad_dataset",
        help="Directory to save generated .npz samples",
    )
    parser.add_argument("--param_idx", type=parameter_index, default=10, help="Photon simulation parameter index from 1 to 10.")
    parser.add_argument("--res", type=positive_int, default=64, help="Positive square output resolution in pixels.")
    parser.add_argument("--bins", type=positive_int, default=1024, help="Positive number of temporal bins.")
    parser.add_argument("--bin_size", type=positive_float, default=80e-12, help="Positive temporal bin size in seconds.")
    parser.add_argument("--irf_sigma", type=positive_float, default=2.0, help="Positive IRF Gaussian sigma in bins for single-surface simulation.")
    parser.add_argument("--limit", type=positive_int, default=None, help="Optional positive cap on the number of processed samples.")
    parser.add_argument("--save_x", action="store_true", help="Also save x=(depth-like, signal-like, background) proxy for visualization/debugging.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow publishing into a nonempty output directory; after a successful run, "
            "stale sample_*.npz files and index.csv from earlier runs are removed."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible incomplete generation from its integrity-checked partial manifest.",
    )
    add_device_arguments(parser)

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_args(parser, args):
    if args.middlebury_depth_max <= args.middlebury_depth_min:
        parser.error("--middlebury_depth_max must be greater than --middlebury_depth_min")

    if args.dataset_mode == "labeled":
        require_file(parser, args.nyu_mat, "--nyu_mat", suffixes=(".mat",))
    elif args.dataset_mode == "raw":
        require_directory(parser, args.raw_root, "--raw_root")
    else:
        require_directory(parser, args.middlebury_root, "--middlebury_root")

    return validate_output_directory(parser, args.output_dir, overwrite=args.overwrite)


def validate_source_image_signatures(parser, pairs, dataset_mode):
    path_keys = ("rgb_path", "depth_path") if dataset_mode == "raw" else ("rgb_path", "disp_path")
    signatures = {
        ".png": (b"\x89PNG\r\n\x1a\n",),
        ".ppm": (b"P3", b"P6"),
        ".pgm": (b"P2", b"P5"),
    }
    for item in pairs:
        for key in path_keys:
            path = Path(item[key])
            expected = signatures.get(path.suffix.lower())
            if expected is None:
                continue
            try:
                with path.open("rb") as handle:
                    header = handle.read(8)
            except OSError as exc:
                parser.error(f"cannot read input image {path}: {exc}")
            if not any(header.startswith(signature) for signature in expected):
                parser.error(f"input image has an invalid {path.suffix} signature: {path}")


def publish_generated_output(parser, staging_dir, output_dir, overwrite):
    create_output_directory(parser, output_dir)
    existing_owned = list(output_dir.glob("sample_*.npz"))
    index_path = output_dir / "index.csv"
    if index_path.exists():
        existing_owned.append(index_path)
    generation_manifest = output_dir / "generation_manifest.json"
    if generation_manifest.exists():
        existing_owned.append(generation_manifest)

    produced = {path.name for path in staging_dir.iterdir() if path.is_file()}
    try:
        for source in staging_dir.iterdir():
            if source.is_file():
                source.replace(output_dir / source.name)
        if overwrite:
            for stale in existing_owned:
                if stale.name not in produced and stale.is_file():
                    stale.unlink()
    except OSError as exc:
        parser.error(f"cannot publish generated files to --output_dir {output_dir}: {exc}")


# =========================================================
# Sample generation wrapper
# =========================================================
def generate_one_sample(args, rgb, depth, mean_signal_photons, mean_background_photons, sbr):
    rgb_resized, depth_resized = resize_rgb_depth(rgb, depth, args.res)
    albedo, intensity = rgb_to_albedo_and_intensity(rgb_resized)

    extra_dict = {}
    transient_clean = None

    if args.surface_model == "single":
        x = build_deepinverse_signal(
            depth=depth_resized,
            albedo=albedo,
            intensity=intensity,
            bins=args.bins,
            bin_size=args.bin_size,
            mean_signal_photons=mean_signal_photons,
            sbr=sbr,
        )
        counts, xhat = simulate_with_deepinverse(
            x,
            bins=args.bins,
            irf_sigma=args.irf_sigma,
            device=getattr(args, "_torch_device", None),
        )
        x_to_save = x if args.save_x else None

    elif args.surface_model == "neighborhood_mix":
        counts, xhat, x_proxy, transient_clean = build_neighborhood_mixed_measurement(
            depth=depth_resized,
            albedo=albedo,
            intensity=intensity,
            bins=args.bins,
            bin_size=args.bin_size,
            mean_signal_photons=mean_signal_photons,
            sbr=sbr,
            mix_kernel_size=args.mix_kernel_size,
            mix_sigma_xy=args.mix_sigma_xy,
            mix_time_sigma_bins=args.mix_time_sigma_bins,
        )
        x_to_save = x_proxy if args.save_x else None

    elif args.surface_model == "translucent_layer":
        counts, xhat, x_proxy, transient_clean, extra_dict = build_translucent_layer_measurement(
            depth=depth_resized,
            albedo=albedo,
            intensity=intensity,
            bins=args.bins,
            bin_size=args.bin_size,
            mean_signal_photons=mean_signal_photons,
            sbr=sbr,
            front_type=args.translucent_front_type,
            front_depth=args.translucent_front_depth,
            front_depth_x_slope=args.translucent_front_depth_x_slope,
            front_depth_y_slope=args.translucent_front_depth_y_slope,
            front_depth_amplitude=args.translucent_front_depth_amplitude,
            front_signal_ratio=args.translucent_front_signal_ratio,
            transmission=args.translucent_transmission,
            time_sigma_bins=args.translucent_time_sigma_bins,
        )
        x_to_save = x_proxy if args.save_x else None

    else:  # volume_scattering
        counts, xhat, x_proxy, transient_clean, extra_dict = build_volume_scattering_measurement(
            depth=depth_resized,
            albedo=albedo,
            intensity=intensity,
            bins=args.bins,
            bin_size=args.bin_size,
            mean_signal_photons=mean_signal_photons,
            sbr=sbr,
            medium_type=args.volume_medium_type,
            extinction_coeff=args.volume_extinction_coeff,
            backscatter_ratio=args.volume_backscatter_ratio,
            scatter_depth_fraction=args.volume_scatter_depth_fraction,
            num_steps=args.volume_num_steps,
            time_sigma_bins=args.volume_time_sigma_bins,
            range_weight_power=args.volume_range_weight_power,
            water_front_boost=args.volume_water_front_boost,
            fog_front_boost=args.volume_fog_front_boost,
        )
        x_to_save = x_proxy if args.save_x else None
        extra_dict["volume_medium_type_id"] = np.float32(0.0 if args.volume_medium_type == "fog" else 1.0)

    result = {
        "counts": counts.astype(np.float32),
        "depth_m": depth_resized.astype(np.float32),
        "rgb": rgb_resized.astype(np.float32),
        "albedo": albedo.astype(np.float32),
        "intensity": intensity.astype(np.float32),
        "xhat": xhat.astype(np.float32),
        "mean_signal_photons": np.float32(mean_signal_photons),
        "mean_background_photons": np.float32(mean_background_photons),
        "sbr": np.float32(sbr),
        "bins": np.int32(args.bins),
        "bin_size": np.float32(args.bin_size),
        "irf_sigma": np.float32(args.irf_sigma),
        "surface_model": args.surface_model,
    }

    if x_to_save is not None:
        result["x"] = x_to_save.astype(np.float32)
    if transient_clean is not None and args.save_clean_transient:
        result["transient_clean"] = transient_clean.astype(np.float32)
    for k, v in extra_dict.items():
        result[k] = v.astype(np.float32)

    return result


# =========================================================
# Main
# =========================================================
def main(argv=None, *, event_callback=None, cancel_check=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    final_output_dir = validate_args(parser, args)

    if args.surface_model == "single":
        try:
            from spimaging.generation.deepinverse import import_deepinv

            import_deepinv()
        except ImportError:
            parser.error("--surface_model single requires DeepInverse; install it with: pip install deepinv")

    images = None
    depths = None
    pairs = None
    if args.dataset_mode == "labeled":
        try:
            images, depths = load_nyuv2_labeled(Path(args.nyu_mat))
        except (KeyError, OSError, ValueError) as exc:
            parser.error(f"cannot read --nyu_mat {args.nyu_mat}: {exc}")
    elif args.dataset_mode == "raw":
        try:
            pairs = scan_nyuv2_raw_pairs(
                raw_root=Path(args.raw_root),
                max_time_diff=args.raw_max_time_diff,
                stride=args.raw_stride,
                drop_first=args.drop_first,
                drop_last=args.drop_last,
                verbose=True,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot scan --raw_root {args.raw_root}: {exc}")
        if not pairs:
            parser.error(
                "no matched raw RGB-depth pairs found; check --raw_root, "
                "--raw_max_time_diff, --drop_first, and --drop_last"
            )
    else:
        try:
            pairs = scan_middlebury_pairs(middlebury_root=Path(args.middlebury_root), verbose=True)
        except (OSError, ValueError) as exc:
            parser.error(f"cannot scan --middlebury_root {args.middlebury_root}: {exc}")
        if not pairs:
            parser.error("no valid Middlebury RGB-disparity pairs found; check --middlebury_root")

    if pairs is not None:
        if args.limit is not None:
            pairs = pairs[:args.limit]
        validate_source_image_signatures(parser, pairs, args.dataset_mode)

    from spimaging.generation.recovery import (
        GenerationResumeError,
        cleanup_completed_session,
        generation_config_fingerprint,
        prepare_generation_session,
        source_fingerprint,
    )
    from spimaging.training_common.events import cancellation_requested, emit_event

    if args.dataset_mode == "labeled":
        total_candidates = int(images.shape[0])
        if args.limit is not None:
            total_candidates = min(total_candidates, args.limit)
        source_paths = [Path(args.nyu_mat)]
    else:
        total_candidates = len(pairs)
        second_key = "depth_path" if args.dataset_mode == "raw" else "disp_path"
        source_paths = [path for item in pairs for path in (item["rgb_path"], item[second_key])]

    create_output_directory(parser, final_output_dir.parent, option="parent of --output_dir")
    try:
        session = prepare_generation_session(
            final_output_dir,
            config_hash=generation_config_fingerprint(args),
            source_hash=source_fingerprint(source_paths),
            total_candidates=total_candidates,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except (GenerationResumeError, OSError) as exc:
        parser.error(f"cannot prepare resumable generation for --output_dir {final_output_dir}: {exc}")
    output_dir = session.directory

    selection = None
    if args.surface_model == "single":
        from spimaging.training_common.device import get_torch_device

        selection = get_torch_device(
            mode=args.device,
            gpu_index=args.gpu_index,
            return_selection=True,
        )
        args._torch_device = selection.device
        emit_event(
            "device",
            callback=event_callback,
            requested=selection.requested,
            selected=str(selection.device),
            gpu_index=selection.gpu_index,
            fallback=selection.fallback,
            reason=selection.reason,
        )
        if selection.fallback:
            emit_event("warning", callback=event_callback, code="device_fallback", message=selection.reason)

    index_rows = session.index_rows
    completed_ids = session.completed_ids
    emit_event(
        "stage",
        callback=event_callback,
        stage="generation",
        status="started",
        total=total_candidates,
        completed=len(completed_ids),
        resumed=args.resume,
    )

    def check_cancelled(next_sample):
        if cancellation_requested(cancel_check):
            session.mark_interrupted("cancelled")
            emit_event(
                "cancelled",
                callback=event_callback,
                phase="generation",
                completed=len(session.completed_ids),
                total=total_candidates,
                next_sample=next_sample,
                partial_manifest=str(output_dir / ".generation_partial.json"),
            )
            print(f"Cancelled safely. Resume with --resume; partial data is in: {output_dir}")
            raise SystemExit(130)

    if args.dataset_mode == "labeled":
        n_samples = images.shape[0]
        if args.limit is not None:
            n_samples = min(n_samples, args.limit)

        for idx in tqdm(range(n_samples), desc=f"Generating SPAD dataset from labeled NYUv2 ({args.surface_model})"):
            check_cancelled(idx)
            if idx in completed_ids:
                continue
            rgb = images[idx]
            depth = depths[idx]
            mean_signal_photons, mean_background_photons, sbr = get_simulation_parameters(args.param_idx)

            save_dict = generate_one_sample(
                args=args,
                rgb=rgb,
                depth=depth,
                mean_signal_photons=mean_signal_photons,
                mean_background_photons=mean_background_photons,
                sbr=sbr,
            )
            save_dict["source_mode"] = "labeled"
            save_dict["sample_id"] = np.int32(idx)

            sample_path = output_dir / f"sample_{idx:05d}.npz"
            np.savez_compressed(sample_path, **save_dict)

            row = {
                "sample_id": idx,
                "file": sample_path.name,
                "source_mode": "labeled",
                "scene": "",
                "rgb_file": "",
                "depth_file": "",
                "time_diff": "",
                "mean_signal_photons": mean_signal_photons,
                "mean_background_photons": mean_background_photons,
                "sbr": sbr,
                "surface_model": args.surface_model,
            }
            index_rows.append(row)
            session.record_sample(idx, sample_path, row)
            completed_ids.add(idx)
            emit_event(
                "sample",
                callback=event_callback,
                phase="generation",
                sample_id=idx,
                completed=len(completed_ids),
                total=total_candidates,
                artifact=str(sample_path),
            )

    elif args.dataset_mode == "raw":
        for idx, item in enumerate(tqdm(pairs, desc=f"Generating SPAD dataset from raw NYUv2 ({args.surface_model})")):
            check_cancelled(idx)
            if idx in completed_ids:
                continue
            try:
                rgb, depth = load_raw_pair(item["rgb_path"], item["depth_path"])
            except (OSError, ValueError) as exc:
                parser.error(f"cannot read raw RGB-depth pair for scene {item['scene']}: {exc}")
            valid = depth > 0.05
            if valid.sum() == 0:
                continue
            mean_signal_photons, mean_background_photons, sbr = get_simulation_parameters(args.param_idx)

            save_dict = generate_one_sample(
                args=args,
                rgb=rgb,
                depth=depth,
                mean_signal_photons=mean_signal_photons,
                mean_background_photons=mean_background_photons,
                sbr=sbr,
            )

            save_dict["source_mode"] = "raw"
            save_dict["scene"] = item["scene"]
            save_dict["rgb_file"] = str(item["rgb_path"])
            save_dict["depth_file"] = str(item["depth_path"])
            save_dict["rgb_ts"] = np.float64(item["rgb_ts"])
            save_dict["depth_ts"] = np.float64(item["depth_ts"])
            save_dict["time_diff"] = np.float64(item["dt"])
            save_dict["sample_id"] = np.int32(idx)

            sample_path = output_dir / f"sample_{idx:05d}.npz"
            np.savez_compressed(sample_path, **save_dict)

            row = {
                "sample_id": idx,
                "file": sample_path.name,
                "source_mode": "raw",
                "scene": item["scene"],
                "rgb_file": item["rgb_path"].name,
                "depth_file": item["depth_path"].name,
                "time_diff": item["dt"],
                "mean_signal_photons": mean_signal_photons,
                "mean_background_photons": mean_background_photons,
                "sbr": sbr,
                "surface_model": args.surface_model,
            }
            index_rows.append(row)
            session.record_sample(idx, sample_path, row)
            completed_ids.add(idx)
            emit_event(
                "sample",
                callback=event_callback,
                phase="generation",
                sample_id=idx,
                completed=len(completed_ids),
                total=total_candidates,
                artifact=str(sample_path),
            )

    else:  # middlebury
        for idx, item in enumerate(tqdm(pairs, desc=f"Generating SPAD dataset from Middlebury ({args.surface_model})")):
            check_cancelled(idx)
            if idx in completed_ids:
                continue
            try:
                rgb, depth = load_middlebury_pair(
                    item["rgb_path"],
                    item["disp_path"],
                    depth_min=args.middlebury_depth_min,
                    depth_max=args.middlebury_depth_max,
                )
            except (OSError, ValueError) as exc:
                parser.error(f"cannot read Middlebury pair for scene {item['scene']}: {exc}")
            mean_signal_photons, mean_background_photons, sbr = get_simulation_parameters(args.param_idx)

            save_dict = generate_one_sample(
                args=args,
                rgb=rgb,
                depth=depth,
                mean_signal_photons=mean_signal_photons,
                mean_background_photons=mean_background_photons,
                sbr=sbr,
            )

            save_dict["source_mode"] = "middlebury"
            save_dict["scene"] = item["scene"]
            save_dict["rgb_file"] = str(item["rgb_path"])
            save_dict["depth_file"] = str(item["disp_path"])
            save_dict["sample_id"] = np.int32(idx)

            sample_path = output_dir / f"sample_{idx:05d}.npz"
            np.savez_compressed(sample_path, **save_dict)

            row = {
                "sample_id": idx,
                "file": sample_path.name,
                "source_mode": "middlebury",
                "scene": item["scene"],
                "rgb_file": item["rgb_path"].name,
                "depth_file": item["disp_path"].name,
                "time_diff": "",
                "mean_signal_photons": mean_signal_photons,
                "mean_background_photons": mean_background_photons,
                "sbr": sbr,
                "surface_model": args.surface_model,
            }
            index_rows.append(row)
            session.record_sample(idx, sample_path, row)
            completed_ids.add(idx)
            emit_event(
                "sample",
                callback=event_callback,
                phase="generation",
                sample_id=idx,
                completed=len(completed_ids),
                total=total_candidates,
                artifact=str(sample_path),
            )

    index_rows.sort(key=lambda row: int(row["sample_id"]))
    with open(output_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "file",
                "source_mode",
                "scene",
                "rgb_file",
                "depth_file",
                "time_diff",
                "mean_signal_photons",
                "mean_background_photons",
                "sbr",
                "surface_model",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    session.complete()
    publish_generated_output(parser, output_dir, final_output_dir, args.overwrite)
    try:
        cleanup_completed_session(session)
    except GenerationResumeError as exc:
        parser.error(f"generated output was published but staging cleanup failed: {exc}")
    if selection is not None:
        print(f"Device: {selection.device}")
        if selection.fallback:
            print(f"Device selection: {selection.reason}")
    print("Done.")
    print(f"Saved {len(index_rows)} samples to: {final_output_dir}")
    emit_event(
        "completed",
        callback=event_callback,
        phase="generation",
        samples=len(index_rows),
        output_dir=str(final_output_dir),
    )


if __name__ == "__main__":
    main()
