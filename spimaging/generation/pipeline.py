import argparse
from pathlib import Path
import csv

import numpy as np
from tqdm import tqdm

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a SPAD-style dataset from NYUv2 or Middlebury. Supports single-surface, neighborhood-mixing multi-surface, translucent-layer, and volume-scattering simulation."
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
    parser.add_argument("--mix_kernel_size", type=int, default=5, help="Odd kernel size for neighborhood mixing, e.g. 3, 5, 7.")
    parser.add_argument("--mix_sigma_xy", type=float, default=1.0, help="Spatial Gaussian sigma (in pixels) for neighborhood mixing.")
    parser.add_argument("--mix_time_sigma_bins", type=float, default=2.0, help="Temporal Gaussian sigma in bins for pulse broadening in neighborhood mixing mode.")

    # translucent_layer parameters
    parser.add_argument("--translucent_front_type", type=str, default="flat", choices=["flat", "sloped", "sinusoidal"], help="Front semi-transparent layer geometry.")
    parser.add_argument("--translucent_front_depth", type=float, default=1.0, help="Base front-layer depth in meters.")
    parser.add_argument("--translucent_front_depth_x_slope", type=float, default=0.0, help="Depth slope along x for sloped front layer (meters across normalized image width).")
    parser.add_argument("--translucent_front_depth_y_slope", type=float, default=0.0, help="Depth slope along y for sloped front layer (meters across normalized image height).")
    parser.add_argument("--translucent_front_depth_amplitude", type=float, default=0.1, help="Amplitude in meters for sinusoidal front-layer depth variation.")
    parser.add_argument("--translucent_front_signal_ratio", type=float, default=0.25, help="Relative strength of the front-layer reflection compared with the normalized back-scene signal level.")
    parser.add_argument("--translucent_transmission", type=float, default=0.6, help="Transmission factor applied to the back scene after passing through the front translucent layer.")
    parser.add_argument("--translucent_time_sigma_bins", type=float, default=2.0, help="Temporal Gaussian sigma in bins for the translucent-layer mode.")

    # volume_scattering parameters
    parser.add_argument("--volume_medium_type", type=str, default="fog", choices=["fog", "water"], help="Scattering medium type.")
    parser.add_argument("--volume_extinction_coeff", type=float, default=None, help="Extinction coefficient (1/m). If not set, use defaults for fog/water.")
    parser.add_argument("--volume_backscatter_ratio", type=float, default=None, help="Relative strength of the integrated backscatter compared with nominal signal. If not set, use defaults for fog/water.")
    parser.add_argument("--volume_scatter_depth_fraction", type=float, default=0.9, help="Scattering is accumulated from the sensor to this fraction of target depth.")
    parser.add_argument("--volume_num_steps", type=int, default=64, help="Number of discrete range samples used to approximate path scattering.")
    parser.add_argument("--volume_time_sigma_bins", type=float, default=2.0, help="Temporal Gaussian sigma in bins for volume scattering.")
    parser.add_argument("--volume_range_weight_power", type=float, default=1.0, help="Additional weighting along depth for scatter accumulation. 1.0 is uniform; >1 emphasizes far scatter.")
    parser.add_argument("--volume_water_front_boost", type=float, default=1.5, help="Extra near-range scattering emphasis for water.")
    parser.add_argument("--volume_fog_front_boost", type=float, default=1.0, help="Extra near-range scattering emphasis for fog.")

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
    parser.add_argument("--raw_max_time_diff", type=float, default=0.05, help="Maximum allowed timestamp difference (seconds) when matching raw RGB/depth frames.")
    parser.add_argument("--raw_stride", type=int, default=1, help="Sample every k matched raw frames after trimming sequence boundaries.")
    parser.add_argument("--drop_first", type=int, default=5, help="Ignore the first N matched frames in each raw scene sequence.")
    parser.add_argument("--drop_last", type=int, default=5, help="Ignore the last N matched frames in each raw scene sequence.")

    # -------------------------
    # Middlebury
    # -------------------------
    parser.add_argument(
        "--middlebury_root",
        type=str,
        default="middlebury/raw",
        help="Root directory of the Middlebury dataset (used when --dataset_mode middlebury).",
    )
    parser.add_argument("--middlebury_depth_min", type=float, default=0.5, help="Minimum pseudo depth in meters after converting disparity to relative depth.")
    parser.add_argument("--middlebury_depth_max", type=float, default=5.0, help="Maximum pseudo depth in meters after converting disparity to relative depth.")

    # -------------------------
    # Output / simulation
    # -------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/spad_dataset",
        help="Directory to save generated .npz samples",
    )
    parser.add_argument("--param_idx", type=int, default=10, help="1-10, following the original MATLAB-style setting.")
    parser.add_argument("--res", type=int, default=64, help="Output spatial resolution.")
    parser.add_argument("--bins", type=int, default=1024, help="Number of temporal bins.")
    parser.add_argument("--bin_size", type=float, default=80e-12, help="Temporal bin size in seconds.")
    parser.add_argument("--irf_sigma", type=float, default=2.0, help="IRF Gaussian sigma in bins for DeepInverse or single-surface temporal broadening.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples for testing.")
    parser.add_argument("--save_x", action="store_true", help="Also save x=(depth-like, signal-like, background) proxy for visualization/debugging.")

    return parser.parse_args()


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
        counts, xhat = simulate_with_deepinverse(x, bins=args.bins, irf_sigma=args.irf_sigma)
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
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.surface_model == "single":
        try:
            from spimaging.generation.deepinverse import import_deepinv

            import_deepinv()
        except ImportError as e:
            raise ImportError("DeepInverse is not installed. Please run: pip install deepinv") from e

    index_rows = []

    if args.dataset_mode == "labeled":
        nyu_mat = Path(args.nyu_mat)
        images, depths = load_nyuv2_labeled(nyu_mat)

        n_samples = images.shape[0]
        if args.limit is not None:
            n_samples = min(n_samples, args.limit)

        for idx in tqdm(range(n_samples), desc=f"Generating SPAD dataset from labeled NYUv2 ({args.surface_model})"):
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

            index_rows.append(
                {
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
            )

    elif args.dataset_mode == "raw":
        raw_root = Path(args.raw_root)
        pairs = scan_nyuv2_raw_pairs(
            raw_root=raw_root,
            max_time_diff=args.raw_max_time_diff,
            stride=args.raw_stride,
            drop_first=args.drop_first,
            drop_last=args.drop_last,
            verbose=True,
        )

        if len(pairs) == 0:
            raise RuntimeError(
                "No matched raw RGB-depth pairs found. "
                "Please check --raw_root / --raw_max_time_diff / trimming settings."
            )

        if args.limit is not None:
            pairs = pairs[:args.limit]

        for idx, item in enumerate(tqdm(pairs, desc=f"Generating SPAD dataset from raw NYUv2 ({args.surface_model})")):
            rgb, depth = load_raw_pair(item["rgb_path"], item["depth_path"])
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

            index_rows.append(
                {
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
            )

    else:  # middlebury
        middlebury_root = Path(args.middlebury_root)
        pairs = scan_middlebury_pairs(middlebury_root=middlebury_root, verbose=True)

        if len(pairs) == 0:
            raise RuntimeError("No valid Middlebury RGB-disparity pairs found. Please check --middlebury_root.")

        if args.limit is not None:
            pairs = pairs[:args.limit]

        for idx, item in enumerate(tqdm(pairs, desc=f"Generating SPAD dataset from Middlebury ({args.surface_model})")):
            rgb, depth = load_middlebury_pair(
                item["rgb_path"],
                item["disp_path"],
                depth_min=args.middlebury_depth_min,
                depth_max=args.middlebury_depth_max,
            )
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

            index_rows.append(
                {
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
            )

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

    print("Done.")
    print(f"Saved {len(index_rows)} samples to: {output_dir}")


if __name__ == "__main__":
    main()
