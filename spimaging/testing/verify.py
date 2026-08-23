import argparse
from pathlib import Path
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    create_output_parent,
    nonnegative_int,
    require_directory,
    require_file,
    validate_npz_archive,
    validate_output_file,
)


def build_parser():
    parser = ArgumentParser(
        prog="spad-verify",
        description="Verify and visualize one generated SPAD dataset sample.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing generated .npz samples.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample_name",
        "--sample",
        dest="sample_name",
        type=str,
        default=None,
        help="Sample filename within --dataset_dir; --sample is a compatibility alias.",
    )
    selection.add_argument(
        "--index",
        type=nonnegative_int,
        default=None,
        metavar="INDEX",
        help="Zero-based sample index in the sorted file list (must exist).",
    )
    selection.add_argument(
        "--random",
        action="store_true",
        help="Randomly choose one sample from dataset_dir.",
    )
    parser.add_argument(
        "--output_fig",
        "--save_fig",
        dest="output_fig",
        type=str,
        default=None,
        help="Optional path for a PNG, JPEG, PDF, or SVG visualization; --save_fig is a compatibility alias.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing --output_fig file to be replaced.",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def list_npz_files(dataset_dir: Path):
    files = sorted(dataset_dir.glob("sample_*.npz"))
    if len(files) == 0:
        files = sorted(dataset_dir.glob("*.npz"))
    return files


def choose_sample(files, args):
    if len(files) == 0:
        raise FileNotFoundError("No .npz files found in the specified dataset directory.")

    if args.sample_name is not None:
        sample_path = Path(args.dataset_dir) / args.sample_name
        if not sample_path.exists():
            raise FileNotFoundError(f"Specified sample not found: {sample_path}")
        return sample_path

    if args.index is not None:
        if args.index < 0 or args.index >= len(files):
            raise IndexError(f"Index {args.index} is out of range. Found {len(files)} samples.")
        return files[args.index]

    if args.random:
        return random.choice(files)

    return files[0]


def normalize_for_display(img):
    img = img.astype(np.float32)
    vmin = np.nanmin(img)
    vmax = np.nanmax(img)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    return (img - vmin) / (vmax - vmin)


def summarize_sample(data, sample_path: Path):
    print("=" * 80)
    print(f"Sample file: {sample_path.name}")
    print(f"Full path   : {sample_path}")
    print("=" * 80)

    for key in data:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"{key:>24s} | shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"{key:>24s} | value={arr}")

    print("=" * 80)

    if "source_mode" in data:
        print(f"Source mode : {data['source_mode']}")
    if "scene" in data:
        print(f"Scene       : {data['scene']}")
    if "rgb_file" in data:
        print(f"RGB file    : {data['rgb_file']}")
    if "depth_file" in data:
        print(f"Depth file  : {data['depth_file']}")
    if "time_diff" in data:
        print(f"Time diff   : {data['time_diff']}")
    if "mean_signal_photons" in data:
        print(f"Mean signal photons     : {data['mean_signal_photons']}")
    if "mean_background_photons" in data:
        print(f"Mean background photons : {data['mean_background_photons']}")
    if "sbr" in data:
        print(f"SBR                     : {data['sbr']}")
    print("=" * 80)


def prepare_xhat_display(xhat):
    """
    Try to produce intuitive visualizations for xhat.

    DeepInverse A_dagger output shape may vary depending on physics implementation.
    Common possibilities:
        - (3, H, W)
        - (bins, H, W)
        - something else
    """
    result = {}

    if xhat.ndim == 3:
        c, h, w = xhat.shape

        if c == 3:
            result["xhat_channel0"] = xhat[0]
            result["xhat_channel1"] = xhat[1]
            result["xhat_channel2"] = xhat[2]
        else:
            # interpret as temporal volume
            result["xhat_sum_over_time"] = xhat.sum(axis=0)
            result["xhat_argmax_time"] = np.argmax(xhat, axis=0).astype(np.float32)
    elif xhat.ndim == 2:
        result["xhat_2d"] = xhat
    else:
        result["xhat_flat"] = np.squeeze(xhat)

    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset_dir = require_directory(parser, args.dataset_dir, "--dataset_dir")

    files = list_npz_files(dataset_dir)
    if not files:
        parser.error(f"--dataset_dir contains no .npz samples: {dataset_dir}")

    if args.sample_name is not None:
        sample_name = Path(args.sample_name)
        if sample_name.name != args.sample_name:
            parser.error("--sample_name must be a filename within --dataset_dir, not a path")
        require_file(
            parser,
            str(dataset_dir / sample_name),
            "--sample_name",
            suffixes=(".npz",),
        )
    elif args.index is not None and args.index >= len(files):
        parser.error(
            f"--index {args.index} is out of range for {len(files)} sample(s); "
            f"expected 0 to {len(files) - 1}"
        )

    sample_path = choose_sample(files, args)
    validate_npz_archive(parser, sample_path, "selected sample")

    output_fig = None
    if args.output_fig is not None:
        output_fig = validate_output_file(
            parser,
            args.output_fig,
            overwrite=args.overwrite,
            option="--output_fig",
            suffixes=(".png", ".jpg", ".jpeg", ".pdf", ".svg"),
        )

    try:
        from spimaging.training_common.security import load_spad_sample

        data = load_spad_sample(sample_path)
        summarize_sample(data, sample_path)

        rgb = data["rgb"] if "rgb" in data else None
        depth = data["depth_m"] if "depth_m" in data else None
        albedo = data["albedo"] if "albedo" in data else None
        intensity = data["intensity"] if "intensity" in data else None
        counts = data["counts"] if "counts" in data else None
        xhat = data["xhat"] if "xhat" in data else None
        x = data["x"] if "x" in data else None
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(f"cannot read selected sample {sample_path}: {exc}")

    # Derived visualizations
    count_map = None
    count_hist = None
    if counts is not None:
        # counts expected shape: (bins, H, W)
        if counts.ndim == 3:
            count_map = counts.sum(axis=0)
            count_hist = counts.sum(axis=(1, 2))
        elif counts.ndim == 2:
            count_map = counts
            count_hist = counts.sum(axis=0)

    xhat_views = {}
    if xhat is not None:
        xhat_views = prepare_xhat_display(xhat)

    # Count panels
    panels = []

    if rgb is not None:
        panels.append(("RGB image", np.clip(rgb / 255.0 if rgb.max() > 1.5 else rgb, 0, 1), "rgb"))

    if depth is not None:
        panels.append(("Depth map (meters)", depth, "viridis"))

    if albedo is not None:
        panels.append(("Albedo surrogate", albedo, "gray"))

    if intensity is not None:
        panels.append(("Intensity surrogate", intensity, "gray"))

    if count_map is not None:
        panels.append(("Photon count map (sum over all time bins)", count_map, "magma"))

    for name, arr in xhat_views.items():
        pretty = {
            "xhat_channel0": "xhat channel 0",
            "xhat_channel1": "xhat channel 1",
            "xhat_channel2": "xhat channel 2",
            "xhat_sum_over_time": "xhat sum over time",
            "xhat_argmax_time": "xhat peak time-bin index",
            "xhat_2d": "xhat 2D view",
            "xhat_flat": "xhat squeezed view",
        }.get(name, name)
        panels.append((pretty, arr, "viridis"))

    if x is not None and x.ndim == 3 and x.shape[0] == 3:
        panels.append(("Saved x channel 0: depth bins", x[0], "viridis"))
        panels.append(("Saved x channel 1: signal", x[1], "magma"))
        panels.append(("Saved x channel 2: background", x[2], "magma"))

    # Layout
    n_image_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil((n_image_panels + 1) / ncols))  # +1 for histogram panel

    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(
        f"SPAD Dataset Verification\nSample: {sample_path.name}",
        fontsize=14,
        fontweight="bold",
    )

    # image panels
    for i, (title, arr, cmap) in enumerate(panels, start=1):
        ax = plt.subplot(nrows, ncols, i)

        if arr.ndim == 3 and arr.shape[-1] == 3:
            ax.imshow(arr)
        else:
            ax.imshow(arr, cmap=cmap)
            plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # histogram panel
    ax = plt.subplot(nrows, ncols, n_image_panels + 1)
    if count_hist is not None:
        ax.plot(count_hist)
        ax.set_title("Global photon histogram over time bins", fontsize=11)
        ax.set_xlabel("Time-bin index")
        ax.set_ylabel("Photon counts summed over all pixels")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No counts available", ha="center", va="center")
        ax.set_title("Global photon histogram over time bins", fontsize=11)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_fig is not None:
        create_output_parent(parser, output_fig, option="--output_fig")
        plt.savefig(output_fig, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {output_fig}")

    plt.show()


if __name__ == "__main__":
    main()
