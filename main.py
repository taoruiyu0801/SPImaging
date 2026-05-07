"""Project-level entry point for generating and viewing SPAD data.

Examples:
    python main.py --surface_model neighborhood_mix --limit 1
    python main.py --surface_model translucent_layer --display browse
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path


SURFACE_MODELS = [
    "single",
    "neighborhood_mix",
    "translucent_layer",
    "volume_scattering",
]


@contextmanager
def patched_argv(argv: list[str]):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SPAD-style data with a selected measurement model and display the result.",
    )

    parser.add_argument(
        "--surface_model",
        choices=SURFACE_MODELS,
        default="neighborhood_mix",
        help="Measurement model used to generate SPAD data.",
    )
    parser.add_argument(
        "--dataset_mode",
        choices=["labeled", "raw", "middlebury"],
        default="middlebury",
        help="Input dataset type.",
    )
    parser.add_argument("--output_dir", default="outputs/main_run", help="Directory for generated .npz samples.")
    parser.add_argument("--limit", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--res", type=int, default=64, help="Output spatial resolution.")
    parser.add_argument("--bins", type=int, default=1024, help="Number of temporal bins.")
    parser.add_argument("--param_idx", type=int, default=10, help="Photon parameter index, 1-10.")
    parser.add_argument("--save_x", action="store_true", help="Save x proxy for visualization/debugging.")
    parser.add_argument(
        "--save_clean_transient",
        action="store_true",
        help="Save clean transient before Poisson sampling for multi-return models.",
    )

    parser.add_argument("--middlebury_root", default="middlebury/raw", help="Middlebury raw dataset root.")
    parser.add_argument("--nyu_mat", default="NYUv2/nyu_depth_v2_labeled.mat", help="NYUv2 labeled .mat file.")
    parser.add_argument("--raw_root", default="NYUv2/raw", help="NYUv2 raw dataset root.")

    parser.add_argument(
        "--display",
        choices=["verify", "browse", "none"],
        default="verify",
        help="Display mode after generation. 'verify' uses Matplotlib, 'browse' uses OpenCV.",
    )
    parser.add_argument(
        "--save_fig",
        default=None,
        help="Optional figure path for verify display, e.g. outputs/main_run_preview.png.",
    )

    parser.add_argument("--mix_kernel_size", type=int, default=5)
    parser.add_argument("--mix_sigma_xy", type=float, default=1.0)
    parser.add_argument("--mix_time_sigma_bins", type=float, default=2.0)

    parser.add_argument("--translucent_front_type", choices=["flat", "sloped", "sinusoidal"], default="sinusoidal")
    parser.add_argument("--translucent_front_depth", type=float, default=1.0)
    parser.add_argument("--translucent_front_signal_ratio", type=float, default=0.25)
    parser.add_argument("--translucent_transmission", type=float, default=0.6)

    parser.add_argument("--volume_medium_type", choices=["fog", "water"], default="fog")
    parser.add_argument("--volume_num_steps", type=int, default=64)

    return parser.parse_args()


def build_generation_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "spad-generate",
        "--dataset_mode",
        args.dataset_mode,
        "--surface_model",
        args.surface_model,
        "--output_dir",
        args.output_dir,
        "--limit",
        str(args.limit),
        "--res",
        str(args.res),
        "--bins",
        str(args.bins),
        "--param_idx",
        str(args.param_idx),
    ]

    if args.dataset_mode == "middlebury":
        argv.extend(["--middlebury_root", args.middlebury_root])
    elif args.dataset_mode == "labeled":
        argv.extend(["--nyu_mat", args.nyu_mat])
    else:
        argv.extend(["--raw_root", args.raw_root])

    if args.save_x:
        argv.append("--save_x")
    if args.save_clean_transient:
        argv.append("--save_clean_transient")

    if args.surface_model == "neighborhood_mix":
        argv.extend(
            [
                "--mix_kernel_size",
                str(args.mix_kernel_size),
                "--mix_sigma_xy",
                str(args.mix_sigma_xy),
                "--mix_time_sigma_bins",
                str(args.mix_time_sigma_bins),
            ]
        )
    elif args.surface_model == "translucent_layer":
        argv.extend(
            [
                "--translucent_front_type",
                args.translucent_front_type,
                "--translucent_front_depth",
                str(args.translucent_front_depth),
                "--translucent_front_signal_ratio",
                str(args.translucent_front_signal_ratio),
                "--translucent_transmission",
                str(args.translucent_transmission),
            ]
        )
    elif args.surface_model == "volume_scattering":
        argv.extend(
            [
                "--volume_medium_type",
                args.volume_medium_type,
                "--volume_num_steps",
                str(args.volume_num_steps),
            ]
        )

    return argv


def run_generation(args: argparse.Namespace) -> None:
    from spimaging.generation.pipeline import main as generate_main

    with patched_argv(build_generation_argv(args)):
        generate_main()


def show_generated_data(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if args.display == "none":
        return

    if args.display == "verify":
        from spimaging.testing.verify import main as verify_main

        argv = ["spad-verify", "--dataset_dir", str(output_dir), "--index", "0"]
        if args.save_fig:
            argv.extend(["--save_fig", args.save_fig])
        with patched_argv(argv):
            verify_main()
        return

    from spimaging.testing.browse import main as browse_main

    argv = [
        "spad-browse",
        "--dataset_dir",
        str(output_dir),
        "--browse_mode",
        "auto",
        "--pixel_source",
        "auto",
    ]
    with patched_argv(argv):
        browse_main()


def main() -> None:
    args = parse_args()
    print(f"Generating data with surface_model={args.surface_model!r} into {args.output_dir}")
    run_generation(args)
    show_generated_data(args)


if __name__ == "__main__":
    main()
