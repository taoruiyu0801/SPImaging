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

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    nonnegative_float,
    parameter_index,
    positive_float,
    positive_int,
    positive_odd_int,
    require_directory,
    require_file,
    unit_interval,
    validate_output_directory,
    validate_output_file,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        prog="python main.py",
        description="Generate SPAD-style data with a selected measurement model and display the result.",
        formatter_class=HelpFormatter,
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
    parser.add_argument("--limit", type=positive_int, default=1, help="Positive number of samples to generate in this quick-run entry point.")
    parser.add_argument("--res", type=positive_int, default=64, help="Positive square output resolution in pixels.")
    parser.add_argument("--bins", type=positive_int, default=1024, help="Positive number of temporal bins.")
    parser.add_argument("--param_idx", type=parameter_index, default=10, help="Photon simulation parameter index from 1 to 10.")
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
        "--output_fig",
        "--save_fig",
        dest="output_fig",
        default=None,
        help="Optional figure file for verify display; --save_fig is a compatibility alias.",
    )

    parser.add_argument("--mix_kernel_size", type=positive_odd_int, default=5, help="Positive odd neighborhood-mixing kernel size.")
    parser.add_argument("--mix_sigma_xy", type=positive_float, default=1.0, help="Positive spatial Gaussian sigma in pixels.")
    parser.add_argument("--mix_time_sigma_bins", type=positive_float, default=2.0, help="Positive temporal Gaussian sigma in bins.")

    parser.add_argument("--translucent_front_type", choices=["flat", "sloped", "sinusoidal"], default="sinusoidal", help="Front semi-transparent layer geometry.")
    parser.add_argument("--translucent_front_depth", type=positive_float, default=1.0, help="Positive front-layer depth in meters.")
    parser.add_argument("--translucent_front_signal_ratio", type=nonnegative_float, default=0.25, help="Nonnegative front-layer reflection strength.")
    parser.add_argument("--translucent_transmission", type=unit_interval, default=0.6, help="Back-scene transmission factor between 0 and 1.")

    parser.add_argument("--volume_medium_type", choices=["fog", "water"], default="fog", help="Volume-scattering medium type.")
    parser.add_argument("--volume_num_steps", type=positive_int, default=64, help="Positive number of path-scattering range samples.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing generated samples and figure output.")

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.dataset_mode == "middlebury":
        require_directory(parser, args.middlebury_root, "--middlebury_root")
    elif args.dataset_mode == "labeled":
        require_file(parser, args.nyu_mat, "--nyu_mat", suffixes=(".mat",))
    else:
        require_directory(parser, args.raw_root, "--raw_root")

    validate_output_directory(parser, args.output_dir, overwrite=args.overwrite)
    if args.output_fig is not None:
        if args.display != "verify":
            parser.error("--output_fig can only be used with --display verify")
        validate_output_file(
            parser,
            args.output_fig,
            overwrite=args.overwrite,
            option="--output_fig",
            suffixes=(".png", ".jpg", ".jpeg", ".pdf", ".svg"),
        )


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
    if args.overwrite:
        argv.append("--overwrite")

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
        if args.output_fig:
            argv.extend(["--output_fig", args.output_fig])
        if args.overwrite:
            argv.append("--overwrite")
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
    if args.overwrite:
        argv.append("--overwrite")
    with patched_argv(argv):
        browse_main()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    print(f"Generating data with surface_model={args.surface_model!r} into {args.output_dir}")
    run_generation(args)
    show_generated_data(args)


if __name__ == "__main__":
    main()
