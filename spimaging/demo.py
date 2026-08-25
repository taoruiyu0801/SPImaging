"""Run the stable inspect-train-predict-evaluate demonstration workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, TextIO

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    create_output_parent,
    require_directory,
    validate_npz_archive,
    validate_output_directory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path("example_data/nyuv2_raw_single_random_snr")
DEFAULT_OUTPUT_DIR = Path("outputs/demo")
OWNED_OUTPUT_NAMES = {
    "verify",
    "train",
    "predict",
    "evaluate",
    "demo.log",
    "demo_summary.json",
}


@dataclass(frozen=True)
class StageSpec:
    """One isolated command in the demonstration workflow."""

    name: str
    label: str
    module: str
    arguments: tuple[str, ...]


class DemoError(RuntimeError):
    """Expected runtime failure with a concise user-facing message."""


class DemoStageError(DemoError):
    def __init__(self, stage: StageSpec, returncode: int):
        super().__init__(
            f"stage '{stage.label}' failed with exit code {returncode}; "
            "see the output above for details"
        )
        self.returncode = returncode


def build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        prog="spad-demo",
        description=(
            "Run the headless SPImaging demonstration: inspect one sample, train a "
            "minimal supervised model, predict one sample, and evaluate the dataset."
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        default=str(DEFAULT_DATASET_DIR),
        metavar="DIR",
        help="Dataset directory containing at least two valid SPAD .npz samples.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help="Dedicated directory for all demonstration artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace prior demo-owned artifacts while preserving unrelated files.",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_from_repository(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def list_samples(dataset_dir: Path) -> list[Path]:
    samples = sorted(dataset_dir.glob("sample_*.npz"))
    if not samples:
        samples = sorted(dataset_dir.glob("*.npz"))
    return samples


def is_ancestor(candidate: Path, path: Path) -> bool:
    try:
        path.relative_to(candidate)
    except ValueError:
        return False
    return True


def prepare_inputs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path, list[Path]]:
    dataset_dir = require_directory(
        parser,
        str(resolve_from_repository(args.dataset_dir)),
        "--dataset_dir",
    ).resolve()
    samples = list_samples(dataset_dir)
    if len(samples) < 2:
        parser.error(
            f"--dataset_dir must contain at least 2 .npz samples for training and "
            f"evaluation; found {len(samples)} in {dataset_dir}"
        )
    for sample in samples:
        validate_npz_archive(
            parser,
            sample,
            "--dataset_dir sample",
            required_keys=("counts", "depth_m"),
        )

    try:
        importlib.import_module("torch")
    except Exception as exc:
        parser.error(
            "PyTorch is required for the demo; install the training extra with "
            f"'python -m pip install -e \".[training]\"' ({exc})"
        )

    output_dir = resolve_from_repository(args.output_dir)
    if output_dir.is_symlink():
        parser.error(f"--output_dir must not be a symbolic link: {output_dir}")
    repository_root = REPOSITORY_ROOT.resolve()
    if is_ancestor(output_dir, repository_root):
        parser.error(f"--output_dir must be a dedicated child directory, not {output_dir}")
    if is_ancestor(dataset_dir, output_dir) or is_ancestor(output_dir, dataset_dir):
        parser.error("--output_dir must be separate from --dataset_dir")
    output_dir = validate_output_directory(
        parser,
        str(output_dir),
        overwrite=args.overwrite,
        option="--output_dir",
    ).resolve()
    return dataset_dir, output_dir, samples


def build_stages(dataset_dir: Path, staging_dir: Path, sample: Path) -> list[StageSpec]:
    checkpoint = staging_dir / "train" / "last.pt"
    return [
        StageSpec(
            name="verify",
            label="checking sample",
            module="spimaging.testing.verify",
            arguments=(
                "--dataset_dir",
                str(dataset_dir),
                "--sample_name",
                sample.name,
                "--output_fig",
                str(staging_dir / "verify" / f"{sample.stem}.png"),
            ),
        ),
        StageSpec(
            name="train",
            label="minimal training",
            module="spimaging.supervised_training.train",
            arguments=(
                "--dataset_dir",
                str(dataset_dir),
                "--output_dir",
                str(staging_dir / "train"),
                "--epochs",
                "1",
                "--batch_size",
                "1",
                "--max_samples",
                "2",
                "--model",
                "simple3d",
                "--base_channels",
                "2",
                "--temporal_downsample",
                "64",
                "--val_fraction",
                "0.5",
                "--num_workers",
                "0",
                "--seed",
                "0",
            ),
        ),
        StageSpec(
            name="predict",
            label="predicting sample",
            module="spimaging.testing.predict",
            arguments=(
                "--checkpoint",
                str(checkpoint),
                "--sample_file",
                str(sample),
                "--output_npz",
                str(staging_dir / "predict" / "prediction.npz"),
                "--output_fig",
                str(staging_dir / "predict" / "comparison.png"),
            ),
        ),
        StageSpec(
            name="evaluate",
            label="evaluating dataset",
            module="spimaging.testing.evaluate",
            arguments=(
                "--checkpoint",
                str(checkpoint),
                "--label",
                "demo-simple3d",
                "--dataset_dir",
                str(dataset_dir),
                "--output_dir",
                str(staging_dir / "evaluate"),
                "--figure_index",
                "0",
            ),
        ),
    ]


def display_path(path: Path, dataset_dir: Path, staging_dir: Path) -> str:
    for root, label in (
        (staging_dir, "<demo-output>"),
        (REPOSITORY_ROOT, "<SPImaging>"),
        (dataset_dir, "<dataset>"),
    ):
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if str(relative) == ".":
            return label
        return f"{label}/{relative.as_posix()}"
    return f"<external>/{path.name}"


def display_command(stage: StageSpec, dataset_dir: Path, staging_dir: Path) -> str:
    parts = [sys.executable, "-m", stage.module]
    for value in stage.arguments:
        path = Path(value)
        if path.is_absolute():
            parts.append(display_path(path, dataset_dir, staging_dir))
        else:
            parts.append(value)
    parts[0] = "python"
    return subprocess.list2cmdline(parts)


def sanitize_output(text: str, dataset_dir: Path, staging_dir: Path) -> str:
    replacements = (
        (str(staging_dir), "<demo-output>"),
        (str(REPOSITORY_ROOT), "<SPImaging>"),
        (str(dataset_dir), "<dataset>"),
    )
    sanitized = text
    for original, replacement in replacements:
        sanitized = sanitized.replace(original, replacement)
        sanitized = sanitized.replace(original.replace("\\", "/"), replacement)
    return sanitized


def child_environment(runtime_cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(runtime_cache)
    environment["PYTHONWARNINGS"] = "ignore:FigureCanvasAgg is non-interactive:UserWarning"
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(REPOSITORY_ROOT), prior_pythonpath) if item
    )
    return environment


def run_stage(
    stage: StageSpec,
    *,
    log_file: TextIO,
    environment: dict[str, str],
    dataset_dir: Path,
    staging_dir: Path,
) -> float:
    command = [sys.executable, "-m", stage.module, *stage.arguments]
    shown_command = display_command(stage, dataset_dir, staging_dir)
    print(f"Command: {shown_command}")
    log_file.write(f"Command: {shown_command}\n")
    log_file.flush()

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = sanitize_output(line, dataset_dir, staging_dir)
        print(line, end="")
        log_file.write(line)
        log_file.flush()
    returncode = process.wait()
    duration = time.perf_counter() - started
    if returncode != 0:
        raise DemoStageError(stage, returncode)
    return duration


def validate_demo_artifacts(
    staging_dir: Path,
    sample: Path,
    sample_count: int,
) -> dict[str, object]:
    required_files = (
        staging_dir / "verify" / f"{sample.stem}.png",
        staging_dir / "train" / "last.pt",
        staging_dir / "predict" / "prediction.npz",
        staging_dir / "predict" / "comparison.png",
        staging_dir / "evaluate" / "metrics_per_sample.csv",
        staging_dir / "evaluate" / "metrics_summary.json",
        staging_dir / "evaluate" / "comparison.png",
    )
    missing = [path.relative_to(staging_dir).as_posix() for path in required_files if not path.is_file()]
    if missing:
        raise DemoError(f"demo completed without required artifact(s): {', '.join(missing)}")

    import numpy as np

    try:
        with np.load(staging_dir / "predict" / "prediction.npz") as prediction:
            if "pred_depth_m" not in prediction.files:
                raise DemoError("prediction.npz is missing pred_depth_m")
            if not np.isfinite(prediction["pred_depth_m"]).all():
                raise DemoError("prediction.npz contains non-finite depth values")
    except (OSError, ValueError) as exc:
        raise DemoError(f"cannot validate prediction.npz: {exc}") from exc

    try:
        metrics = json.loads(
            (staging_dir / "evaluate" / "metrics_summary.json").read_text(encoding="utf-8")
        )
        demo_metrics = metrics["demo-simple3d"]
        if int(demo_metrics["n_samples"]) != sample_count:
            raise DemoError(
                f"evaluation covered {demo_metrics['n_samples']} samples; expected {sample_count}"
            )
        for metric_name in ("mae_m", "rmse_m", "abs_rel"):
            if not math.isfinite(float(demo_metrics[metric_name])):
                raise DemoError(f"evaluation metric {metric_name} is not finite")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, DemoError):
            raise
        raise DemoError(f"cannot validate metrics_summary.json: {exc}") from exc
    return metrics


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def copy_preserved_path(source: Path, destination: Path) -> None:
    if source.is_symlink():
        destination.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def publish_staging(staging_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        os.replace(staging_dir, output_dir)
        return
    if not any(output_dir.iterdir()):
        output_dir.rmdir()
        os.replace(staging_dir, output_dir)
        return
    if not overwrite:
        raise DemoError(f"output directory is not empty: {output_dir}")

    for previous in output_dir.iterdir():
        if previous.name not in OWNED_OUTPUT_NAMES:
            copy_preserved_path(previous, staging_dir / previous.name)

    backup_dir = output_dir.parent / f".{output_dir.name}.backup-{os.getpid()}-{time.time_ns()}"
    os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
        shutil.rmtree(backup_dir)
    except Exception:
        if output_dir.exists():
            remove_path(output_dir)
        if backup_dir.exists():
            os.replace(backup_dir, output_dir)
        raise


StageRunner = Callable[..., float]


def run_demo(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    stage_runner: StageRunner = run_stage,
) -> Path:
    dataset_dir, output_dir, samples = prepare_inputs(parser, args)
    sample = samples[0]
    create_output_parent(parser, output_dir, option="--output_dir")
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    runtime_cache = staging_dir / ".runtime-cache"
    environment = child_environment(runtime_cache)
    stages = build_stages(dataset_dir, staging_dir, sample)
    records = []
    started = time.perf_counter()
    published = False

    try:
        log_path = staging_dir / "demo.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
            log_file.write("SPImaging stable demo workflow\n")
            log_file.write(f"Dataset: {display_path(dataset_dir, dataset_dir, staging_dir)}\n")
            log_file.write(f"Samples: {len(samples)}\n\n")
            for index, stage in enumerate(stages, start=1):
                prefix = f"[{index}/{len(stages)}]"
                print(f"{prefix} {stage.label.capitalize()}...")
                log_file.write(f"{prefix} {stage.label}\n")
                duration = stage_runner(
                    stage,
                    log_file=log_file,
                    environment=environment,
                    dataset_dir=dataset_dir,
                    staging_dir=staging_dir,
                )
                command = display_command(stage, dataset_dir, staging_dir)
                records.append(
                    {
                        "name": stage.name,
                        "status": "success",
                        "duration_seconds": round(float(duration), 3),
                        "command": command,
                    }
                )
                print(f"{prefix} {stage.label.capitalize()}... OK ({duration:.1f}s)")
                log_file.write(f"Status: success ({duration:.3f}s)\n\n")

        if runtime_cache.exists():
            shutil.rmtree(runtime_cache)
        metrics = validate_demo_artifacts(staging_dir, sample, len(samples))
        summary = {
            "status": "success",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset_dir": display_path(dataset_dir, dataset_dir, staging_dir),
            "sample_count": len(samples),
            "prediction_sample": sample.name,
            "configuration": {
                "model": "simple3d",
                "epochs": 1,
                "batch_size": 1,
                "max_samples": 2,
                "base_channels": 2,
                "temporal_downsample": 64,
                "val_fraction": 0.5,
                "num_workers": 0,
                "seed": 0,
            },
            "stages": records,
            "artifacts": {
                "verify_figure": f"verify/{sample.stem}.png",
                "checkpoint": "train/last.pt",
                "best_checkpoint": "train/best.pt" if (staging_dir / "train" / "best.pt").is_file() else None,
                "prediction": "predict/prediction.npz",
                "prediction_figure": "predict/comparison.png",
                "metrics_per_sample": "evaluate/metrics_per_sample.csv",
                "metrics_summary": "evaluate/metrics_summary.json",
                "evaluation_figure": "evaluate/comparison.png",
                "log": "demo.log",
            },
            "metrics": metrics,
            "total_duration_seconds": round(time.perf_counter() - started, 3),
        }
        (staging_dir / "demo_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_staging(staging_dir, output_dir, args.overwrite)
        published = True
        print(f"Demo completed successfully: {output_dir}")
        return output_dir
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_demo(args, parser)
    except DemoStageError as exc:
        parser.exit(exc.returncode or 1, f"spad-demo: error: {exc}\n")
    except (DemoError, OSError) as exc:
        parser.exit(1, f"spad-demo: error: {exc}\n")


if __name__ == "__main__":
    main()
