"""Local-only diagnostics with path redaction."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Mapping
import zipfile


_WINDOWS_USER_PATH = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s\"']+")


def redaction_pairs(extra_paths: Iterable[str | Path] = ()) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    candidates = [
        (Path.home(), "<USER_HOME>"),
        (Path(os.environ.get("LOCALAPPDATA", "")), "<LOCAL_APP_DATA>"),
        (Path(os.environ.get("APPDATA", "")), "<APP_DATA>"),
        *[(Path(value), f"<PATH_{index}>") for index, value in enumerate(extra_paths, 1)],
    ]
    seen: set[str] = set()
    for path, token in candidates:
        raw = str(path)
        if not raw or raw == "." or raw in seen:
            continue
        seen.add(raw)
        pairs.append((raw, token))
        pairs.append((raw.replace("\\", "/"), token))
    return tuple(sorted(pairs, key=lambda item: len(item[0]), reverse=True))


def redact_text(text: str, extra_paths: Iterable[str | Path] = ()) -> str:
    redacted = text
    for raw, token in redaction_pairs(extra_paths):
        redacted = redacted.replace(raw, token)
    return _WINDOWS_USER_PATH.sub(r"C:\\Users\\<USER>", redacted)


def collect_environment() -> dict[str, Any]:
    information: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "executable": redact_text(sys.executable),
        "process_architecture": platform.machine(),
    }
    try:
        import torch

        information["torch"] = torch.__version__
        information["cuda_available"] = bool(torch.cuda.is_available())
        information["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            information["gpu_count"] = torch.cuda.device_count()
            information["gpus"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # diagnostics must never prevent startup
        information["torch_error"] = redact_text(str(exc))
    return information


def export_diagnostic_bundle(
    destination: str | Path,
    *,
    run_dir: str | Path | None = None,
    additional_files: Iterable[str | Path] = (),
) -> Path:
    """Export logs/config metadata only; sample/checkpoint contents are excluded."""

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sensitive_roots = [run_dir] if run_dir is not None else []
    allowed_suffixes = {".json", ".jsonl", ".log", ".txt", ".csv", ".md"}
    file_candidates: list[Path] = []
    if run_dir is not None:
        root = Path(run_dir).expanduser().resolve()
        for relative in (
            "run.json",
            "result_manifest.json",
            "events.jsonl",
            "logs/worker.log",
            "training/history.csv",
            "training/history.jsonl",
        ):
            candidate = root / relative
            if candidate.is_file():
                file_candidates.append(candidate)
    file_candidates.extend(Path(value).expanduser().resolve() for value in additional_files)

    with zipfile.ZipFile(destination_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        environment = json.dumps(collect_environment(), ensure_ascii=False, indent=2) + "\n"
        archive.writestr("environment.json", redact_text(environment, sensitive_roots))
        for index, candidate in enumerate(file_candidates, 1):
            if not candidate.is_file() or candidate.suffix.lower() not in allowed_suffixes:
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            archive.writestr(
                f"files/{index:02d}_{candidate.name}",
                redact_text(text, sensitive_roots),
            )
    return destination_path

