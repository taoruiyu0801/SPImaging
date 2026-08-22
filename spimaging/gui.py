"""Tkinter desktop interface for the stable SPImaging demo workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from spimaging import demo


REPOSITORY_ROOT = demo.REPOSITORY_ROOT
DEFAULT_DATASET_DIR = (REPOSITORY_ROOT / demo.DEFAULT_DATASET_DIR).resolve()
DEFAULT_OUTPUT_DIR = (REPOSITORY_ROOT / demo.DEFAULT_OUTPUT_DIR).resolve()
_PROGRESS_PATTERN = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)\s*$")


@dataclass(frozen=True)
class ProgressUpdate:
    """Progress information parsed from one spad-demo output line."""

    current: int
    total: int
    percent: int
    label: str
    completed: bool


def build_demo_command(
    dataset_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool,
) -> list[str]:
    """Build the unbuffered child command used by the GUI."""

    command = [
        sys.executable,
        "-u",
        "-m",
        "spimaging.demo",
        "--dataset_dir",
        str(dataset_dir),
        "--output_dir",
        str(output_dir),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def validate_gui_inputs(
    dataset_value: str,
    output_value: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Perform fast, non-mutating validation before starting the child process."""

    if not dataset_value.strip():
        raise ValueError("请选择数据集目录。")
    if not output_value.strip():
        raise ValueError("请选择输出目录。")

    dataset_dir = demo.resolve_from_repository(dataset_value.strip())
    output_dir = demo.resolve_from_repository(output_value.strip())
    if not dataset_dir.is_dir():
        raise ValueError(f"数据集目录不存在：{dataset_dir}")
    sample_count = len(demo.list_samples(dataset_dir))
    if sample_count < 2:
        raise ValueError(
            f"数据集目录至少需要 2 个 NPZ 样本；当前找到 {sample_count} 个。"
        )

    if output_dir.is_symlink():
        raise ValueError(f"输出目录不能是符号链接：{output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"输出路径不是目录：{output_dir}")
    if demo.is_ancestor(output_dir, REPOSITORY_ROOT.resolve()):
        raise ValueError("输出目录必须是项目目录内的专用子目录，不能是项目根目录或其上级目录。")
    if demo.is_ancestor(dataset_dir, output_dir) or demo.is_ancestor(
        output_dir, dataset_dir
    ):
        raise ValueError("输出目录必须与数据集目录分开，不能互相包含。")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError("输出目录不是空目录；如需重跑，请勾选“覆盖已有演示结果”。")
    return dataset_dir, output_dir


def progress_from_line(line: str) -> ProgressUpdate | None:
    """Parse the stage markers printed by :mod:`spimaging.demo`."""

    match = _PROGRESS_PATTERN.match(line.strip())
    if match is None:
        return None
    current = int(match.group(1))
    total = int(match.group(2))
    if total <= 0 or current < 1 or current > total:
        return None
    raw_label = match.group(3)
    completed = bool(re.search(r"\.\.\.\s+OK(?:\s|\(|$)", raw_label))
    label = re.sub(r"\.\.\..*$", "", raw_label).strip()
    finished_stages = current if completed else current - 1
    percent = round(100 * finished_stages / total)
    return ProgressUpdate(current, total, percent, label, completed)


def load_demo_summary(output_dir: str | Path) -> dict[str, Any]:
    """Load and minimally validate a completed demo summary."""

    summary_path = Path(output_dir) / "demo_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取演示摘要 {summary_path}：{exc}") from exc
    if not isinstance(summary, dict) or summary.get("status") != "success":
        raise ValueError(f"演示摘要没有成功状态：{summary_path}")
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Format the key completion statistics for the status area."""

    parts = [f"总耗时 {float(summary.get('total_duration_seconds', 0)):.1f} 秒"]
    parts.append(f"样本数 {int(summary.get('sample_count', 0))}")
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        for values in metrics.values():
            if not isinstance(values, dict):
                continue
            try:
                parts.extend(
                    [
                        f"MAE {float(values['mae_m']):.4g} m",
                        f"RMSE {float(values['rmse_m']):.4g} m",
                        f"AbsRel {float(values['abs_rel']):.4g}",
                    ]
                )
            except (KeyError, TypeError, ValueError):
                pass
            break
    return "；".join(parts)


def open_directory(path: Path) -> None:
    """Open a directory in the platform file manager without invoking a shell."""

    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class SPImagingDemoApp:
    """Small desktop controller for the headless demonstration command."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SPImaging 一键演示")
        self.root.minsize(820, 580)
        self.root.geometry("920x680")

        self.dataset_var = tk.StringVar(value=str(DEFAULT_DATASET_DIR))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.overwrite_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪")
        self.stage_var = tk.StringVar(value="等待开始")
        self.progress_var = tk.IntVar(value=0)

        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._running = False
        self._form_widgets: list[tk.Widget] = []
        self._output_dir: Path | None = None

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(7, weight=1)

        title = ttk.Label(
            outer,
            text="SPImaging 检查—训练—预测—评估",
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(
            outer,
            text="选择输入与输出目录后，程序会在后台依次运行四个稳定演示阶段。",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 16))

        ttk.Label(outer, text="数据集目录").grid(row=2, column=0, sticky="w", pady=5)
        dataset_entry = ttk.Entry(outer, textvariable=self.dataset_var)
        dataset_entry.grid(row=2, column=1, sticky="ew", padx=(12, 8), pady=5)
        dataset_button = ttk.Button(outer, text="浏览…", command=self._browse_dataset)
        dataset_button.grid(row=2, column=2, sticky="ew", pady=5)

        ttk.Label(outer, text="输出目录").grid(row=3, column=0, sticky="w", pady=5)
        output_entry = ttk.Entry(outer, textvariable=self.output_var)
        output_entry.grid(row=3, column=1, sticky="ew", padx=(12, 8), pady=5)
        output_button = ttk.Button(outer, text="浏览…", command=self._browse_output)
        output_button.grid(row=3, column=2, sticky="ew", pady=5)

        overwrite_check = ttk.Checkbutton(
            outer,
            text="覆盖已有演示结果（保留输出目录中的无关文件）",
            variable=self.overwrite_var,
        )
        overwrite_check.grid(row=4, column=1, columnspan=2, sticky="w", pady=(4, 12))

        actions = ttk.Frame(outer)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 14))
        self.start_button = ttk.Button(actions, text="开始演示", command=self.start_demo)
        self.start_button.pack(side="left")
        self.open_button = ttk.Button(
            actions,
            text="打开输出目录",
            command=self._open_output,
            state="disabled",
        )
        self.open_button.pack(side="left", padx=(10, 0))
        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

        progress_frame = ttk.Frame(outer)
        progress_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Label(progress_frame, textvariable=self.stage_var).grid(row=0, column=0, sticky="w")
        ttk.Label(progress_frame, textvariable=self.progress_var, width=5).grid(
            row=0, column=1, sticky="e"
        )
        ttk.Label(progress_frame, text="%").grid(row=0, column=2, sticky="w")
        ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        log_frame = ttk.LabelFrame(outer, text="运行日志", padding=8)
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self._form_widgets = [
            dataset_entry,
            dataset_button,
            output_entry,
            output_button,
            overwrite_check,
        ]

    def _browse_dataset(self) -> None:
        selected = filedialog.askdirectory(
            title="选择 SPImaging 数据集目录",
            initialdir=self.dataset_var.get() or str(REPOSITORY_ROOT),
            mustexist=True,
        )
        if selected:
            self.dataset_var.set(selected)

    def _browse_output(self) -> None:
        current = Path(self.output_var.get()).expanduser()
        initial = current if current.is_dir() else current.parent
        selected = filedialog.askdirectory(
            title="选择演示输出目录",
            initialdir=str(initial),
            mustexist=False,
        )
        if selected:
            self.output_var.set(selected)

    def _set_running(self, running: bool) -> None:
        self._running = running
        state = "disabled" if running else "normal"
        self.start_button.configure(state=state)
        for widget in self._form_widgets:
            widget.configure(state=state)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_demo(self) -> None:
        if self._running:
            return
        try:
            dataset_dir, output_dir = validate_gui_inputs(
                self.dataset_var.get(),
                self.output_var.get(),
                self.overwrite_var.get(),
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法开始演示", str(exc), parent=self.root)
            return

        self.dataset_var.set(str(dataset_dir))
        self.output_var.set(str(output_dir))
        self._output_dir = output_dir
        self.open_button.configure(state="disabled")
        self.progress_var.set(0)
        self.stage_var.set("正在启动…")
        self.status_var.set("运行中")
        self._clear_log()
        command = build_demo_command(dataset_dir, output_dir, self.overwrite_var.get())
        self._append_log(f"> {subprocess.list2cmdline(command)}\n\n")
        self._set_running(True)
        threading.Thread(
            target=self._run_process,
            args=(command,),
            daemon=True,
            name="spimaging-demo",
        ).start()
        self.root.after(100, self._poll_events)

    def _run_process(self, command: list[str]) -> None:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUNBUFFERED"] = "1"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
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
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._events.put(("line", line))
            returncode = self._process.wait()
            self._events.put(("done", returncode))
        except Exception as exc:
            self._events.put(("error", str(exc)))
        finally:
            self._process = None

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if event == "line":
                line = str(payload)
                self._append_log(line)
                update = progress_from_line(line)
                if update is not None:
                    self.progress_var.set(update.percent)
                    self.stage_var.set(
                        f"第 {update.current}/{update.total} 步：{update.label}"
                    )
            elif event == "done":
                self._finish_process(int(payload))
            elif event == "error":
                self._finish_with_error(f"无法启动演示进程：{payload}")

        if self._running:
            self.root.after(100, self._poll_events)

    def _finish_process(self, returncode: int) -> None:
        if returncode != 0:
            self._finish_with_error(f"演示运行失败，退出码为 {returncode}。请查看上方日志。")
            return
        try:
            if self._output_dir is None:
                raise ValueError("未记录输出目录。")
            summary = load_demo_summary(self._output_dir)
            details = format_summary(summary)
        except ValueError as exc:
            self._finish_with_error(str(exc))
            return

        self.progress_var.set(100)
        self.stage_var.set(f"演示完成：{details}")
        self.status_var.set("成功")
        self._set_running(False)
        self.open_button.configure(state="normal")
        messagebox.showinfo(
            "演示完成",
            f"检查、训练、预测和评估均已完成。\n\n{details}",
            parent=self.root,
        )

    def _finish_with_error(self, message: str) -> None:
        self.status_var.set("失败")
        self.stage_var.set(message)
        self._set_running(False)
        if self._output_dir is not None and self._output_dir.is_dir():
            self.open_button.configure(state="normal")
        messagebox.showerror("演示失败", message, parent=self.root)

    def _open_output(self) -> None:
        if self._output_dir is None:
            candidate = demo.resolve_from_repository(self.output_var.get())
        else:
            candidate = self._output_dir
        if not candidate.is_dir():
            messagebox.showerror("无法打开目录", f"目录不存在：{candidate}", parent=self.root)
            return
        try:
            open_directory(candidate)
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc), parent=self.root)

    def _on_close(self) -> None:
        if self._running:
            messagebox.showwarning(
                "演示正在运行",
                "为避免留下不完整的训练进程，请等待演示结束后再关闭窗口。",
                parent=self.root,
            )
            return
        self.root.destroy()


def main() -> None:
    """Launch the SPImaging demo window."""

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f"spimaging-gui: 无法创建图形窗口：{exc}") from exc
    SPImagingDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
