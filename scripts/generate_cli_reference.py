"""Generate the checked-in SPImaging CLI parameter reference."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as quickstart  # noqa: E402
from spimaging import cli  # noqa: E402
from spimaging.generation import pipeline  # noqa: E402
from spimaging.self_supervised_training import train as selfsup_train  # noqa: E402
from spimaging.supervised_training import train as supervised_train  # noqa: E402
from spimaging.testing import browse, evaluate, predict, verify  # noqa: E402


PARSERS = (
    ("python main.py", quickstart.build_parser),
    ("spad-generate", pipeline.build_parser),
    ("spad-train", supervised_train.build_parser),
    ("spad-train-selfsup", selfsup_train.build_parser),
    ("spad-predict", predict.build_parser),
    ("spad-evaluate", evaluate.build_parser),
    ("spad-verify", verify.build_parser),
    ("spad-browse", browse.build_parser),
)


RANGES = {
    cli.finite_float: "有限浮点数",
    cli.positive_int: "整数 >= 1",
    cli.nonnegative_int: "整数 >= 0",
    cli.random_seed: "整数 0—4294967295",
    cli.positive_float: "有限浮点数 > 0",
    cli.nonnegative_float: "有限浮点数 >= 0",
    cli.fraction: "0 <= 值 < 1",
    cli.unit_interval: "0 <= 值 <= 1",
    cli.positive_unit_interval: "0 < 值 <= 1",
    cli.parameter_index: "整数 1—10",
    cli.positive_odd_int: "正奇数",
    cli.model_base_channels: "整数 1—256",
    cli.model_num_blocks: "整数 1—100",
    cli.super_resolution_scale: "整数 1—64",
}


def format_default(action: argparse.Action) -> str:
    if action.required:
        return "必填"
    if action.default is argparse.SUPPRESS:
        return "未设置"
    if action.default is None:
        return "未设置"
    return str(action.default)


def format_type(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "布尔开关"
    if action.type is None:
        return "字符串"
    if action.type in {
        cli.positive_int,
        cli.nonnegative_int,
        cli.random_seed,
        cli.parameter_index,
        cli.positive_odd_int,
        cli.model_base_channels,
        cli.model_num_blocks,
        cli.super_resolution_scale,
    }:
        return "整数"
    if action.type in {
        cli.finite_float,
        cli.positive_float,
        cli.nonnegative_float,
        cli.fraction,
        cli.unit_interval,
        cli.positive_unit_interval,
    }:
        return "浮点数"
    return {int: "整数", float: "浮点数", str: "字符串"}.get(action.type, action.type.__name__)


def format_range(action: argparse.Action) -> str:
    if action.choices is not None:
        return " / ".join(str(choice) for choice in action.choices)
    return RANGES.get(action.type, "—")


def path_behavior(command: str, action: argparse.Action) -> str:
    dest = action.dest
    if dest in {"checkpoint", "sample_file"}:
        return "输入文件：必须存在，不自动创建"
    if dest == "nyu_mat":
        return "输入文件：labeled 数据模式启用时必须存在"
    if dest in {"raw_root", "middlebury_root"}:
        return "输入目录：对应数据模式启用时必须存在"
    if dest == "dataset_dir":
        if command in {"spad-train", "spad-train-selfsup"}:
            return "输入路径：必须是现有目录或 .npz 文件"
        return "输入目录：必须存在且包含 .npz 文件"
    if dest == "sample_name":
        return "输入文件名：相对于 --dataset_dir，不允许绝对路径"
    if dest == "output_dir":
        if command == "spad-browse":
            return "输出目录：首次按 G 保存时自动创建"
        return "输出目录：输入校验通过后自动创建；非空需 --overwrite"
    if dest in {"output_fig", "output_npz"}:
        return "输出文件：输入校验通过后自动创建父目录；已存在需 --overwrite"
    return "—"


def rows():
    for command, factory in PARSERS:
        parser = factory()
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            yield {
                "命令": command,
                "参数": ", ".join(action.option_strings) or action.dest,
                "类型": format_type(action),
                "必填": "是" if action.required else "否",
                "默认值": format_default(action),
                "合法范围/选项": format_range(action),
                "用途": action.help or "—",
                "路径行为": path_behavior(command, action),
            }


def escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, data: list[dict[str, str]]) -> None:
    columns = list(data[0])
    lines = [
        "# SPImaging CLI 参数说明表",
        "",
        "本表由 `scripts/generate_cli_reference.py` 从各入口的 `argparse` 定义生成。",
        "`spad-browse-4modes` 是 `spad-browse` 的兼容别名，两者参数完全相同。",
        "",
        "## 统一规则",
        "",
        "- 参数名称统一使用下划线风格；发生更名的旧名称至少保留一个兼容周期。",
        "- 必填输入不存在或类型不符时，给出无 traceback 的明确错误并以状态码 2 退出。",
        "- 输入目录和输入文件绝不自动创建；输出目录及输出文件父目录仅在输入校验通过后创建。",
        "- 相对路径以当前工作目录为基准；已有输出默认不覆盖，显式传入 `--overwrite` 才允许替换。",
        "- `python main.py` 是单样本快速演示入口，因此部分业务默认值可与完整命令 `spad-generate` 不同。",
        "",
        "## 参数表",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in data:
        lines.append("| " + " | ".join(escape_markdown(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, data: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    output_dir = ROOT / "record_of_SPI" / "Day_13-14"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = list(rows())
    write_markdown(output_dir / "参数说明表.md", data)
    write_csv(output_dir / "参数说明表.csv", data)
    print(f"Wrote {len(data)} parameter rows to: {output_dir}")


if __name__ == "__main__":
    main()
