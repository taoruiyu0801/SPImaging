# SPImaging Windows 打包说明

当前默认发布形态是轻量 Windows 安装包。安装包包含启动器、应用源码、`public_demo/` 和 `uv.exe`，不包含 Conda、PyTorch、CUDA Toolkit 或 NVIDIA 驱动。

## 构建要求

- Windows 10/11 x64
- Python 3.10+、PyInstaller
- `uv.exe`
- Inno Setup 6（`ISCC.exe`）
- 可选：现有 Authenticode 证书与 Windows SDK `signtool.exe`

构建脚本不会创建证书。没有签名输入时，只生成名称明确的 `SPImaging-Setup-unsigned-beta.exe`。

## 当前构建流程

```powershell
python -m pip install pyinstaller pytest uv
python -m pytest -q tests/test_launcher.py tests/test_cuda_engine.py tests/test_desktop.py
packaging/scripts/Build-Launcher.ps1
packaging/scripts/Build-Installer.ps1 -Version 0.2.0-beta.1
```

默认产物位于 `packaging/out/`。安装器按当前用户安装到 `%LOCALAPPDATA%\Programs\SPImaging`，不修改 `PATH`，不要求管理员权限。

首次启动时，`SPImaging.exe` 按以下顺序准备计算引擎：

1. 使用 `nvidia-smi` 读取显卡、驱动和计算能力。
2. 搜索已有 Python/venv/Conda 环境，但只接受能完成真实 CUDA 张量与 3D 卷积的环境。
3. 如果没有兼容环境，由用户确认后使用随安装包提供的 `uv.exe` 创建私有 venv，并按驱动选择 CUDA 版 PyTorch。
4. CUDA 自检失败则停止并显示原因，不回退 CPU，也不安装或修改 NVIDIA 驱动。

私有计算环境和实验结果位于安装目录之外，卸载应用时会保留，便于重装和恢复。当前安装版需要首次联网下载 Python/PyTorch 组件，并建议至少预留 8 GiB 可用空间；准备完成后可以离线启动。

## 安装包内容边界

Inno Setup 只收集：

- `packaging/out/launcher/SPImaging.exe`
- `packaging/out/tools/uv.exe`
- `spimaging/`
- `public_demo/`
- `LICENSE`、`NOTICE`、第三方许可证与 SBOM

`example_data/` 中的 NYUv2 衍生样本、开发测试输出、截图和本机环境记录不会进入安装包。

## 签名

`Build-Installer.ps1` 支持两条显式签名路径：

- `-InnoSignToolCommand`：让 Inno Setup 在构建时签名。
- `-SigningPfxPath` 与 `-CertificateThumbprint`：先构建 unsigned beta，再调用 `Sign-Artifact.ps1` 签名并原子改名为 `SPImaging-Setup.exe`。

没有 Authenticode 证书的公开测试版会触发 Windows SmartScreen“未知发布者”提示，不能称为正式签名版。

## 旧版兼容代码

`packaging/runtime/`、发布清单脚本以及启动器中的 `--legacy-runtime` 路径保留用于验证早期 beta 介质，不是当前默认安装架构。新安装包不要构建或发布旧的 CPU/CUDA `conda-pack` 运行时。
