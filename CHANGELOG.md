# SPImaging 版本说明

## V1.0（技术构建 0.2.0-beta.1）- 2026-08-27

- 提供 Windows 10/11 x64 中文 PySide6 实验工作台。
- 启动时由用户明确选择 CPU 或 NVIDIA GPU；安装过程显示阶段、进度、实时日志、耗时和下载缓存增长。
- 不内置 Conda、PyTorch 或 CUDA Toolkit；优先复用通过真实张量与 3D 卷积自检的现有环境，必要时使用随安装器提供的 uv 创建私有 Python 环境并按 CPU/GPU 模式安装套件。
- 支持 Single、Neighborhood Mix、Translucent Layer、Volume Scattering 四种仿真模型。
- 支持 Simple3D、PRSNet、PENonLocal、STIN 和 SPISR 五种重建模型，以及快速、标准和自定义参数预设。
- 支持结构化运行记录、安全取消、兼容断点恢复、历史索引重建、结果画廊、结果导出与脱敏诊断包。
- 提供 4 个 CC0 合成演示样本和 Simple3D 演示 checkpoint；不在公开安装包中分发 NYUv2/Middlebury 衍生数据。
- 当前安装构建未做 Authenticode 签名，Windows 可能显示“未知发布者”；取得证书前继续明确标记为 unsigned beta。
