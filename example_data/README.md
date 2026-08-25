# 内置演示数据

`nyuv2_raw_single_random_snr/` 是 SPImaging 仓库自带的小型演示数据，也是 Day 19 任务所要求的 demo data。该目录已经被 README、`spad-demo`、训练、预测和评估入口共同使用，因此不再复制一份同内容的 `demo_data/`。

## 数据内容

```text
nyuv2_raw_single_random_snr/
├── index.csv
├── sample_00000.npz
├── sample_00001.npz
├── sample_00002.npz
└── sample_00003.npz
```

- 样本数：4。
- 4 个 NPZ 合计：4,073,807 字节（约 3.9 MiB）。
- 每个样本的 `counts` 为 `(1024, 64, 64)`，`depth_m` 为 `(64, 64)`。
- 数据来自 NYUv2 raw，经项目的 `single` 单表面模型生成，包含不同的信号光子数和 SBR。

这套数据适合安装检查、可视化、训练 smoke test、预测和评估，不适合正式训练或学术精度比较。

## 公开演示 checkpoint

公开、可重现的合成数据与预训练模型位于：

```text
public_demo/dataset/
public_demo/checkpoint/simple3d_synthetic.pt
```

文件大小、SHA-256、生成参数和训练配方见 `public_demo/manifest.json`。Windows 安装包仅包含这套 CC0 合成资产，不包含本目录中的 NYUv2 衍生数据。
