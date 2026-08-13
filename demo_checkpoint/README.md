# Simple3D 演示 checkpoint

`simple3d_demo_best.pt` 是使用仓库内置 4 个演示样本重新训练得到的监督式 Simple3D checkpoint。它用于快速证明“加载模型—预测—评估”链路可以运行，不代表正式数据集上的模型精度。

## 使用前准备

命令入口安装在 Conda `spimaging` 环境中。可以先激活环境：

```powershell
conda activate spimaging
```

如果当前 PowerShell 仍在其他环境，直接使用 `conda run -n spimaging` 最稳妥。

## 单样本预测

```powershell
conda run -n spimaging spad-predict `
    --checkpoint demo_checkpoint/simple3d_demo_best.pt `
    --sample_file example_data/nyuv2_raw_single_random_snr/sample_00000.npz `
    --output_npz outputs/day19_quick_demo/predict/prediction.npz `
    --output_fig outputs/day19_quick_demo/predict/comparison.png
```

## 全量评估

```powershell
conda run -n spimaging spad-evaluate `
    --checkpoint demo_checkpoint/simple3d_demo_best.pt `
    --label day19-simple3d `
    --dataset_dir example_data/nyuv2_raw_single_random_snr `
    --output_dir outputs/day19_quick_demo/evaluate
```

在 Day 19 验证环境中，单样本预测加 4 样本评估的实测总耗时为：

- 自动选择 CUDA：6.267 秒。
- 强制使用 CPU：5.217 秒。

两种方式均明显低于任务规定的 600 秒。完整元数据、哈希和指标见 `manifest.json`。
