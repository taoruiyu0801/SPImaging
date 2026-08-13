# Single-Photon Imaging

面向单光子成像的 SPAD photon-counting 数据生成、检查、浏览、训练与推理工具箱。项目可以从 NYUv2 或 Middlebury RGB-D 数据生成 `.npz` 格式的时间直方图 photon cube，并提供监督深度重建与自监督 SPISR 训练入口。

## 功能概览

- 从 RGB-D 数据生成 SPAD 测量数据，输出 `counts`、`depth_m`、`rgb`、`albedo`、`intensity`、`xhat` 等字段。
- 支持 4 类测量模型：单表面、邻域混合多返回、半透明前层、雾/水体体散射。
- 提供 Matplotlib 单样本检查与 OpenCV 交互式数据浏览器。
- 支持监督式单光子深度重建训练：`simple3d`、`prsnet`、`penonlocal`、`stin`。
- 支持自监督 SPISR 训练：使用 PUKL 与 equivariance 约束，不需要 HR 标签。
- 支持 checkpoint 推理，输出预测深度图和可选对比图。

## 环境安装

推荐使用 Conda 环境：

```bash
conda env create -f environment.yml
conda activate spimaging
python -m pip install --upgrade --force-reinstall \
  "torch==2.11.0+cu130" \
  "torchvision==0.26.0+cu130" \
  "torchaudio==2.11.0+cu130" \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple
python -m pip install --force-reinstall "numpy<2"
python -m pip install deepinv
```

如果环境已经存在，只需要激活并确认 CUDA 版 PyTorch：

```bash
conda activate spimaging
python -m pip install --upgrade --force-reinstall \
  "torch==2.11.0+cu130" \
  "torchvision==0.26.0+cu130" \
  "torchaudio==2.11.0+cu130" \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple
python -m pip install --force-reinstall "numpy<2"
python -m pip install deepinv
```

也可以用 pip 安装基础依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

说明：

- Python 版本建议为 3.10，`pyproject.toml` 要求 `>=3.9`。
- `torch` 和 `deepinv` 对 `--surface_model single`、训练与推理是必需的。
- `neighborhood_mix`、`translucent_layer`、`volume_scattering` 使用项目内 NumPy 实现。
- `spad-browse` 需要可用的本机 OpenCV 图形窗口环境。
- 训练、推理和 `single` 生成会自动选择计算设备：如果当前 PyTorch 可用 CUDA GPU，则优先使用 GPU；否则回退到 CPU。
- 本机已验证的 GPU 配置为 `torch 2.11.0+cu130` / `torchvision 0.26.0+cu130` / `torchaudio 2.11.0+cu130`，驱动显示 CUDA 13.1，PyTorch 运行时显示 CUDA 13.0。
- 如果机器有 NVIDIA GPU 但 `torch.cuda.is_available()` 为 `False`，说明当前环境安装的是 CPU 版 PyTorch，按上面的 CUDA PyTorch 命令重装即可。
- 官方安装命令可按机器情况从 PyTorch Get Started 页面选择：<https://docs.pytorch.org/get-started/locally/>。

验证 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

安装后可以检查命令行入口：

```bash
spad-generate --help
spad-verify --help
spad-browse --help
spad-train --help
spad-train-selfsup --help
spad-predict --help
spad-evaluate --help
spad-demo --help
```

这些命令安装在当前 Python/Conda 环境中。如果 PowerShell 提示命令不存在，先确认已激活 `spimaging`；也可以不切换当前环境，直接使用 `conda run -n spimaging spad-demo`。如果电脑上有多套 Conda，使用 `conda activate <spimaging环境的完整路径>` 可以避免激活到错误环境。

### 命令行参数与路径规则

- 参数名称统一使用下划线风格。`spad-predict` 使用 `--sample_file`，`spad-verify` 使用 `--sample_name` 和 `--output_fig`，`spad-browse` 使用 `--output_dir`。
- 旧参数 `--sample`、`--save_fig`、`--save_dir` 暂时保留为兼容别名；新脚本应使用上述规范名称。
- 输入文件和输入目录必须已经存在，程序不会为输入路径创建空目录；错误输入会给出明确提示、无 Python traceback，并以状态码 2 退出。
- 输出目录以及输出文件的父目录会在全部输入校验通过后自动创建。相对路径以当前工作目录为基准。
- 已有输出默认不会被覆盖；确认需要替换同名结果或写入非空输出目录时，显式传入 `--overwrite`。
- 各入口的默认值、合法范围和路径行为见 [`record_of_SPI/Day_13-14/参数说明表.xlsx`](record_of_SPI/Day_13-14/参数说明表.xlsx)；同目录另提供便于代码审阅和自动校验的 Markdown/CSV 版本。

## 项目结构

```text
.
├── main.py                         # 生成数据并立即可视化的项目级入口
├── pyproject.toml                  # 包配置与 console script 入口
├── environment.yml                 # Conda 环境配置
├── requirements.txt                # pip 依赖列表
├── spimaging/
│   ├── demo.py                     # 无界面的一键检查、训练、预测和评估入口
│   ├── generation/                 # 数据生成：数据读取、预处理、SPAD 测量模型、spad-generate
│   ├── supervised_training/        # 有监督深度重建训练入口，对应 spad-train
│   ├── self_supervised_training/   # 自监督 SPISR 训练入口，对应 spad-train-selfsup
│   ├── testing/                    # 检查、浏览与 checkpoint 推理，对应 spad-verify/browse/predict
│   ├── cli.py                      # 共享 CLI 数值、路径与输出校验
│   └── training_common/            # 训练共享的 Dataset、loss、network、checkpoint 工具
├── tests/                          # CLI 参数、错误提示和无副作用回归测试
├── record_of_SPI/Day_13-14/        # 参数表与代表性异常提示截图
├── example_data/                   # 仓库内置的少量可运行样例数据
├── demo_checkpoint/                # 与内置样例配套的预训练 Simple3D 演示模型
├── middlebury/
│   ├── raw/                        # Middlebury 原始数据，需自行准备
│   └── processed*/                 # 生成后的样本目录
└── outputs/                        # 训练、推理、预览输出
```

## 数据准备

本仓库包含一小份可直接运行的样例数据，路径为：

```text
example_data/nyuv2_raw_single_random_snr/
├── index.csv
├── sample_00000.npz
├── sample_00001.npz
├── sample_00002.npz
└── sample_00003.npz
```

这 4 个样本来自 NYUv2 raw 数据，经本项目 `single` 单表面模型生成，空间分辨率为 `64x64`，时间维为 `1024` bins，`param_idx=10`，即信号光子数与 SBR 随机采样。它们适合快速验证安装、可视化、Dataset 读取、训练 smoke test 和评估脚本，不适合作为正式训练集。

这套已跟踪的 `example_data` 同时作为项目的 demo data，不另建一份内容重复的 `demo_data`。数据说明见 [`example_data/README.md`](example_data/README.md)。

快速检查内置样例：

```bash
spad-verify \
  --dataset_dir example_data/nyuv2_raw_single_random_snr \
  --index 0 \
  --output_fig outputs/example_verify_sample_00000.png
```

用内置样例跑一个最小监督训练：

```bash
spad-train \
  --dataset_dir example_data/nyuv2_raw_single_random_snr \
  --output_dir outputs/train_example_simple3d \
  --epochs 1 \
  --batch_size 1 \
  --model simple3d \
  --base_channels 2 \
  --temporal_downsample 64 \
  --tv_weight 0.005 \
  --val_fraction 0.25 \
  --num_workers 0
```

### 一键稳定演示

安装训练依赖后，可以用一个无界面入口串行完成“检查样例—最小训练—预测—评估”：

```bash
spad-demo
```

默认读取 `example_data/nyuv2_raw_single_random_snr`，输出到 `outputs/demo`。也可以指定其他包含至少两个有效样本的数据集：

```bash
spad-demo \
  --dataset_dir example_data/nyuv2_raw_single_random_snr \
  --output_dir outputs/demo
```

演示固定使用轻量配置训练 1 个 epoch，并在独立子进程中执行四个现有入口。Matplotlib 使用无界面后端，因此不会打开窗口或抢占桌面焦点。输出包括：

```text
outputs/demo/
├── verify/sample_00000.png
├── train/last.pt
├── train/best.pt
├── predict/prediction.npz
├── predict/comparison.png
├── evaluate/metrics_per_sample.csv
├── evaluate/metrics_summary.json
├── evaluate/comparison.png
├── demo.log
└── demo_summary.json
```

已有非空输出目录默认拒绝写入。需要重跑时使用 `--overwrite`；该参数只替换 demo 管理的产物，输出目录中的无关文件会保留。四个阶段全部成功并通过产物校验后，临时 staging 结果才会发布到正式目录；失败时不会留下半成品正式输出。

### 使用预训练 checkpoint 快速演示

如果只需要演示“加载已有模型—预测—评估”，可以使用 Day 19 基于上述 4 个样本重新训练的轻量 checkpoint：

```powershell
conda run -n spimaging spad-predict `
    --checkpoint demo_checkpoint/simple3d_demo_best.pt `
    --sample_file example_data/nyuv2_raw_single_random_snr/sample_00000.npz `
    --output_npz outputs/day19_quick_demo/predict/prediction.npz `
    --output_fig outputs/day19_quick_demo/predict/comparison.png

conda run -n spimaging spad-evaluate `
    --checkpoint demo_checkpoint/simple3d_demo_best.pt `
    --label day19-simple3d `
    --dataset_dir example_data/nyuv2_raw_single_random_snr `
    --output_dir outputs/day19_quick_demo/evaluate
```

checkpoint 只有约 33 KiB；配套的 4 个 NPZ 约 3.9 MiB。在 Day 19 验收中，单样本预测加 4 样本评估使用 CUDA 耗时 6.267 秒，强制使用 CPU 耗时 5.217 秒，均远低于 10 分钟。训练与评估使用同一小型样本集合，指标只用于 smoke demo，不代表独立测试集精度。完整哈希、参数和指标见 [`demo_checkpoint/manifest.json`](demo_checkpoint/manifest.json)。

完整数据集不随仓库上传，请按需要从官方来源下载：

| 数据集 | 官方入口 | 本项目默认放置位置 |
| --- | --- | --- |
| NYU Depth Dataset V2 labeled/raw | <https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html> | `NYUv2/nyu_depth_v2_labeled.mat`、`NYUv2/raw/` |
| Middlebury Stereo 2006 scenes | <https://vision.middlebury.edu/stereo/data/scenes2006/> | `middlebury/raw/` |

下载完整数据后，再使用 `spad-generate` 转成项目需要的 `.npz` photon-counting 样本。

### Middlebury

默认 Middlebury 目录为 `middlebury/raw`。每个 scene 目录至少需要：

```text
middlebury/raw/<scene_name>/
├── view1.png
└── disp1.png
```

生成时项目会把 disparity 转换为相对深度，默认深度范围为 `0.5m` 到 `5.0m`，可通过 `--middlebury_depth_min` 和 `--middlebury_depth_max` 调整。

### NYUv2 labeled

默认路径为：

```text
NYUv2/nyu_depth_v2_labeled.mat
```

使用 `--dataset_mode labeled` 时读取 `.mat` 中的 `images` 与 `depths`。

### NYUv2 raw

默认路径为：

```text
NYUv2/raw/<scene_name>/
```

每个 scene 中需要 NYUv2 raw 风格的 `r-*.ppm` RGB 文件和 `d-*.pgm` depth 文件。项目会按时间戳匹配 RGB/depth，并可用 `--raw_max_time_diff`、`--raw_stride`、`--drop_first`、`--drop_last` 控制筛选。

## 生成数据

最常用入口是 `spad-generate`。下面示例从 Middlebury 生成 2 个 64x64、1024-bin 样本：

```bash
spad-generate \
  --dataset_mode middlebury \
  --middlebury_root middlebury/raw \
  --output_dir middlebury/processed_test \
  --surface_model neighborhood_mix \
  --param_idx 10 \
  --res 64 \
  --bins 1024 \
  --limit 2 \
  --save_x \
  --save_clean_transient
```

也可以使用项目根目录的快捷入口，生成后自动打开单样本检查图：

```bash
python main.py \
  --dataset_mode middlebury \
  --middlebury_root middlebury/raw \
  --surface_model neighborhood_mix \
  --output_dir outputs/main_run \
  --limit 1 \
  --display verify
```

如果只想生成数据，不打开可视化：

```bash
python main.py --surface_model volume_scattering --limit 1 --display none
```

### 推荐：从 NYUv2 raw 生成单表面随机 SNR 数据

当前监督训练实验推荐先使用 NYUv2 raw 生成单表面数据。`--param_idx 10` 会在预设信号光子数和 SBR 中随机采样，因此每个样本的信噪比不同，更适合检查模型在 photon-efficient 场景下的鲁棒性：

```bash
spad-generate \
  --dataset_mode raw \
  --raw_root NYUv2/raw \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --surface_model single \
  --param_idx 10 \
  --res 64 \
  --bins 1024 \
  --limit 80 \
  --save_x
```

生成后必须先做数据合理性检查。建议至少看第一个样本和中间样本，确认 RGB/depth 对齐、count map 非空、深度范围合理、直方图峰值位置和深度图趋势一致：

```bash
spad-verify \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --index 0 \
  --output_fig outputs/pipeline_nyuv2_raw_single_random_snr/verify_sample_00000.png

spad-verify \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --index 40 \
  --output_fig outputs/pipeline_nyuv2_raw_single_random_snr/verify_sample_00040.png
```

也可以用交互浏览器逐个检查异常样本：

```bash
spad-browse \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --browse_mode auto \
  --pixel_source auto
```

### 测量模型

| `--surface_model` | 含义 | 主要参数 |
| --- | --- | --- |
| `single` | 单表面 SPAD/LiDAR 模型，调用 DeepInverse `SinglePhotonLidar` | `--irf_sigma` |
| `neighborhood_mix` | 邻域混合，多返回/边缘混合效应 | `--mix_kernel_size`、`--mix_sigma_xy`、`--mix_time_sigma_bins` |
| `translucent_layer` | 半透明前层 + 后景返回 | `--translucent_front_type`、`--translucent_front_depth`、`--translucent_front_signal_ratio`、`--translucent_transmission` |
| `volume_scattering` | 雾或水体散射、背景、多路径影响 | `--volume_medium_type`、`--volume_extinction_coeff`、`--volume_backscatter_ratio`、`--volume_num_steps` |

### 光子参数

`--param_idx` 控制平均信号光子数、背景光子数和 SBR：

| `param_idx` | mean signal | mean background | SBR |
| --- | ---: | ---: | ---: |
| 1 | 10 | 2 | 5 |
| 2 | 5 | 2 | 2.5 |
| 3 | 2 | 2 | 1 |
| 4 | 10 | 10 | 1 |
| 5 | 5 | 10 | 0.5 |
| 6 | 2 | 10 | 0.2 |
| 7 | 10 | 50 | 0.2 |
| 8 | 5 | 50 | 0.1 |
| 9 | 2 | 50 | 0.04 |
| 10 | 从预设信号与 SBR 中随机采样 | 自动计算 | 随机 |

## 输出数据格式

每个样本保存为 `sample_*.npz`，同一目录会生成 `index.csv`。常用字段如下：

| 字段 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `counts` | `(T,H,W)` | Poisson 采样后的 SPAD photon counting cube |
| `depth_m` | `(H,W)` | 米制深度图 |
| `rgb` | `(H,W,3)` | resize 后的 RGB 图 |
| `albedo` | `(H,W)` | 由 RGB 蓝色通道构造的反照率代理 |
| `intensity` | `(H,W)` | 由 RGB 转灰度得到的强度代理 |
| `xhat` | 通常为 `(3,H,W)` | DeepInverse 反投影或项目内 peak-based 估计 |
| `x` | `(3,H,W)`，可选 | 生成模型使用的深度 bin / signal / background 代理，需 `--save_x` |
| `transient_clean` | `(T,H,W)`，可选 | Poisson 采样前的干净瞬态，需 `--save_clean_transient` |
| `surface_model` | scalar string | 生成该样本的测量模型 |
| `bin_size` | scalar | 时间 bin 长度，单位秒 |
| `mean_signal_photons` | scalar | 平均信号光子数 |
| `mean_background_photons` | scalar | 平均背景光子数 |
| `sbr` | scalar | Signal-to-background ratio |

`translucent_layer` 会额外保存 `front_depth_m`、`front_signal`、`back_signal_after_transmission`、`front_tof_bin`、`back_tof_bin` 等字段。`volume_scattering` 会额外保存体散射深度、散射信号、介质后表面信号等字段。

## 检查和浏览

单样本检查会打印字段摘要，并显示 RGB、depth、count map、时间直方图等图：

```bash
spad-verify \
  --dataset_dir middlebury/processed_test \
  --index 0 \
  --output_fig outputs/verify_sample_00000.png
```

交互式浏览器：

```bash
spad-browse \
  --dataset_dir middlebury/processed_test \
  --browse_mode auto \
  --pixel_source auto
```

浏览器按键：

| 按键 | 功能 |
| --- | --- |
| `A` / `D` 或左右方向键 | 上一个 / 下一个样本 |
| `R` | 随机样本 |
| `Home` / `End` | 第一个 / 最后一个样本 |
| `G` | 保存当前浏览画布 |
| `I` | 在终端打印当前样本字段信息 |
| `Q` / `Esc` | 退出 |

## 监督式深度重建训练

入口为 `spad-train`。输入是 `counts`，网络输出每个像素在时间维上的 logits，并通过 soft-argmax 得到预测深度：

```text
input:  (B,1,T,H,W)
output: (B,1,T,H,W)
loss:   KL(target_time_distribution || predicted_time_distribution)
      + tv_weight * TV(predicted_depth)
```

训练脚本会对所有可训练的 Conv/ConvTranspose/Linear 层使用 Kaiming normal 初始化。日志采用表格形式实时输出，每做一次梯度下降、执行 `optimizer.step()` 后，打印当前 step 对应的 KL 损失和 TV 损失；每个 epoch 结束后，会输出训练集与验证集的 `loss`、`KL`、`TV`、`MAE(m)`。早停默认依据验证集 MAE 是否继续改善。

基础示例：

```bash
spad-train \
  --dataset_dir middlebury/processed_test \
  --output_dir outputs/train_simple3d \
  --epochs 20 \
  --batch_size 1 \
  --model simple3d \
  --base_channels 8 \
  --temporal_downsample 16 \
  --tv_weight 0.005
```

`--dataset_dir` 可以重复传入多个目录：

```bash
spad-train \
  --dataset_dir middlebury/processed_single \
  --dataset_dir middlebury/processed_mix \
  --output_dir outputs/train_multi_dataset \
  --model simple3d
```

可选模型：

| `--model` | 说明 | 使用建议 |
| --- | --- | --- |
| `simple3d` | 轻量 3D CNN baseline | 快速验证、消融实验 |
| `prsnet` | PRS-Net 风格结构，含 temporal window、DDFS、pixel-wise residual shrinkage | `T` 经下采样后最好能适配 4 次时间下采样/上采样 |
| `penonlocal` | PENonLocal / DeepBoosting 风格结构，含 3D non-local 与 dense fusion | 显存压力比 `simple3d` 更高 |
| `stin` | DA-STIN 风格 spatio-temporal inception + 7 次时间池化/反卷积 | 更适合 1024-bin transient，建议先用较小 batch |

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--temporal_downsample` | 训练前沿时间维按邻近 bin 求和，降低显存与计算量 |
| `--target_source depth` | 用 `depth_m` 构造高斯时间分布 target，默认选项 |
| `--target_source clean` | 若样本含 `transient_clean`，直接学习干净瞬态分布 |
| `--target_sigma_bins` | 深度 target 高斯宽度，单位为下采样后的 bin |
| `--no_log_counts` | 禁用 `log1p(counts)` 输入压缩 |
| `--tv_weight` | 对预测深度加入 TV 正则 |
| `--max_samples` | 快速调试时限制样本数 |
| `--early_stopping_patience` | 验证集 MAE 连续多少个 epoch 未明显改善后停止 |
| `--early_stopping_min_delta` | 判定 MAE 改善所需的最小变化量，单位米 |

### 当前推荐训练配置

对 1024-bin 数据，当前推荐使用 `--temporal_downsample 4`，即网络实际处理 256 个时间 bin。该设置在显存、速度和时间分辨率之间比较平衡。若需要完全不下采样，可设为 `1`，但显存和训练时间会明显增加。

下面命令使用同一批 NYUv2 raw 单表面随机 SNR 数据训练四个监督模型。`batch_size=4` 是当前实验的默认设置；学习率可以从 `0.003` 起步，如果发现 KL 震荡或验证 MAE 变差，可降到 `0.001`。

Simple3D：

```bash
spad-train \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/train_simple3d_td4_kaiming \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.003 \
  --model simple3d \
  --base_channels 8 \
  --num_blocks 10 \
  --temporal_downsample 4 \
  --tv_weight 0.005 \
  --val_fraction 0.2 \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 0.0001 \
  --num_workers 0
```

PRSNet：

```bash
spad-train \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/train_prsnet_td4_kaiming \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.003 \
  --model prsnet \
  --base_channels 8 \
  --num_blocks 10 \
  --temporal_downsample 4 \
  --tv_weight 0.005 \
  --val_fraction 0.2 \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 0.0001 \
  --num_workers 0
```

PENonLocal：

```bash
spad-train \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/train_penonlocal_td4_kaiming \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.001 \
  --model penonlocal \
  --base_channels 8 \
  --num_blocks 10 \
  --temporal_downsample 4 \
  --tv_weight 0.005 \
  --val_fraction 0.2 \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 0.0001 \
  --num_workers 0
```

STIN / STIM：

```bash
spad-train \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/train_stin_td4_kaiming \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.003 \
  --model stin \
  --base_channels 8 \
  --num_blocks 10 \
  --temporal_downsample 4 \
  --tv_weight 0.005 \
  --val_fraction 0.2 \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 0.0001 \
  --num_workers 0
```

说明：代码入口中的模型名是 `stin`。如果论文或笔记中写作 STIM，本项目仍统一使用命令行参数 `--model stin`。

监督训练快速冒烟测试：

```bash
spad-train \
  --dataset_dir middlebury/processed_test \
  --output_dir outputs/train_supervised_smoke \
  --epochs 1 \
  --batch_size 1 \
  --max_samples 2 \
  --model simple3d \
  --base_channels 2 \
  --temporal_downsample 64 \
  --tv_weight 0.005
```

## 自监督 SPISR 训练

入口为 `spad-train-selfsup`，用于复现 `Single-Photon Image Super-Resolution via Self-Supervised Learning` 的核心思路。它只使用低分辨率 photon cube，不需要高分辨率标签：

```text
L = L_PUKL + alpha * L_equivariance
```

训练示例：

```bash
spad-train-selfsup \
  --dataset_dir middlebury/processed_test \
  --output_dir outputs/train_spisr_selfsup \
  --epochs 20 \
  --batch_size 1 \
  --model spisr \
  --base_channels 16 \
  --num_blocks 4 \
  --temporal_downsample 8 \
  --spatial_downsample 2 \
  --time_scale 2 \
  --spatial_scale 2 \
  --gamma 0.005 \
  --tau 0.001 \
  --alpha 1
```

自监督快速冒烟测试：

```bash
spad-train-selfsup \
  --dataset_dir middlebury/processed_test \
  --output_dir outputs/train_spisr_selfsup_smoke \
  --epochs 1 \
  --batch_size 1 \
  --max_samples 2 \
  --base_channels 4 \
  --num_blocks 1 \
  --temporal_downsample 64 \
  --spatial_downsample 2 \
  --time_scale 2 \
  --spatial_scale 2 \
  --max_shift 2
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--temporal_downsample` | 从原始 `counts` 构造 LR 输入时的时间下采样 |
| `--spatial_downsample` | 从原始 `counts` 构造 LR 输入时的空间下采样 |
| `--time_scale` | 网络输出时间维超分倍率 |
| `--spatial_scale` | 网络输出空间维超分倍率 |
| `--gamma` | PUKL 估计中的 Poisson 噪声参数 |
| `--tau` | PUKL 有限差分扰动尺度 |
| `--alpha` | equivariance KL loss 权重 |
| `--max_shift` | HR cube 时间平移增强的最大 shift |
| `--no_normalize` | 禁用每样本最大值归一化 |

## 推理

`spad-predict` 会根据 checkpoint 中保存的 `method_family` 自动选择监督或自监督推理路径。

监督 checkpoint 输出 `pred_depth_m`，如果样本中有 `depth_m`，还会保存 `target_depth_m` 和 `abs_error_m`：

```bash
spad-predict \
  --checkpoint outputs/train_simple3d/best.pt \
  --sample_file middlebury/processed_test/sample_00000.npz \
  --output_npz outputs/prediction/simple3d_sample_00000.npz \
  --output_fig outputs/prediction/simple3d_sample_00000.png
```

自监督 SPISR checkpoint 会额外输出 `pred_cube`：

```bash
spad-predict \
  --checkpoint outputs/train_spisr_selfsup/best.pt \
  --sample_file middlebury/processed_test/sample_00000.npz \
  --output_npz outputs/prediction/spisr_sample_00000.npz \
  --output_fig outputs/prediction/spisr_sample_00000.png
```

## 批量评估与模型对比

`spad-evaluate` 用于在同一测试集上比较一个或多个监督 checkpoint。它会输出：

- `metrics_summary.json`：每个模型的整体指标。
- `metrics_per_sample.csv`：逐样本指标。
- `comparison.png`：指定样本的 RGB、GT depth、预测 depth、误差图对比。

评价指标：

| 指标 | 含义 |
| --- | --- |
| `MAE(m)` | 平均绝对深度误差，单位米，越低越好 |
| `RMSE(m)` | 均方根深度误差，单位米，越低越好 |
| `AbsRel` | 平均相对绝对误差，越低越好 |

比较 Simple3D 和 STIN：

```bash
spad-evaluate \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_simple3d_td4_kaiming/best.pt \
  --label Simple3D_td4 \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_stin_td4_kaiming/best.pt \
  --label STIN_td4 \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/evaluation_simple3d_stin_td4 \
  --figure_index 0
```

比较 PRSNet 和 PENonLocal：

```bash
spad-evaluate \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_prsnet_td4_kaiming/best.pt \
  --label PRSNet_td4 \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_penonlocal_td4_kaiming/best.pt \
  --label PENonLocal_td4 \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/evaluation_prsnet_penonlocal_td4 \
  --figure_index 0
```

比较四个监督模型：

```bash
spad-evaluate \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_simple3d_td4_kaiming/best.pt \
  --label Simple3D \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_prsnet_td4_kaiming/best.pt \
  --label PRSNet \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_penonlocal_td4_kaiming/best.pt \
  --label PENonLocal \
  --checkpoint outputs/pipeline_nyuv2_raw_single_random_snr/train_stin_td4_kaiming/best.pt \
  --label STIN \
  --dataset_dir outputs/pipeline_nyuv2_raw_single_random_snr/data \
  --output_dir outputs/pipeline_nyuv2_raw_single_random_snr/evaluation_all_td4 \
  --figure_index 0
```

### 已验证的一组实验结果

在 `outputs/pipeline_nyuv2_raw_single_random_snr/data` 的 80 个 NYUv2 raw 单表面随机 SNR 样本上，`temporal_downsample=4`、`batch_size=4`、Kaiming 初始化、`tv_weight=0.005` 的一组测试结果如下。该表用于 sanity check，不同随机种子、数据量和训练轮数会造成数值差异。

| 模型 | MAE(m) | RMSE(m) | AbsRel |
| --- | ---: | ---: | ---: |
| PRSNet | 0.025604 | 0.087287 | 0.003918 |
| PENonLocal | 0.037116 | 0.134057 | 0.006035 |
| Simple3D | 0.067318 | 0.259395 | 0.009251 |
| STIN | 0.061436 | 0.201463 | 0.010740 |

训练时如果看到 KL 长时间停留在接近均匀分布的水平，通常说明输出 logits 学不到有效时间分布。当前监督模型的最后一层保持为线性输出，训练循环内部再做 `log_softmax` / `softmax`，不要在最终 logits 后额外接 ReLU。

## 典型工作流

1. 准备 Middlebury 或 NYUv2 原始数据。
2. 用 `spad-generate` 生成 `.npz` 样本。
3. 用 `spad-verify` 或 `spad-browse` 检查字段、深度图、count map 和时间直方图。
4. 用 `spad-train` 训练监督深度重建模型，或用 `spad-train-selfsup` 训练 SPISR。
5. 用 `spad-evaluate` 批量评估多个 checkpoint，并输出指标和对比图。
6. 用 `spad-predict` 在单个样本上推理并保存预测结果。

## 开发指引

| 目标 | 修改位置 |
| --- | --- |
| 新增数据源读取 | `spimaging/generation/datasets.py`，然后在 `spimaging/generation/pipeline.py` 中接入 |
| 新增测量/生成模型 | `spimaging/generation/models.py`，然后在 `spimaging/generation/pipeline.py` 中添加参数和分支 |
| 新增样本字段 | `spimaging/generation/pipeline.py` 和相关可视化/训练 Dataset |
| 新增监督网络 | `spimaging/training_common/networks.py` 的 `MODEL_REGISTRY` |
| 新增自监督网络 | `spimaging/training_common/networks.py` 的 `SELF_SUPERVISED_MODEL_REGISTRY` |
| 新增损失函数 | `spimaging/training_common/losses.py` |
| 修改监督训练循环 | `spimaging/supervised_training/train.py` |
| 修改自监督训练循环 | `spimaging/self_supervised_training/train.py` |
| 修改推理输出 | `spimaging/testing/predict.py` |

## 常见问题

### `single` 模型提示 DeepInverse 未安装

`--surface_model single` 会调用 `deepinv.physics.SinglePhotonLidar`。请确认已安装训练依赖：

```bash
pip install torch deepinv
```

或使用 `environment.yml` 创建完整环境。

### Middlebury 扫描不到样本

确认目录形如：

```text
middlebury/raw/<scene_name>/view1.png
middlebury/raw/<scene_name>/disp1.png
```

`spad-generate` 只会扫描包含这两个文件的 scene。

### 显存不足

优先尝试：

- 减小 `--batch_size`。
- 增大 `--temporal_downsample`。
- 减小 `--base_channels` 或 `--num_blocks`。
- 先用 `simple3d` 做流程验证，再切到 `prsnet`、`penonlocal` 或 `stin`。

### 训练 target 与输出时间维不一致

训练代码会使用 `match_distribution_shape` 对齐输出和 target，但复杂模型仍建议选择合理的 `--temporal_downsample`。对于 PRS-Net/PENonLocal，建议下采样后的时间维能被多次 2 倍下采样较好地整除；对于 STIN，1024-bin 数据可先从 `--temporal_downsample 8` 或 `16` 试起。

### KL 损失不下降

优先检查以下几项：

- 数据是否合理：用 `spad-verify` 检查直方图峰值、count map、depth 是否一致。
- 学习率是否过高：PENonLocal 对学习率更敏感，`0.003` 不稳定时可降到 `0.001`。
- 输出层是否是线性 logits：最终层不要接 ReLU，否则可能把时间分布压成近似均匀。
- `temporal_downsample` 是否合适：下采样过强会丢失时间细节；下采样过弱会让优化和显存压力变大。
- target 是否过窄或过宽：可调 `--target_sigma_bins`。
- 数据量是否过小：`--limit` 太小只能做流程验证，不适合作为收敛判断。

## 参考模型

- Simple3D: 本项目提供的轻量 3D CNN baseline，用于快速验证 SPAD histogram 到深度重建的监督训练流程。
- PRS-Net: 来源于 `Robust Photon-Efficient Imaging Using PRSNet` / `Robust-Photon-Efficient-Imaging-Using-PRSNet` 风格结构。本项目实现了 temporal window、dense dilated feature stack 和 pixel-wise residual shrinkage block，并适配为输出时间维 logits。
- PENonLocal: 来源于 `Photon-Efficient 3D Imaging with A Non-Local Neural Network` / DeepBoosting 风格结构。本项目实现了 3D non-local block、dense feature fusion 和多层 residual refinement。
- STIN / STIM: 模型来源于 `Deep Domain Adversarial Adaptation for Photon-Efficient Imaging` 中的 spatio-temporal inception network 思路。项目代码中命名为 `stin`，由 ST-inception feature extractor 和 temporal transpose-conv reconstructor 组成，并将最后一层改为线性 logits 以适配当前 KL 监督训练。
- Self-supervised SPISR: 来源于 `Single-Photon Image Super-Resolution via Self-Supervised Learning`，本项目实现 PUKL 与 equivariance 约束的自监督训练入口。
