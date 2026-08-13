# 异常提示截图

以下截图由 `scripts/capture_cli_errors.ps1` 在后台执行真实命令并渲染生成，不会打开窗口或抢占桌面焦点。每个命令均应以状态码 2 退出，给出简短错误原因和 `--help` 提示，且不显示 Python traceback。

| 截图 | 命令 | 验证点 |
| --- | --- | --- |
| `01_res范围错误.png` | `spad-generate --res 0` | 正整数下界 |
| `02_param_idx范围错误.png` | `spad-generate --param_idx 11` | 参数索引范围 1–10 |
| `03_输入目录不存在.png` | `spad-verify --dataset_dir missing_data` | 输入目录必须存在 |
| `04_互斥参数冲突.png` | `spad-verify --dataset_dir data --index 0 --random` | 样本选择参数互斥 |
| `05_checkpoint不存在.png` | `spad-predict --checkpoint missing.pt --sample_file sample.npz` | 输入 checkpoint 必须存在 |

复现命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/capture_cli_errors.ps1
```
