[CmdletBinding()]
param(
    [string]$EnvironmentName = "",
    [switch]$ReuseEnvironment,
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($EnvironmentName)) {
    $EnvironmentName = "spimaging-day27-$Timestamp"
}

$ResultRoot = Join-Path $ProjectRoot "day27_result_$Timestamp"
$FullLog = Join-Path $ResultRoot "full-validation.log"
$EnvironmentLog = Join-Path $ResultRoot "environment.txt"
$HelpLog = Join-Path $ResultRoot "cli-help.txt"
$PytestLog = Join-Path $ResultRoot "pytest.txt"
$DemoLog = Join-Path $ResultRoot "spad-demo.txt"
$DemoOutput = Join-Path $ResultRoot "demo"
$ReportPath = Join-Path $ResultRoot "second-computer-test-report.md"

New-Item -ItemType Directory -Path $ResultRoot | Out-Null

function Write-ValidationLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $FullLog -Append
}

function Find-CondaExecutable {
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $command -and $command.CommandType -eq "Application") {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "Conda was not found. Install Miniconda/Anaconda and rerun this script."
}

function Invoke-CondaLogged {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [string]$StepLog = $FullLog
    )

    Write-ValidationLog "$Label"
    Write-ValidationLog ("conda " + ($Arguments -join " "))
    if ([System.IO.Path]::GetFullPath($StepLog) -eq [System.IO.Path]::GetFullPath($FullLog)) {
        & $script:CondaExecutable @Arguments 2>&1 | Tee-Object -FilePath $FullLog -Append
    }
    else {
        & $script:CondaExecutable @Arguments 2>&1 | Tee-Object -FilePath $StepLog -Append | Tee-Object -FilePath $FullLog -Append
    }
    $exitCode = $LASTEXITCODE
    Write-ValidationLog "$Label exit code: $exitCode"
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode."
    }
}

function Verify-PackageManifest {
    $manifestPath = Join-Path $ProjectRoot "PACKAGE_SHA256SUMS.csv"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        Write-ValidationLog "PACKAGE_SHA256SUMS.csv is absent; package hash verification skipped."
        return
    }

    $rows = Import-Csv -LiteralPath $manifestPath
    foreach ($row in $rows) {
        $filePath = Join-Path $ProjectRoot $row.Path
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
            throw "Package file is missing: $($row.Path)"
        }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $filePath).Hash.ToLowerInvariant()
        if ($actual -ne $row.SHA256.ToLowerInvariant()) {
            throw "Package hash mismatch: $($row.Path)"
        }
    }
    Write-ValidationLog "Package manifest verified: $($rows.Count) files."
}

$StartedAt = Get-Date
$Status = "FAIL"
$FailureReason = ""
$FreshEnvironment = $false
$PytestSummary = "not run"
$DemoStatus = "not run"
$DemoDuration = ""
$ComputerName = $env:COMPUTERNAME
$OperatingSystem = "unknown"
$Processor = "unknown"
$Graphics = "unknown"

try {
    Set-Location -LiteralPath $ProjectRoot
    Write-ValidationLog "SPImaging Day 27 independent validation started."
    Write-ValidationLog "Project root: $ProjectRoot"
    Write-ValidationLog "Requested environment: $EnvironmentName"

    Verify-PackageManifest
    $script:CondaExecutable = Find-CondaExecutable
    Write-ValidationLog "Conda executable: $script:CondaExecutable"

    $environmentListing = & $script:CondaExecutable env list --json | ConvertFrom-Json
    $existingEnvironment = $environmentListing.envs | Where-Object {
        (Split-Path -Leaf $_) -eq $EnvironmentName
    }
    if ($existingEnvironment -and -not $ReuseEnvironment) {
        throw "Conda environment '$EnvironmentName' already exists. Use another name; a fresh environment is required."
    }

    if (-not $existingEnvironment) {
        Invoke-CondaLogged -Label "Create fresh Conda environment" -Arguments @(
            "env", "create", "-n", $EnvironmentName, "-f", (Join-Path $ProjectRoot "environment.yml")
        )
        $FreshEnvironment = $true
    }
    else {
        Write-ValidationLog "Reusing existing environment because -ReuseEnvironment was supplied."
    }

    if (-not $SkipDependencyInstall) {
        Invoke-CondaLogged -Label "Install CPU PyTorch" -Arguments @(
            "run", "-n", $EnvironmentName, "python", "-m", "pip", "install",
            "numpy<2", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"
        )
        Invoke-CondaLogged -Label "Install DeepInverse" -Arguments @(
            "run", "-n", $EnvironmentName, "python", "-m", "pip", "install", "deepinv"
        )
        Invoke-CondaLogged -Label "Refresh editable installation" -Arguments @(
            "run", "-n", $EnvironmentName, "python", "-m", "pip", "install", "--editable", "."
        )
    }
    else {
        Write-ValidationLog "Dependency installation skipped because -SkipDependencyInstall was supplied."
    }

    Invoke-CondaLogged -Label "Check installed dependencies" -Arguments @(
        "run", "-n", $EnvironmentName, "python", "-m", "pip", "check"
    )

    $environmentCode = (@"
import platform, sys
import numpy, pytest, torch
import spimaging
print('computer=$ComputerName')
print('platform=' + platform.platform())
print('python=' + sys.version.replace(chr(10), ' '))
print('executable=' + sys.executable)
print('spimaging=' + spimaging.__version__)
print('numpy=' + numpy.__version__)
print('pytest=' + pytest.__version__)
print('torch=' + torch.__version__)
print('cuda_available=' + str(torch.cuda.is_available()))
print('device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
"@).Trim() -replace "`r?`n", ";"
    Invoke-CondaLogged -Label "Record runtime environment" -Arguments @(
        "run", "-n", $EnvironmentName, "python", "-c", $environmentCode
    ) -StepLog $EnvironmentLog

    $commands = @(
        "spad-generate", "spad-verify", "spad-browse", "spad-train",
        "spad-train-selfsup", "spad-predict", "spad-evaluate", "spad-demo"
    )
    foreach ($commandName in $commands) {
        Invoke-CondaLogged -Label "Check $commandName --help" -Arguments @(
            "run", "-n", $EnvironmentName, $commandName, "--help"
        ) -StepLog $HelpLog
    }

    Invoke-CondaLogged -Label "Run complete pytest suite" -Arguments @(
        "run", "-n", $EnvironmentName, "python", "-m", "pytest", "-q"
    ) -StepLog $PytestLog
    $PytestSummary = (Get-Content -LiteralPath $PytestLog | Where-Object { $_ -match "passed" } | Select-Object -Last 1).Trim()

    Invoke-CondaLogged -Label "Run complete SPImaging demo" -Arguments @(
        "run", "-n", $EnvironmentName, "spad-demo",
        "--dataset_dir", (Join-Path $ProjectRoot "example_data\nyuv2_raw_single_random_snr"),
        "--output_dir", $DemoOutput
    ) -StepLog $DemoLog

    $summaryPath = Join-Path $DemoOutput "demo_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Demo summary was not created."
    }
    $demoSummary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    if ($demoSummary.status -ne "success") {
        throw "Demo summary status is '$($demoSummary.status)', expected 'success'."
    }
    $DemoStatus = $demoSummary.status
    $DemoDuration = $demoSummary.total_duration_seconds

    $requiredArtifacts = @(
        "verify\sample_00000.png",
        "train\last.pt",
        "train\best.pt",
        "predict\prediction.npz",
        "predict\comparison.png",
        "evaluate\metrics_per_sample.csv",
        "evaluate\metrics_summary.json",
        "evaluate\comparison.png",
        "demo.log",
        "demo_summary.json"
    )
    foreach ($relativePath in $requiredArtifacts) {
        $artifact = Join-Path $DemoOutput $relativePath
        if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
            throw "Required demo artifact is missing: $relativePath"
        }
    }

    try {
        $OperatingSystem = (Get-CimInstance Win32_OperatingSystem).Caption
        $Processor = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
        $Graphics = ((Get-CimInstance Win32_VideoController).Name -join "; ")
    }
    catch {
        Write-ValidationLog "Hardware query warning: $($_.Exception.Message)"
    }

    $Status = "PASS"
    Write-ValidationLog "Independent validation completed successfully."
}
catch {
    $FailureReason = $_.Exception.Message
    Write-ValidationLog "VALIDATION FAILED: $FailureReason"
}
finally {
    $FinishedAt = Get-Date
    $Elapsed = [math]::Round(($FinishedAt - $StartedAt).TotalSeconds, 3)
    $report = @"
# SPImaging Day 27 第二台电脑独立测试报告

- 结论：**$Status**
- 计算机名：`$ComputerName`
- 操作系统：$OperatingSystem
- 处理器：$Processor
- 显卡：$Graphics
- Conda 环境：`$EnvironmentName`
- 是否由脚本新建环境：$FreshEnvironment
- 开始时间：$($StartedAt.ToString("yyyy-MM-dd HH:mm:ss zzz"))
- 结束时间：$($FinishedAt.ToString("yyyy-MM-dd HH:mm:ss zzz"))
- 总耗时：$Elapsed 秒
- pytest：$PytestSummary
- 完整演示状态：$DemoStatus
- 完整演示内部耗时：$DemoDuration 秒
- 失败原因：$FailureReason

## 测试约束

测试者仅运行验收包中的 `scripts/run_day27_validation.ps1`，无需向开发者询问操作步骤。脚本负责环境建立、依赖安装、入口检查、pytest、完整演示和产物核验。

## 回传文件

请将整个 `$([System.IO.Path]::GetFileName($ResultRoot))` 目录压缩后回传。详细命令和原始输出见 `full-validation.log`。
"@
    Set-Content -LiteralPath $ReportPath -Value $report -Encoding utf8

    Get-ChildItem -LiteralPath $ResultRoot -File -Recurse |
        Where-Object { $_.FullName -ne (Join-Path $ResultRoot "result-file-hashes.csv") } |
        ForEach-Object {
            [pscustomobject]@{
                Path = $_.FullName.Substring($ResultRoot.Length + 1)
                Bytes = $_.Length
                SHA256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        } |
        Export-Csv -LiteralPath (Join-Path $ResultRoot "result-file-hashes.csv") -NoTypeInformation -Encoding utf8
}

Write-Host ""
Write-Host "Day 27 result: $Status"
Write-Host "Result directory: $ResultRoot"
if ($Status -ne "PASS") {
    exit 1
}
