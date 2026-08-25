param(
    [string]$PythonExe = "",
    [string]$OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "record_of_SPI\Day_13-14\异常提示截图"
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $environmentPython = Join-Path $env:USERPROFILE "anaconda3\envs\spimaging\python.exe"
    if (Test-Path -LiteralPath $environmentPython -PathType Leaf) {
        $PythonExe = $environmentPython
    }
    else {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Add-Type -AssemblyName System.Drawing

$cases = @(
    @{
        File = "01_res范围错误.png"
        Title = "Invalid resolution"
        Display = "spad-generate --res 0"
        Arguments = @("-m", "spimaging.generation.pipeline", "--res", "0")
    },
    @{
        File = "02_param_idx范围错误.png"
        Title = "Invalid parameter index"
        Display = "spad-generate --param_idx 11"
        Arguments = @("-m", "spimaging.generation.pipeline", "--param_idx", "11")
    },
    @{
        File = "03_输入目录不存在.png"
        Title = "Missing dataset directory"
        Display = "spad-verify --dataset_dir missing_data"
        Arguments = @("-m", "spimaging.testing.verify", "--dataset_dir", "missing_data")
    },
    @{
        File = "04_互斥参数冲突.png"
        Title = "Conflicting sample selectors"
        Display = "spad-verify --dataset_dir data --index 0 --random"
        Arguments = @(
            "-m", "spimaging.testing.verify",
            "--dataset_dir", "data",
            "--index", "0", "--random"
        )
    },
    @{
        File = "05_checkpoint不存在.png"
        Title = "Missing checkpoint"
        Display = "spad-predict --checkpoint missing.pt --sample_file sample.npz"
        Arguments = @(
            "-m", "spimaging.testing.predict",
            "--checkpoint", "missing.pt",
            "--sample_file", "sample.npz"
        )
    }
)

function Invoke-CliCase {
    param([hashtable]$Case)

    $quotedArguments = ($Case.Arguments | ForEach-Object {
        '"' + ($_ -replace '"', '\"') + '"'
    }) -join " "

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $PythonExe
    $startInfo.Arguments = $quotedArguments
    $startInfo.WorkingDirectory = $repoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [Text.Encoding]::UTF8
    $startInfo.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"
    $startInfo.EnvironmentVariables["MPLBACKEND"] = "Agg"
    $startInfo.EnvironmentVariables["MPLCONFIGDIR"] = Join-Path (
        [IO.Path]::GetTempPath()
    ) "spimaging-cli-screenshot-matplotlib"

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw "Failed to start $($Case.Display)"
        }
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        $exitCode = $process.ExitCode
    }
    finally {
        $process.Dispose()
    }

    $outputParts = @()
    if (-not [string]::IsNullOrWhiteSpace($stdout)) {
        $outputParts += $stdout.TrimEnd()
    }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        $outputParts += $stderr.TrimEnd()
    }

    [pscustomobject]@{
        ExitCode = $exitCode
        Output = $outputParts -join [Environment]::NewLine
    }
}

function Save-TerminalImage {
    param(
        [hashtable]$Case,
        [pscustomobject]$Result,
        [string]$Path
    )

    $lines = New-Object System.Collections.Generic.List[object]
    $lines.Add([pscustomobject]@{ Text = "SPImaging CLI validation"; Tone = "yellow" })
    $lines.Add([pscustomobject]@{ Text = "PS SPImaging> $($Case.Display)"; Tone = "cyan" })
    $lines.Add([pscustomobject]@{ Text = ""; Tone = "text" })
    foreach ($line in ($Result.Output -split "\r?\n")) {
        $lines.Add([pscustomobject]@{ Text = $line; Tone = "text" })
    }
    $lines.Add([pscustomobject]@{ Text = ""; Tone = "text" })
    $lines.Add([pscustomobject]@{ Text = "Exit code: $($Result.ExitCode)"; Tone = "yellow" })
    $lines.Add([pscustomobject]@{ Text = "PS SPImaging>"; Tone = "text" })

    $width = 1180
    $titleHeight = 48
    $lineHeight = 28
    $height = [Math]::Max(320, $titleHeight + 24 + $lines.Count * $lineHeight + 24)
    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $font = New-Object System.Drawing.Font(
        "Consolas", 18, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel
    )
    $titleFont = New-Object System.Drawing.Font(
        "Segoe UI", 17, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel
    )
    $backgroundBrush = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(12, 12, 12))
    $titleBrush = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(35, 35, 35))
    $titleTextBrush = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(190, 190, 190))
    $textBrush = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(232, 232, 232))
    $yellowBrush = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(250, 235, 125))
    $cyanBrush = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(80, 220, 225))

    try {
        $graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::ClearTypeGridFit
        $graphics.FillRectangle($backgroundBrush, 0, 0, $width, $height)
        $graphics.FillRectangle($titleBrush, 0, 0, $width, $titleHeight)
        $graphics.DrawString(
            "SPImaging CLI - $($Case.Title)",
            $titleFont,
            $titleTextBrush,
            18,
            13
        )

        $y = $titleHeight + 18
        foreach ($line in $lines) {
            $brush = switch ($line.Tone) {
                "yellow" { $yellowBrush }
                "cyan" { $cyanBrush }
                default { $textBrush }
            }
            $graphics.DrawString($line.Text, $font, $brush, 18, $y)
            $y += $lineHeight
        }

        $bitmap.Save($Path, [Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $cyanBrush.Dispose()
        $yellowBrush.Dispose()
        $textBrush.Dispose()
        $titleTextBrush.Dispose()
        $titleBrush.Dispose()
        $backgroundBrush.Dispose()
        $titleFont.Dispose()
        $font.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

foreach ($case in $cases) {
    $result = Invoke-CliCase -Case $case
    if ($result.ExitCode -ne 2) {
        throw "$($case.Display) returned $($result.ExitCode), expected 2"
    }
    if ($result.Output -match "Traceback \(most recent call last\)") {
        throw "$($case.Display) unexpectedly produced a Python traceback"
    }
    if ($result.Output -notmatch "--help") {
        throw "$($case.Display) did not include the expected --help hint"
    }

    $outputPath = Join-Path $OutputDir $case.File
    Save-TerminalImage -Case $case -Result $result -Path $outputPath
    Write-Output $outputPath
}

Write-Host "Saved $($cases.Count) background-rendered CLI screenshots to: $OutputDir"
