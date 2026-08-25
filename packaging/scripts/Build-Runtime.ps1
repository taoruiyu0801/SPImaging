[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cpu", "cuda")]
    [string]$Variant,
    [string]$Version = "0.2.0-beta.1",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\out"),
    [string]$LockDirectory = (Join-Path $PSScriptRoot "..\runtime\locks")
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
$lockFile = [IO.Path]::GetFullPath((Join-Path $LockDirectory "environment-$Variant.conda-lock.yml"))
if (-not (Test-Path -LiteralPath $lockFile -PathType Leaf)) {
    throw "Missing locked runtime: $lockFile. Run packaging/scripts/New-RuntimeLocks.ps1 first."
}
foreach ($tool in @("conda-lock", "conda-pack", "python")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing build prerequisite: $tool"
    }
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$prefix = [IO.Path]::GetFullPath((Join-Path $tempBase "SPImaging-runtime-$Variant-$([Guid]::NewGuid().ToString('N'))"))
if (-not $prefix.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing unsafe temporary prefix: $prefix"
}
$archive = Join-Path $outputRoot "spimaging-runtime-$Variant-$Version.zip"

try {
    & conda-lock install --prefix $prefix $lockFile
    if ($LASTEXITCODE -ne 0) { throw "conda-lock install failed for $Variant" }
    $runtimePython = Join-Path $prefix "python.exe"
    if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
        throw "locked runtime is missing python.exe"
    }
    # conda-lock 4.0.2 can resolve but silently skip the pip section of a
    # unified Windows lock. Render the same lock and install its hash-pinned
    # wheel URLs explicitly before the runtime is packed.
    $renderTemplate = Join-Path $prefix "spimaging-pip-{platform}.lock"
    & conda-lock render --kind explicit --filename-template $renderTemplate $lockFile
    if ($LASTEXITCODE -ne 0) { throw "could not render pip entries for $Variant" }
    $renderedLock = $renderTemplate.Replace("{platform}", "win-64")
    $pipArtifacts = @(
        Get-Content -LiteralPath $renderedLock | ForEach-Object {
            if ($_ -match '^# pip ([A-Za-z0-9_.-]+) @ (https://.+#sha256=[0-9a-f]{64})$') {
                [pscustomobject]@{ Name = $Matches[1]; Url = $Matches[2] }
            }
        }
    )
    if ($pipArtifacts.Count -eq 0) { throw "locked runtime has no rendered pip artifacts for $Variant" }
    $missingPipUrls = @(
        foreach ($artifact in $pipArtifacts) {
            & $runtimePython -c "import importlib.metadata as m; m.version('$($artifact.Name)')" 2>$null
            if ($LASTEXITCODE -ne 0) { $artifact.Url }
        }
    )
    if ($missingPipUrls.Count -gt 0) {
        & $runtimePython -m pip install --disable-pip-version-check --no-deps @missingPipUrls
        if ($LASTEXITCODE -ne 0) { throw "locked pip dependency installation failed for $Variant" }
    }
    [System.IO.File]::Delete($renderedLock)
    $healthScript = @'
import importlib.metadata as metadata
import PySide6
import deepinv
import torch

required = {
    "deepinv": "0.4.1",
    "torch": "2.5.1",
}
for name, expected in required.items():
    actual = metadata.version(name)
    if actual != expected:
        raise SystemExit(f"{name} version mismatch: expected {expected}, got {actual}")
'@
    & $runtimePython -c $healthScript
    if ($LASTEXITCODE -ne 0) { throw "locked runtime dependency health check failed for $Variant" }
    if ($Variant -eq "cpu") {
        & $runtimePython -c "import torch; assert torch.version.cuda is None, torch.version.cuda"
    }
    else {
        & $runtimePython -c "import torch; assert torch.version.cuda, 'CUDA runtime resolved to a CPU-only PyTorch build'"
    }
    if ($LASTEXITCODE -ne 0) { throw "locked runtime accelerator health check failed for $Variant" }
    & conda-pack -p $prefix -o $archive --format zip --force
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw "conda-pack failed for $Variant"
    }
    & python (Join-Path $PSScriptRoot "split_asset.py") $archive $outputRoot
    if ($LASTEXITCODE -ne 0) { throw "runtime asset splitting failed" }
    $metadata = [ordered]@{
        schema_version = 1
        variant = $Variant
        version = $Version
        lock_sha256 = (Get-FileHash -LiteralPath $lockFile -Algorithm SHA256).Hash.ToLowerInvariant()
        archive_sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        built_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    $metadata | ConvertTo-Json | Set-Content -LiteralPath "$archive.build.json" -Encoding utf8
}
finally {
    $resolvedPrefix = [IO.Path]::GetFullPath($prefix)
    if ($resolvedPrefix.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase) -and
        ([IO.Path]::GetFileName($resolvedPrefix) -like "SPImaging-runtime-$Variant-*")) {
        if (Test-Path -LiteralPath $resolvedPrefix) {
            Remove-Item -LiteralPath $resolvedPrefix -Recurse -Force
        }
    }
}

Write-Host "Built $archive"
