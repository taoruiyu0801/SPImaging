[CmdletBinding()]
param(
    [string]$Version = "0.2.0-beta.1",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\out")
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$archive = Join-Path $outputRoot "spimaging-app-$Version.zip"

& python (Join-Path $PSScriptRoot "build_app_asset.py") --repo $repoRoot --output $archive
if ($LASTEXITCODE -ne 0) { throw "deterministic app asset build failed" }
& python (Join-Path $PSScriptRoot "split_asset.py") $archive $outputRoot
if ($LASTEXITCODE -ne 0) { throw "app asset splitting failed" }

Write-Host "Built $archive"
