[CmdletBinding()]
param(
    [string]$Version = "0.2.0-beta.1",
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\out"),
    [switch]$WithoutCuda,
    [string]$SignerThumbprint
)

$ErrorActionPreference = "Stop"
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
$cpu = Join-Path $outputRoot "spimaging-runtime-cpu-$Version.zip"
$cuda = Join-Path $outputRoot "spimaging-runtime-cuda-$Version.zip"
$app = Join-Path $outputRoot "spimaging-app-$Version.zip"
$manifest = Join-Path $outputRoot "spimaging-release-manifest.json"

$arguments = @(
    (Join-Path $PSScriptRoot "build_release_manifest.py"),
    "--version", $Version,
    "--base-url", $BaseUrl,
    "--cpu-archive", $cpu,
    "--app-archive", $app,
    "--output", $manifest
)
if (-not $WithoutCuda) { $arguments += @("--cuda-archive", $cuda) }
if ($SignerThumbprint) {
    $arguments += @("--signer-thumbprint", $SignerThumbprint)
} else {
    if ($Version -notmatch "-") {
        throw "A stable version requires -SignerThumbprint and detached CMS signatures."
    }
    $arguments += "--unsigned-beta"
}

& python @arguments
if ($LASTEXITCODE -ne 0) { throw "release manifest generation failed" }
& python (Join-Path $PSScriptRoot "verify_release_manifest.py") $manifest --asset-dir $outputRoot
if ($LASTEXITCODE -ne 0) { throw "release manifest dry-run failed" }
Write-Host "Built and verified $manifest"
