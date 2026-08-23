[CmdletBinding()]
param(
    [string]$Version = "0.2.0-beta.1",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\out"),
    [string]$IsccPath,
    [string]$InnoSignToolCommand,
    [string]$SigningPfxPath,
    [string]$CertificateThumbprint
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (-not $IsccPath) {
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) { $IsccPath = $command.Source }
}
if (-not $IsccPath) {
    $common = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"
    if (Test-Path -LiteralPath $common) { $IsccPath = $common }
}
if (-not $IsccPath -or -not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) {
    throw "Missing external prerequisite: Inno Setup 6 ISCC.exe. Launcher and release assets remain usable."
}
$launcher = Join-Path $outputRoot "launcher\SPImaging.exe"
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Missing launcher artifact: $launcher. Run Build-Launcher.ps1 first."
}
if ($InnoSignToolCommand -and $SigningPfxPath) {
    throw "Choose either -InnoSignToolCommand or -SigningPfxPath, not both."
}
if ($SigningPfxPath -and -not $CertificateThumbprint) {
    throw "-SigningPfxPath requires -CertificateThumbprint for publisher pinning."
}

$definition = Join-Path $repoRoot "packaging\inno\SPImaging.iss"
$arguments = @(
    "/DMyAppVersion=$Version",
    "/DMySourceRoot=$outputRoot",
    "/DMyRepoRoot=$repoRoot"
)
if ($InnoSignToolCommand) {
    $arguments += @("/DMySignedBuild=1", "/Sspimaging=$InnoSignToolCommand")
} else {
    $arguments += "/DMySignedBuild=0"
}
$arguments += $definition
& $IsccPath @arguments
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed" }
if ($SigningPfxPath) {
    $unsignedInstaller = Join-Path $outputRoot "SPImaging-Setup-unsigned-beta.exe"
    $signedInstaller = Join-Path $outputRoot "SPImaging-Setup.exe"
    if (-not (Test-Path -LiteralPath $unsignedInstaller -PathType Leaf)) {
        throw "Expected unsigned installer output is missing: $unsignedInstaller"
    }
    & (Join-Path $PSScriptRoot "Sign-Artifact.ps1") `
        -Artifact $unsignedInstaller `
        -Mode Authenticode `
        -PfxPath ([IO.Path]::GetFullPath($SigningPfxPath)) `
        -CertificateThumbprint $CertificateThumbprint
    if ($LASTEXITCODE -ne 0) { throw "Installer Authenticode signing failed" }
    Move-Item -LiteralPath $unsignedInstaller -Destination $signedInstaller -Force
    if (-not (Test-Path -LiteralPath $signedInstaller -PathType Leaf)) {
        throw "Signed installer finalization failed: $signedInstaller"
    }
    Write-Host "Built signed installer $signedInstaller"
} elseif ($InnoSignToolCommand) {
    $signedInstaller = Join-Path $outputRoot "SPImaging-Setup.exe"
    if (-not (Test-Path -LiteralPath $signedInstaller -PathType Leaf)) {
        throw "Inno did not produce the expected signed installer: $signedInstaller"
    }
    Write-Host "Built signed installer $signedInstaller"
} else {
    $unsignedInstaller = Join-Path $outputRoot "SPImaging-Setup-unsigned-beta.exe"
    if (-not (Test-Path -LiteralPath $unsignedInstaller -PathType Leaf)) {
        throw "Inno did not produce the expected unsigned beta installer: $unsignedInstaller"
    }
    Write-Host "Built unsigned beta installer $unsignedInstaller"
}
