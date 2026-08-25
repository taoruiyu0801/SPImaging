[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\runtime\locks")
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$condaLock = Get-Command conda-lock -ErrorAction SilentlyContinue
if (-not $condaLock) {
    throw "Missing prerequisite: conda-lock. Install it on the build host; it is never required on an end-user machine."
}

foreach ($variant in @("cpu", "cuda")) {
    $inputFile = Join-Path $repoRoot "packaging\runtime\environment-$variant.in.yml"
    $lockFile = Join-Path $outputRoot "environment-$variant.conda-lock.yml"
    & $condaLock.Source lock --file $inputFile --platform win-64 --lockfile $lockFile
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $lockFile -PathType Leaf)) {
        throw "conda-lock failed for $variant"
    }
    $hash = (Get-FileHash -LiteralPath $lockFile -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath "$lockFile.sha256" -Value "$hash  $([IO.Path]::GetFileName($lockFile))" -Encoding ascii
}

Write-Host "Runtime locks written to $outputRoot"
