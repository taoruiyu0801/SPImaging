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
