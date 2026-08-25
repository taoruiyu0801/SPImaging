[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\out\launcher")
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Missing prerequisite: PyInstaller in the build Python environment."
}
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$workRoot = Join-Path $outputRoot "pyinstaller-work"
$spec = Join-Path $repoRoot "packaging\launcher\SPImagingLauncher.spec"

python -m PyInstaller --noconfirm --clean --distpath $outputRoot --workpath $workRoot $spec
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $outputRoot "SPImaging.exe") -PathType Leaf)) {
    throw "PyInstaller launcher build failed"
}
Write-Host "Built $(Join-Path $outputRoot 'SPImaging.exe')"
