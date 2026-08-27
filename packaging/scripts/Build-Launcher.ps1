[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\out\launcher"),
    [string]$PythonPath
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)

if (-not $PythonPath) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { $pythonCommand = Get-Command python -ErrorAction SilentlyContinue }
    if ($pythonCommand) { $PythonPath = $pythonCommand.Source }
}
if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Missing build Python. Pass -PythonPath with the absolute path to python.exe."
}
$PythonPath = [IO.Path]::GetFullPath($PythonPath)

& $PythonPath -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Missing prerequisite: PyInstaller in build Python $PythonPath."
}
$prefixOutput = & $PythonPath -c "import sys; print(sys.prefix)"
if ($LASTEXITCODE -ne 0 -or -not $prefixOutput) {
    throw "Unable to resolve sys.prefix from build Python $PythonPath."
}
$pythonPrefix = [IO.Path]::GetFullPath([string]($prefixOutput | Select-Object -Last 1))

$tclLibrary = @(
    (Join-Path $pythonPrefix "Library\lib\tcl8.6"),
    (Join-Path $pythonPrefix "tcl\tcl8.6")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -First 1
$tkLibrary = @(
    (Join-Path $pythonPrefix "Library\lib\tk8.6"),
    (Join-Path $pythonPrefix "tcl\tk8.6")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -First 1
if (-not $tclLibrary -or -not $tkLibrary) {
    throw "Build Python does not contain a complete Tcl/Tk runtime under $pythonPrefix."
}

$environmentNames = @(
    "PATH",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONUSERBASE",
    "PYTHONNOUSERSITE",
    "VIRTUAL_ENV",
    "TCL_LIBRARY",
    "TK_LIBRARY",
    "_CONDA_EXE",
    "_CE_CONDA",
    "_CE_M"
)
$environmentNames += Get-ChildItem Env: |
    Where-Object { $_.Name -like "CONDA_*" } |
    Select-Object -ExpandProperty Name
$environmentNames = $environmentNames | Select-Object -Unique
$originalEnvironment = @{}
foreach ($name in $environmentNames) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$systemRoot = [Environment]::GetEnvironmentVariable("SystemRoot", "Process")
$cleanPathParts = @(
    $pythonPrefix,
    (Join-Path $pythonPrefix "Scripts"),
    (Join-Path $pythonPrefix "DLLs"),
    (Join-Path $pythonPrefix "Library\bin"),
    (Join-Path $systemRoot "System32"),
    $systemRoot,
    (Join-Path $systemRoot "System32\Wbem"),
    (Join-Path $systemRoot "System32\WindowsPowerShell\v1.0")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } | Select-Object -Unique

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$workRoot = Join-Path $outputRoot "pyinstaller-work"
$spec = Join-Path $repoRoot "packaging\launcher\SPImagingLauncher.spec"
$launcher = Join-Path $outputRoot "SPImaging.exe"

try {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
    $env:PATH = [string]::Join([IO.Path]::PathSeparator, $cleanPathParts)
    $env:PYTHONNOUSERSITE = "1"
    $env:TCL_LIBRARY = [IO.Path]::GetFullPath($tclLibrary)
    $env:TK_LIBRARY = [IO.Path]::GetFullPath($tkLibrary)

    Write-Host "Building launcher with isolated Python $PythonPath"
    Write-Host "Tcl library: $env:TCL_LIBRARY"
    Write-Host "Tk library:  $env:TK_LIBRARY"
    & $PythonPath -m PyInstaller --noconfirm --clean --distpath $outputRoot --workpath $workRoot $spec
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
        throw "PyInstaller launcher build failed"
    }

    $selfTest = Start-Process `
        -FilePath $launcher `
        -ArgumentList "--launcher-tcl-self-test" `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($selfTest.ExitCode -ne 0) {
        throw "Frozen launcher Tcl/Tk self-test failed with exit code $($selfTest.ExitCode). Refusing to publish this build."
    }
} finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], "Process")
    }
}
Write-Host "Built and Tcl-tested $launcher"
