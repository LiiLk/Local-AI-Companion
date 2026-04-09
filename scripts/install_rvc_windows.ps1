param(
    [string]$PythonExe = ".\venv\Scripts\python.exe",
    [string]$InstallDir = ".\.rvc-overlay",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$installPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $InstallDir))
$requirementsFile = Join-Path (Get-Location) "scripts\requirements-rvc-worker.txt"
if (-not (Test-Path $requirementsFile)) {
    throw "RVC requirements file not found: $requirementsFile"
}

$stagingPath = "$installPath.staging"
if ($Clean -and (Test-Path $installPath)) {
    Remove-Item -LiteralPath $installPath -Recurse -Force
}
if (Test-Path $stagingPath) {
    Remove-Item -LiteralPath $stagingPath -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingPath -Force | Out-Null

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found. Install Visual Studio 2022 Build Tools with C++ support."
}

$sdkIncludeRoot = "C:\Program Files (x86)\Windows Kits\10\Include"
$sdkVersion = Get-ChildItem $sdkIncludeRoot -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1 -ExpandProperty Name

if (-not $sdkVersion) {
    throw "Windows SDK include directory not found under $sdkIncludeRoot"
}

$msvcRoot = Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC" -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $msvcRoot) {
    throw "MSVC toolchain not found under Visual Studio Build Tools."
}

Write-Host "Using Python: $PythonExe"
Write-Host "Installing isolated RVC packages into staging dir: $stagingPath"
Write-Host "Using Windows SDK: $sdkVersion"
Write-Host "Using MSVC root: $msvcRoot"

& $PythonExe -m pip install --upgrade "pip<24.1"

$cmdPath = Join-Path (Get-Location) ".install_rvc_windows.cmd"
$cmd = @"
call "$vcvars"
set DISTUTILS_USE_SDK=1
set MSSdk=1
set PATH=$msvcRoot\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\$sdkVersion\x64;%PATH%
set INCLUDE=$msvcRoot\include;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\VS\include;C:\Program Files (x86)\Windows Kits\10\Include\$sdkVersion\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\$sdkVersion\um;C:\Program Files (x86)\Windows Kits\10\Include\$sdkVersion\shared;C:\Program Files (x86)\Windows Kits\10\Include\$sdkVersion\winrt;C:\Program Files (x86)\Windows Kits\10\Include\$sdkVersion\cppwinrt
set LIB=$msvcRoot\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\$sdkVersion\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\$sdkVersion\um\x64
set LIBPATH=$msvcRoot\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\$sdkVersion\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\$sdkVersion\um\x64
"$PythonExe" -m pip install --upgrade --target "$stagingPath" -r "$requirementsFile"
"$PythonExe" -m pip install --upgrade --target "$stagingPath" "inferrvc==1.0" --no-deps
"@

Set-Content -Path $cmdPath -Value $cmd -Encoding ASCII

try {
    & "C:\Windows\System32\cmd.exe" /d /s /c $cmdPath
    if ($LASTEXITCODE -ne 0) {
        throw "RVC dependency installation failed with exit code $LASTEXITCODE"
    }

    # Keep using the project's CUDA-enabled torch/torchaudio from the main venv.
    # The wheels resolved for --target are often CPU-only on Windows and can make
    # InferRVC crash during import with "Torch not compiled with CUDA enabled".
    $bundledRuntimeDirs = @("torch", "torchaudio", "torchgen", "functorch")
    foreach ($name in $bundledRuntimeDirs) {
        $path = Join-Path $stagingPath $name
        if (Test-Path $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
    Get-ChildItem -LiteralPath $stagingPath -Directory -Filter "torch-*.dist-info" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $stagingPath -Directory -Filter "torchaudio-*.dist-info" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force

    & $PythonExe "scripts/rvc_worker.py" --check-imports --site-packages-dir $stagingPath
    if ($LASTEXITCODE -ne 0) {
        throw "RVC worker import check failed with exit code $LASTEXITCODE"
    }

    if (Test-Path $installPath) {
        Remove-Item -LiteralPath $installPath -Recurse -Force
    }
    Move-Item -LiteralPath $stagingPath -Destination $installPath
}
finally {
    Remove-Item $cmdPath -ErrorAction SilentlyContinue
    if (Test-Path $stagingPath) {
        Remove-Item -LiteralPath $stagingPath -Recurse -Force
    }
}

Write-Host ""
Write-Host "RVC install finished."
Write-Host "The overlay was built in a clean staging directory, then swapped atomically."
Write-Host "Recommended config:"
Write-Host "  tts.rvc.backend: worker"
Write-Host "  tts.rvc.python_path: $PythonExe"
Write-Host "  tts.rvc.site_packages_dir: $InstallDir"
