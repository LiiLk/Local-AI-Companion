param(
    [string]$BasePythonExe = ".\venv\Scripts\python.exe",
    [string]$VenvDir = ".\.venv-qwen3-tts",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$TorchVersion = "2.11.0+cu128",
    [string]$TorchaudioVersion = "2.11.0+cu128",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Get-NativeExitCode {
    if ($null -eq $LASTEXITCODE) {
        if ($?) {
            return 0
        }
        return 1
    }
    return [int]$LASTEXITCODE
}

if (-not (Test-Path $BasePythonExe)) {
    throw "Base Python executable not found: $BasePythonExe"
}

$venvPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $VenvDir))
$requirementsFile = Join-Path (Get-Location) "scripts\requirements-qwen3-tts-worker.txt"
if (-not (Test-Path $requirementsFile)) {
    throw "Qwen3-TTS requirements file not found: $requirementsFile"
}

if ($Clean -and (Test-Path $venvPath)) {
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating dedicated Qwen3-TTS worker venv: $venvPath"
    & $BasePythonExe -m venv $venvPath
    $exitCode = Get-NativeExitCode
    if ($exitCode -ne 0) {
        throw "Failed to create Qwen3-TTS venv with exit code $exitCode"
    }
}

$workerPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $workerPython)) {
    throw "Qwen3-TTS worker python not found: $workerPython"
}

Write-Host "Base Python: $BasePythonExe"
Write-Host "Worker Python: $workerPython"
Write-Host "Installing isolated Qwen3-TTS packages into dedicated worker venv"

& $workerPython -m pip install --upgrade "pip<24.1"
$exitCode = Get-NativeExitCode
if ($exitCode -ne 0) {
    throw "Failed to upgrade worker pip with exit code $exitCode"
}

Write-Host "Installing CUDA-enabled torch runtime for Qwen3-TTS worker"
& $workerPython -m pip install --upgrade --force-reinstall --index-url $TorchIndexUrl "torch==$TorchVersion" "torchaudio==$TorchaudioVersion"
$exitCode = Get-NativeExitCode
if ($exitCode -ne 0) {
    throw "Failed to install CUDA-enabled torch runtime with exit code $exitCode"
}

& $workerPython -m pip install --upgrade -r "$requirementsFile"
$exitCode = Get-NativeExitCode
if ($exitCode -ne 0) {
    throw "Qwen3-TTS dependency installation failed with exit code $exitCode"
}

& $workerPython "scripts/qwen3_tts_worker.py" --check-imports
$exitCode = Get-NativeExitCode
if ($exitCode -ne 0) {
    throw "Qwen3-TTS worker import check failed with exit code $exitCode"
}

Write-Host ""
Write-Host "Qwen3-TTS install finished."
Write-Host "Recommended config:"
Write-Host "  tts.qwen3.backend: worker"
Write-Host "  tts.qwen3.python_path: $VenvDir\Scripts\python.exe"
Write-Host "  tts.qwen3.site_packages_dir: null"
Write-Host "  tts.qwen3.worker_script: .\scripts\qwen3_tts_worker.py"
