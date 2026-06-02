param(
    [switch]$SkipTests,
    [switch]$FullTests
)

$ErrorActionPreference = "Continue"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$reportDir = Join-Path $repoRoot "reports"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:SEMGREP_SEND_METRICS = "off"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

New-Item -ItemType Directory -Force $reportDir | Out-Null
Set-Location $repoRoot

$results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param(
        [string]$Name,
        [string]$Status,
        [int]$ExitCode,
        [string]$Report,
        [string]$Note
    )

    $results.Add([pscustomobject]@{
        name = $Name
        status = $Status
        exitCode = $ExitCode
        report = $Report
        note = $Note
    }) | Out-Null
}

function Test-CommandAvailable {
    param([string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Resolve-AuditPython {
    $auditPython = Join-Path $repoRoot ".venv-audit\Scripts\python.exe"
    if (Test-Path $auditPython) {
        return $auditPython
    }
    return $null
}

function Resolve-TestPython {
    $prcheckPython = Join-Path $repoRoot ".venv-prcheck\Scripts\python.exe"
    if (Test-Path $prcheckPython) {
        return $prcheckPython
    }

    $venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    if (Test-CommandAvailable "python") {
        return "python"
    }

    return $null
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Report,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "==> $Name"

    $global:LASTEXITCODE = 0
    try {
        & $Action
        $exitCode = if ($null -eq $global:LASTEXITCODE) { 0 } else { [int]$global:LASTEXITCODE }
        if ($exitCode -eq 0) {
            Add-Result $Name "passed" $exitCode $Report ""
        } else {
            Add-Result $Name "review-required" $exitCode $Report "Non-zero exit code; review the report."
        }
    } catch {
        Add-Result $Name "failed" 1 $Report $_.Exception.Message
        Write-Warning $_.Exception.Message
    }
}

function Invoke-Skip {
    param(
        [string]$Name,
        [string]$Reason
    )

    Write-Host ""
    Write-Host "==> $Name"
    Write-Host "Skipped: $Reason"
    Add-Result $Name "skipped" 0 "" $Reason
}

$testPython = Resolve-TestPython
$auditPython = Resolve-AuditPython

if (-not $SkipTests) {
    if ($null -eq $testPython) {
        Invoke-Skip "pytest" "No Python interpreter found for tests."
    } else {
        $pytestReport = Join-Path $reportDir "pytest-security.txt"
        $pytestArgs = @("-m", "pytest")
        if ($FullTests) {
            $pytestArgs += @("tests", "-q")
        } else {
            $pytestArgs += @(
                "tests/test_server_settings.py",
                "tests/test_server_routes.py",
                "tests/test_websocket_security_limits.py",
                "tests/test_rvc_provider.py",
                "tests/test_pipeline_runtime.py",
                "tests/test_websocket_turn_scheduling.py",
                "tests/test_live2d_assistant.py",
                "-q"
            )
        }

        Invoke-Step "pytest" $pytestReport {
            & $testPython @pytestArgs 2>&1 | Tee-Object -FilePath $pytestReport
        }
    }
}

if (Test-CommandAvailable "gitleaks") {
    $gitleaksReport = Join-Path $reportDir "gitleaks.json"
    Invoke-Step "gitleaks" $gitleaksReport {
        & gitleaks detect --source . --report-format json --report-path $gitleaksReport --no-banner
    }
} else {
    Invoke-Skip "gitleaks" "Install gitleaks to scan for committed secrets."
}

if (Test-Path (Join-Path $repoRoot ".venv-audit\Scripts\semgrep.exe")) {
    $semgrep = Join-Path $repoRoot ".venv-audit\Scripts\semgrep.exe"
    $semgrepReport = Join-Path $reportDir "semgrep.json"
    Invoke-Step "semgrep" $semgrepReport {
        & $semgrep scan --metrics off --config p/python --config p/javascript --config p/owasp-top-ten --json --output $semgrepReport .
    }
} elseif (Test-CommandAvailable "semgrep") {
    $semgrepReport = Join-Path $reportDir "semgrep.json"
    Invoke-Step "semgrep" $semgrepReport {
        & semgrep scan --metrics off --config p/python --config p/javascript --config p/owasp-top-ten --json --output $semgrepReport .
    }
} else {
    Invoke-Skip "semgrep" "Install semgrep in .venv-audit or globally."
}

if ($null -ne $auditPython) {
    $banditReport = Join-Path $reportDir "bandit.json"
    Invoke-Step "bandit" $banditReport {
        & $auditPython -m bandit -r src scripts -f json -o $banditReport
    }

    $sitePackages = Join-Path $repoRoot ".venv-prcheck\Lib\site-packages"
    if (Test-Path $sitePackages) {
        $pipAuditReport = Join-Path $reportDir "pip-audit-app-env.json"
        $pipAuditCache = Join-Path $reportDir "pip-audit-cache"
        Invoke-Step "pip-audit app env" $pipAuditReport {
            & $auditPython -m pip_audit --path $sitePackages -f json -o $pipAuditReport --cache-dir $pipAuditCache --progress-spinner off
        }
    } else {
        Invoke-Skip "pip-audit app env" ".venv-prcheck\Lib\site-packages was not found."
    }
} else {
    Invoke-Skip "bandit" "Create .venv-audit and install bandit."
    Invoke-Skip "pip-audit app env" "Create .venv-audit and install pip-audit."
}

if (Test-CommandAvailable "osv-scanner") {
    $osvReport = Join-Path $reportDir "osv.json"
    Invoke-Step "osv-scanner" $osvReport {
        & osv-scanner scan source -r . --format json --output-file $osvReport
    }
} else {
    Invoke-Skip "osv-scanner" "Install osv-scanner to scan dependency manifests."
}

if (Test-CommandAvailable "npm") {
    $npmAuditReport = Join-Path $reportDir "npm-audit.json"
    Invoke-Step "npm audit" $npmAuditReport {
        & npm audit --prefix desktop/tauri --json 2>&1 | Out-File $npmAuditReport -Encoding utf8
    }
} else {
    Invoke-Skip "npm audit" "Install Node.js/npm to audit desktop/tauri."
}

if (Test-CommandAvailable "cargo") {
    $cargoAuditAvailable = $false
    $cargoList = & cargo --list 2>$null
    if ($cargoList -match "(?m)^\s+audit\s") {
        $cargoAuditAvailable = $true
    }

    if ($cargoAuditAvailable) {
        $cargoAuditReport = Join-Path $reportDir "cargo-audit.json"
        Invoke-Step "cargo audit" $cargoAuditReport {
            Push-Location (Join-Path $repoRoot "desktop\tauri\src-tauri")
            try {
                & cargo audit --json 2>&1 | Out-File $cargoAuditReport -Encoding utf8
            } finally {
                Pop-Location
            }
        }
    } else {
        Invoke-Skip "cargo audit" "Install with: cargo install cargo-audit"
    }
} else {
    Invoke-Skip "cargo audit" "Install Rust/Cargo to audit desktop/tauri/src-tauri."
}

$summaryPath = Join-Path $reportDir "security-audit-summary.json"
$results | ConvertTo-Json -Depth 4 | Out-File $summaryPath -Encoding utf8

Write-Host ""
Write-Host "Security audit summary:"
$results | Format-Table -AutoSize
Write-Host ""
Write-Host "Summary written to $summaryPath"

$failed = $results | Where-Object { $_.status -eq "failed" }
if ($failed.Count -gt 0) {
    exit 1
}

exit 0
