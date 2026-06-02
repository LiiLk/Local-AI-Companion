# Security Audit Baseline

This document records the local audit baseline for `LIL-43`.

The goal is not to make every scan perfect or mandatory in CI. The goal is to
make the pre-release security and quality checks reproducible before public
announcements or security-sensitive PRs.

## Scope

The baseline covers:

- Python tests and runtime sanity checks.
- Secret scanning with Gitleaks.
- Static analysis with Semgrep and Bandit.
- Python dependency auditing with `pip-audit`.
- OSV dependency scanning across lockfiles and requirement files.
- Tauri frontend dependency auditing with `npm audit`.
- Rust dependency auditing with `cargo audit` when installed.

Generated reports must stay local:

- `reports/`
- `.venv-audit/`
- `.venv-prcheck/`

These paths are already ignored by Git.

## Recommended Environments

Keep the app environment and audit tooling environment separate.

App / test environment:

```powershell
python -m venv .venv-prcheck
.venv-prcheck\Scripts\python.exe -m pip install --upgrade pip
.venv-prcheck\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Audit tooling environment:

```powershell
python -m venv .venv-audit
.venv-audit\Scripts\python.exe -m pip install --upgrade pip
.venv-audit\Scripts\python.exe -m pip install pip-audit bandit semgrep
```

External tools:

```powershell
# Install separately through your preferred package manager:
# - gitleaks
# - osv-scanner
# - cargo-audit
#
# Example cargo audit install:
cargo install cargo-audit
```

Do not install Semgrep, Bandit, or pip-audit into the app runtime venv just to
run scans.

## One-Command Local Baseline

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_security_audit_windows.ps1
```

The script:

- creates `reports/`;
- sets UTF-8 environment variables for Windows CLI output;
- keeps tool caches inside `reports/` where possible;
- runs available tools;
- skips missing tools with a clear message;
- writes JSON or text reports under `reports/`;
- prints a summary at the end.

Some tools return non-zero when they find vulnerabilities or findings. Treat a
non-zero audit step as "review required", not automatically as "the script is
broken".

## Manual Commands

Use these if you want to run a specific check.

### Pytest

```powershell
.venv-prcheck\Scripts\python.exe -m pytest tests -q
```

For fast security-regression validation after local hardening changes:

```powershell
.venv-prcheck\Scripts\python.exe -m pytest tests/test_server_settings.py tests/test_server_routes.py tests/test_websocket_security_limits.py tests/test_rvc_provider.py tests/test_pipeline_runtime.py tests/test_websocket_turn_scheduling.py tests/test_live2d_assistant.py -q
```

If Codex sandbox runs fail on Windows temp/cache permissions, rerun outside the
sandbox or from a normal PowerShell terminal.

### Gitleaks

```powershell
gitleaks detect --source . --report-format json --report-path reports/gitleaks.json --no-banner
```

### Semgrep

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:SEMGREP_SEND_METRICS = "off"
.venv-audit\Scripts\semgrep.exe scan --metrics off --config p/python --config p/javascript --config p/owasp-top-ten --json --output reports/semgrep.json .
```

The UTF-8 environment variables avoid Windows `charmap` encoding failures when
Semgrep writes JSON containing replacement characters.

### Bandit

```powershell
.venv-audit\Scripts\python.exe -m bandit -r src scripts -f json -o reports/bandit.json
```

### pip-audit

Prefer auditing the clean installed runtime environment:

```powershell
.venv-audit\Scripts\python.exe -m pip_audit --path .venv-prcheck\Lib\site-packages -f json -o reports/pip-audit-app-env.json --cache-dir reports\pip-audit-cache --progress-spinner off
```

Auditing `requirements.txt` directly can hit resolver limits or package-index
differences for ML packages. If that happens, do not weaken runtime constraints
just to satisfy the resolver. Use the installed clean environment result and
document the resolver limitation.

`pip-audit` needs network access to query vulnerability services. If it fails
from a restricted Codex sandbox because traffic is routed to a refused local
proxy, rerun it from a normal PowerShell terminal.

### OSV-Scanner

```powershell
osv-scanner scan source -r . --format json --output-file reports/osv.json
```

`--output-file` is preferred over the deprecated `--output` flag.

OSV also needs network access. Treat offline/proxy failures as environment
failures, not as a clean security result.

### npm audit

```powershell
npm audit --prefix desktop/tauri --json | Out-File reports/npm-audit.json -Encoding utf8
```

### cargo audit

Install first if needed:

```powershell
cargo install cargo-audit
```

Then run from `desktop/tauri/src-tauri`:

```powershell
cargo audit --json | Out-File ..\..\..\reports\cargo-audit.json -Encoding utf8
```

## CI Decision

No new GitHub Actions workflow is added for `LIL-43`.

Reason:

- The current repository has heavyweight ML dependencies and optional provider
  stacks that can make dependency resolution noisy or expensive in CI.
- `pip-audit -r requirements.txt` and OSV requirement extraction may fail for
  resolver/tooling reasons even when the clean runtime venv is auditable.
- Security scans are still being stabilized locally; adding a flaky CI gate now
  would slow review without clearly improving safety.

Recommended future CI once the dependency graph is cleaner:

- Gitleaks on every PR.
- A small Python test shard focused on config/server/security boundaries.
- Semgrep with curated configs.
- npm audit for `desktop/tauri`.
- cargo audit after `cargo-audit` is installed in the CI job.

Avoid CI jobs that download voice models, initialize GPU backends, or install
experimental omni provider stacks by default.

## Pre-Announcement Checklist

Before posting a public demo or asking for broad GitHub review:

1. Run `gitleaks detect` and confirm no leaks.
2. Run the focused security pytest shard.
3. Run Semgrep and Bandit; triage high-confidence findings.
4. Run `pip-audit` against a clean app/test environment.
5. Run OSV and separate real vulnerable packages from resolver/tool limitations.
6. Run `npm audit` and `cargo audit` for the Tauri app.
7. Confirm `reports/`, `.venv-audit/`, and local config files are not staged.
8. Check `git status` before opening the PR.

## Known Exceptions

- Optional omni provider dependency files are not the stable product baseline.
  Findings there should be tracked, but they should not block stable pipeline
  hardening unless those modes are promoted.
- RVC and voice model files are local trust-boundary artifacts. Prefer
  publishing SHA-256 hashes for distributed voice packs.
- Reports under `reports/` are local evidence, not public docs. Summarize
  relevant findings in `docs/` or Linear instead of committing raw scanner
  output.

## Related Notes

- `docs/python-dependency-audit.md`
- `docs/security-static-findings.md`
- `docs/security-threat-model.md`
- `docs/lil-19-security-scan.md`
