# Python Dependency Audit

This note records the reproducible Python audit flow used for `LIL-40`.

## Environments

Keep two environments separate:

- App/runtime environment: installs `requirements.txt` or `requirements-dev.txt`.
- Audit tooling environment: installs tools such as `pip-audit`, `bandit`, and `semgrep`.

Do not install audit tooling into the app environment just to run scans.

## Install

Runtime install:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Development and tests:

```powershell
python -m pip install -r requirements-dev.txt
```

Experimental omni / multimodal providers:

```powershell
python -m venv .venv-omni-experiment
.venv-omni-experiment\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-optional-omni.txt
```

Do not install `requirements-optional-omni.txt` into the stable runtime
environment for now. `minicpmo-utils` currently pins `pillow==10.4.0`, which
conflicts with the audited runtime lower bound `pillow>=12.2.0`.

Audit tooling:

```powershell
python -m venv .venv-audit
.venv-audit\Scripts\python -m pip install --upgrade pip
.venv-audit\Scripts\python -m pip install pip-audit bandit semgrep
```

## pip-audit

Audit the stable dependency declaration first:

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
.venv-audit\Scripts\python -m pip_audit -r requirements.txt -f json -o reports/pip-audit-requirements.json
```

Then audit a clean installed app environment:

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
.venv-audit\Scripts\python -m pip_audit --path venv\Lib\site-packages -f json -o reports/pip-audit-app-env.json
```

If `pip check` or `pip-audit --local` reports packages that are not installed by
`requirements.txt`, recreate the app venv instead of weakening the audited
runtime constraints. Historical local venvs may still contain experimental
packages such as `minicpmo-utils`, `gradio`, `diffusers`, or old RVC packages.

If you need to audit the declared dependency files too, run OSV against the
source tree and treat dependency-resolution failures as tool limitations to
investigate, not as a clean result:

```powershell
osv-scanner scan source -r . --format json --output-file reports/osv.json
```

OSV may still flag `requirements-optional-omni.txt`. That file is intentionally
outside the stable audited runtime until its upstream Pillow pin is fixed.

## Current security policy

- `constraints-security.txt` carries known safe lower bounds for vulnerable
  transitive packages reported by the May 2026 audit.
- `requirements.txt` stays focused on runtime dependencies.
- `requirements-dev.txt` owns test dependencies.
- `requirements-optional-omni.txt` owns experimental omni / multimodal
  dependencies and is not part of the audited stable install until its Pillow
  pin is fixed upstream.
- Worker requirement files reuse the same security constraints.
- Do not bump Torch/CUDA packages as part of dependency hygiene unless the
  ticket explicitly targets GPU runtime compatibility.
