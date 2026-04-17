# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
entry_script = project_root / "src" / "assistant" / "app.py"
config_dir = project_root / "config"

datas = []
if config_dir.exists():
    datas.append((str(config_dir), "config"))

hiddenimports = [
    "yaml",
    "websockets",
    "pynput",
    "webview",
]

a = Analysis(
    [str(entry_script)],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="local-ai-companion-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
