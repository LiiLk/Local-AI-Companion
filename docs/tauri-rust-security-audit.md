# SEC-03 - Tauri and Rust Security Audit

This note records the SEC-03 audit scope for the desktop Tauri scaffold.

## Scope

- Tauri JavaScript CLI and Rust crate versions.
- Tauri capability and IPC exposure.
- Local desktop frontend CSP.
- Rust advisory results from OSV Scanner.

## Changes Applied

- Updated `@tauri-apps/cli` to `2.11.2`.
- Updated `tauri` to `2.11.1`.
- Updated `tauri-build` to `2.6.2`.
- Removed unused `tauri-plugin-dialog` and `tauri-plugin-shell`.
- Kept the Tauri capability set minimal:
  - `core:default`
  - `core:window:allow-start-dragging`
- Added a desktop CSP instead of leaving CSP disabled, while explicitly allowing Tauri's IPC endpoints (`ipc:` and `http://ipc.localhost`) plus the local assistant bridge.

## IPC Review

The current Tauri shell is still a scaffold. The frontend detects Tauri through
`window.__TAURI__`, but the active desktop bridge sends assistant commands over
the local WebSocket bridge at `ws://127.0.0.1:8765`.

The Rust-side Tauri commands currently return local state only:

- `get_runtime_state`
- `send_text`
- `interrupt`
- `toggle_mute`
- `toggle_debug`

They do not execute shell commands, read files, write files, or expose local
secrets. The removed shell and dialog plugins were not used by the frontend and
did not need to stay in the process.

## Advisory Results

Before the update, OSV reported the Tauri advisory
`GHSA-7gmj-67g7-phm9 / CVE-2026-42184` on `tauri 2.10.3`, fixed in `2.11.1`.
After the update, that advisory is no longer present.

Remaining OSV findings are transitive advisories from the Tauri/Wry dependency
stack:

- GTK3 / `gtk-rs` unmaintained advisories:
  - `atk`
  - `atk-sys`
  - `gdk`
  - `gdk-sys`
  - `gdkwayland-sys`
  - `gdkx11`
  - `gdkx11-sys`
  - `gtk`
  - `gtk-sys`
  - `gtk3-macros`
- `glib 0.18.5` unsoundness advisory.
- `proc-macro-error 1.0.4` unmaintained advisory.
- `unic-*` unmaintained advisories through `urlpattern` / `tauri-utils`.

These should be monitored, but they are not currently direct application-level
findings in the Windows-first desktop scaffold. The GTK and GLib findings come
through the Linux WebKitGTK/Wry path. Removing them would require upstream Tauri
/ Wry dependency movement or a deliberate platform support change, not a local
one-line patch.

## Reproduction Commands

```powershell
osv-scanner scan source desktop\tauri\src-tauri --format json --output-file reports\osv-tauri-src-tauri-after.json
npm audit --prefix desktop\tauri --json
cd desktop\tauri\src-tauri
cargo check
```

Optional `cargo-audit` setup:

```powershell
cargo install cargo-audit --locked
cd desktop\tauri\src-tauri
cargo audit
```

`reports/` is ignored locally and should not be committed.
