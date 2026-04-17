#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    net::TcpListener,
    process::Command as StdCommand,
    sync::{Mutex, MutexGuard},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{
    menu::MenuBuilder,
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, RunEvent, State, WebviewWindow, WindowEvent,
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use window_vibrancy::{apply_acrylic, apply_mica, clear_acrylic, clear_mica};

const MAIN_WINDOW_LABEL: &str = "main";
const TRAY_ID: &str = "companion-tray";
const SIDECAR_NAME: &str = "binaries/local-ai-companion-sidecar";
const READY_TIMEOUT: Duration = Duration::from_secs(15);
const EXTERNAL_BACKEND_PORT_ENV: &str = "LOCAL_AI_COMPANION_BACKEND_PORT";

#[derive(Default)]
struct AppStateInner {
    backend: Option<BackendProcess>,
    allow_exit: bool,
    preferences: HostPreferences,
}

struct AppState {
    inner: Mutex<AppStateInner>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(AppStateInner {
                backend: None,
                allow_exit: false,
                preferences: HostPreferences::default(),
            }),
        }
    }
}

impl AppState {
    fn lock(&self) -> Result<MutexGuard<'_, AppStateInner>, String> {
        self.inner.lock().map_err(|_| "host state mutex poisoned".to_string())
    }
}

struct BackendProcess {
    port: u16,
    child: Option<CommandChild>,
    runtime: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HostPreferences {
    always_on_top: bool,
    start_minimized_to_tray: bool,
    auto_hide_enabled: bool,
    character_id: String,
    panel_side: String,
}

impl Default for HostPreferences {
    fn default() -> Self {
        Self {
            always_on_top: true,
            start_minimized_to_tray: false,
            auto_hide_enabled: true,
            character_id: "march7th".into(),
            panel_side: "left".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BootstrapPayload {
    bridge_port: u16,
    runtime: Value,
    preferences: HostPreferences,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HostActionPayload {
    action: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelSurface {
    Pet,
    Chat,
    Expand,
    Settings,
}

impl PanelSurface {
    fn from_str(value: &str) -> Self {
        match value {
            "chat" => Self::Chat,
            "expand" => Self::Expand,
            "settings" => Self::Settings,
            _ => Self::Pet,
        }
    }
}

#[tauri::command]
async fn bootstrap_app(app: AppHandle, state: State<'_, AppState>) -> Result<BootstrapPayload, String> {
    ensure_backend(&app, state.inner()).await
}

#[tauri::command]
async fn restart_backend(app: AppHandle, state: State<'_, AppState>) -> Result<BootstrapPayload, String> {
    stop_backend(&state)?;
    let payload = ensure_backend(&app, state.inner()).await?;
    let _ = app.emit("host://backend-restarted", payload.clone());
    Ok(payload)
}

#[tauri::command]
fn quit_app(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    {
        let mut guard = state.lock()?;
        guard.allow_exit = true;
    }
    stop_backend(&state)?;
    app.exit(0);
    Ok(())
}

#[tauri::command]
fn set_host_preferences(
    app: AppHandle,
    state: State<'_, AppState>,
    preferences: HostPreferences,
) -> Result<HostPreferences, String> {
    {
        let mut guard = state.lock()?;
        guard.preferences = preferences.clone();
    }
    if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
        let _ = window.set_always_on_top(preferences.always_on_top);
        let _ = window.set_skip_taskbar(preferences.start_minimized_to_tray);
    }
    Ok(preferences)
}

#[tauri::command]
fn sync_window_surface(
    app: AppHandle,
    state: State<'_, AppState>,
    surface: String,
) -> Result<(), String> {
    let window = app
        .get_webview_window(MAIN_WINDOW_LABEL)
        .ok_or_else(|| "main window not found".to_string())?;
    let preferences = {
        let guard = state.lock()?;
        guard.preferences.clone()
    };
    apply_surface_effects(&window, PanelSurface::from_str(&surface))
        .map_err(|error| error.to_string())?;
    let _ = window.set_always_on_top(preferences.always_on_top);
    Ok(())
}

fn emit_host_action(app: &AppHandle, action: &str) {
    let _ = app.emit(
        "host://action",
        HostActionPayload {
            action: action.to_string(),
        },
    );
}

fn apply_surface_effects(window: &WebviewWindow, surface: PanelSurface) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        let _ = clear_acrylic(window);
        let _ = clear_mica(window);

        match surface {
            PanelSurface::Pet => {}
            PanelSurface::Chat | PanelSurface::Expand => {
                apply_acrylic(window, Some((10, 16, 29, 185)))
                    .context("failed to apply acrylic effect")?;
            }
            PanelSurface::Settings => {
                apply_mica(window, Some(true)).context("failed to apply mica effect")?;
            }
        }
    }

    Ok(())
}

fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("failed to bind ephemeral port")?;
    let port = listener
        .local_addr()
        .context("failed to resolve ephemeral port")?
        .port();
    drop(listener);
    Ok(port)
}

async fn ensure_backend(app: &AppHandle, state: &AppState) -> Result<BootstrapPayload, String> {
    {
        let guard = state.lock()?;
        if let Some(backend) = &guard.backend {
            return Ok(BootstrapPayload {
                bridge_port: backend.port,
                runtime: backend.runtime.clone(),
                preferences: guard.preferences.clone(),
            });
        }
    }

    let external_port = external_backend_port();
    let port = if let Some(port) = external_port {
        port
    } else {
        find_free_port().map_err(|error| error.to_string())?
    };
    let child = if external_port.is_some() {
        None
    } else {
        Some(spawn_sidecar(app, port).map_err(|error| error.to_string())?)
    };
    let runtime = wait_for_backend_ready(port)
        .await
        .map_err(|error| error.to_string())?;

    let payload = {
        let mut guard = state.lock()?;
        let preferences = guard.preferences.clone();
        guard.backend = Some(BackendProcess {
            port,
            child,
            runtime: runtime.clone(),
        });

        BootstrapPayload {
            bridge_port: port,
            runtime,
            preferences,
        }
    };

    let _ = app.emit("host://backend-ready", payload.clone());
    Ok(payload)
}

fn external_backend_port() -> Option<u16> {
    std::env::var(EXTERNAL_BACKEND_PORT_ENV)
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
}

fn spawn_sidecar(app: &AppHandle, port: u16) -> Result<CommandChild> {
    let port_string = port.to_string();
    let (mut rx, child) = app
        .shell()
        .sidecar(SIDECAR_NAME)
        .context("failed to resolve desktop backend sidecar")?
        .args(["--bridge-server", "--bridge-port", port_string.as_str()])
        .spawn()
        .context("failed to spawn desktop backend sidecar")?;

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    if let Ok(text) = String::from_utf8(line) {
                        let _ = handle.emit("host://backend-log", json!({ "stream": "stdout", "line": text }));
                    }
                }
                CommandEvent::Stderr(line) => {
                    if let Ok(text) = String::from_utf8(line) {
                        let _ = handle.emit("host://backend-log", json!({ "stream": "stderr", "line": text }));
                    }
                }
                CommandEvent::Terminated(payload) => {
                    let _ = handle.emit("host://backend-exited", payload);
                }
                CommandEvent::Error(error) => {
                    let _ = handle.emit("host://backend-error", error);
                }
                _ => {}
            }
        }
    });

    Ok(child)
}

async fn wait_for_backend_ready(port: u16) -> Result<Value> {
    let started_at = Instant::now();
    let url = format!("ws://127.0.0.1:{port}");

    loop {
        if started_at.elapsed() > READY_TIMEOUT {
            anyhow::bail!("desktop backend did not report ready on port {port}");
        }

        match connect_async(&url).await {
            Ok((mut stream, _)) => {
                while let Some(message) = stream.next().await {
                    match message.context("failed to read backend bridge message")? {
                        Message::Text(payload) => {
                            let value: Value = serde_json::from_str(&payload)
                                .context("failed to decode backend ready payload")?;
                            if value.get("type").and_then(Value::as_str) == Some("backend_ready") {
                                return Ok(value.get("runtime").cloned().unwrap_or_else(|| json!({})));
                            }
                        }
                        Message::Binary(payload) => {
                            let text = String::from_utf8(payload)
                                .context("failed to decode binary backend ready payload")?;
                            let value: Value = serde_json::from_str(&text)
                                .context("failed to decode backend ready binary payload")?;
                            if value.get("type").and_then(Value::as_str) == Some("backend_ready") {
                                return Ok(value.get("runtime").cloned().unwrap_or_else(|| json!({})));
                            }
                        }
                        Message::Close(_) => break,
                        _ => {}
                    }
                }
            }
            Err(_) => {
                sleep(Duration::from_millis(180)).await;
            }
        }
    }
}

fn stop_backend(state: &AppState) -> Result<(), String> {
    let backend = {
        let mut guard = state.lock()?;
        guard.backend.take()
    };

    if let Some(backend) = backend {
        if let Some(backend_child) = backend.child {
            let pid = backend_child.pid();
            let _ = backend_child.kill();
            #[cfg(target_os = "windows")]
            {
                let _ = StdCommand::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/T", "/F"])
                    .status();
            }
        }
    }

    Ok(())
}

fn toggle_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
        if window.is_visible().unwrap_or(false) {
            let _ = window.hide();
        } else {
            let _ = window.show();
            let _ = window.set_focus();
        }
    }
}

fn build_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let tray_menu = MenuBuilder::new(app)
        .text("toggle-chat", "Toggle chat")
        .text("toggle-expand", "Toggle expand")
        .text("toggle-mute", "Mute / unmute")
        .separator()
        .text("restart-backend", "Restart backend")
        .text("open-settings", "Settings")
        .separator()
        .text("quit", "Quit")
        .build()?;

    let app_handle = app.handle().clone();
    TrayIconBuilder::with_id(TRAY_ID)
        .icon(app.default_window_icon().expect("default app icon missing").clone())
        .menu(&tray_menu)
        .show_menu_on_left_click(false)
        .on_tray_icon_event(move |_tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                toggle_main_window(&app_handle);
            }
        })
        .build(app)?;

    Ok(())
}

fn register_shortcuts(app: &tauri::AppHandle) {
    let shortcuts = [
        Shortcut::new(None, Code::F2),
        Shortcut::new(None, Code::F3),
        Shortcut::new(None, Code::F11),
        Shortcut::new(None, Code::F12),
    ];

    for shortcut in shortcuts {
        if let Err(error) = app.global_shortcut().register(shortcut) {
            eprintln!("warning: failed to register desktop shortcut {shortcut:?}: {error}");
        }
    }
}

fn handle_shortcut(app: &AppHandle, shortcut: &Shortcut) {
    if *shortcut == Shortcut::new(None, Code::F2) {
        emit_host_action(app, "toggle-mute");
    } else if *shortcut == Shortcut::new(None, Code::F3) {
        emit_host_action(app, "interrupt");
    } else if *shortcut == Shortcut::new(None, Code::F11) {
        emit_host_action(app, "toggle-expand");
    } else if *shortcut == Shortcut::new(None, Code::F12) {
        emit_host_action(app, "toggle-debug");
    }
}

fn handle_menu_event(app: &AppHandle, item_id: &str, state: &AppState) {
    match item_id {
        "toggle-chat" => emit_host_action(app, "toggle-chat"),
        "toggle-expand" => emit_host_action(app, "toggle-expand"),
        "toggle-mute" => emit_host_action(app, "toggle-mute"),
        "restart-backend" => {
            let handle = app.clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<AppState>();
                let _ = restart_backend(handle.clone(), state).await;
            });
        }
        "open-settings" => emit_host_action(app, "open-settings"),
        "quit" => {
            if let Ok(mut guard) = state.lock() {
                guard.allow_exit = true;
            }
            let _ = stop_backend(state);
            app.exit(0);
        }
        _ => {}
    }
}

fn should_prevent_close(state: &AppState) -> bool {
    state
        .lock()
        .map(|guard| !guard.allow_exit)
        .unwrap_or(true)
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        handle_shortcut(app, shortcut);
                    }
                })
                .build(),
        )
        .invoke_handler(tauri::generate_handler![
            bootstrap_app,
            restart_backend,
            quit_app,
            set_host_preferences,
            sync_window_surface
        ])
        .setup(|app| {
            build_tray(app).expect("failed to build tray");
            register_shortcuts(&app.handle());

            if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
                apply_surface_effects(&window, PanelSurface::Pet).ok();
                let state = app.state::<AppState>();
                if state
                    .lock()
                    .map(|guard| guard.preferences.start_minimized_to_tray)
                    .unwrap_or(false)
                {
                    let _ = window.hide();
                }
            }

            Ok(())
        })
        .on_menu_event(|app, event| {
            let state = app.state::<AppState>();
            handle_menu_event(app, event.id().as_ref(), &state);
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle();
                let state = app.state::<AppState>();
                if should_prevent_close(&state) {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::ExitRequested { api, .. } = event {
                let state = app.state::<AppState>();
                if should_prevent_close(&state) {
                    api.prevent_exit();
                    if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
                        let _ = window.hide();
                    }
                } else {
                    let _ = stop_backend(&state);
                    let _ = app.global_shortcut().unregister_all();
                }
            }
        });
}
