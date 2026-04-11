#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use tauri::{Emitter, Manager};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeState {
    mode: String,
    mic_state: String,
    active_turn_id: Option<u64>,
    response_active: bool,
    playback_active: bool,
    debug_visible: bool,
    character_name: String,
    backend: String,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            mode: "pipeline".into(),
            mic_state: "muted".into(),
            active_turn_id: None,
            response_active: false,
            playback_active: false,
            debug_visible: false,
            character_name: "March 7th".into(),
            backend: "tauri-scaffold".into(),
        }
    }
}

#[tauri::command]
fn get_runtime_state() -> RuntimeState {
    RuntimeState::default()
}

#[tauri::command]
fn send_text(text: String) -> RuntimeState {
    let mut state = RuntimeState::default();
    if !text.trim().is_empty() {
        state.response_active = true;
    }
    state
}

#[tauri::command]
fn interrupt() -> RuntimeState {
    RuntimeState::default()
}

#[tauri::command]
fn toggle_mute() -> RuntimeState {
    let mut state = RuntimeState::default();
    state.mic_state = "listening".into();
    state
}

#[tauri::command]
fn toggle_debug() -> RuntimeState {
    let mut state = RuntimeState::default();
    state.debug_visible = true;
    state
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            get_runtime_state,
            send_text,
            interrupt,
            toggle_mute,
            toggle_debug,
        ])
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.emit("backend-ready", RuntimeState::default());
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
