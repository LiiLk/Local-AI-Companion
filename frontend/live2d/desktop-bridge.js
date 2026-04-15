(function bootstrapDesktopBridge() {
  const url = new URL(window.location.href);
  const queryBackendPort = Number(url.searchParams.get("backendPort")) || 8765;
  let tauriSocket = null;
  let tauriConnectPromise = null;
  let requestCounter = 0;
  const pendingRequests = new Map();
  let tauriRuntime = {
    mode: "pipeline",
    mic_state: "loading",
    active_turn_id: null,
    response_active: false,
    playback_active: false,
    debug_visible: false,
    character_name: "March 7th",
    backend: "assistant-bridge",
  };

  function hasPywebview() {
    return !!window.pywebview?.api;
  }

  function hasTauri() {
    return !!window.__TAURI__;
  }

  async function ensureBridgeReady() {
    if (hasPywebview()) {
      return "pywebview";
    }

    if (hasTauri()) {
      await ensureTauriSocket();
      return "tauri";
    }

    for (let attempt = 0; attempt < 100; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 100));
      if (hasPywebview()) {
        return "pywebview";
      }
      if (hasTauri()) {
        await ensureTauriSocket();
        return "tauri";
      }
    }

    throw new Error("No desktop bridge available");
  }

  function websocketUrl() {
    return `ws://127.0.0.1:${queryBackendPort}`;
  }

  function dispatchFrontendEvent(handlerName, ...args) {
    window[handlerName]?.(...args);
  }

  function applyRuntime(runtime) {
    if (!runtime) {
      return;
    }
    tauriRuntime = { ...tauriRuntime, ...runtime };
  }

  function trackRuntimeFromEvent(name, args, runtime) {
    applyRuntime(runtime);

    switch (name) {
      case "onMicStateChange":
        tauriRuntime.mic_state = args[0] || tauriRuntime.mic_state;
        break;
      case "onSpeechStart":
        tauriRuntime.response_active = false;
        break;
      case "onResponseStart":
        tauriRuntime.active_turn_id = args[0] ?? tauriRuntime.active_turn_id;
        tauriRuntime.response_active = true;
        break;
      case "onTranscription":
        tauriRuntime.active_turn_id = args[1] ?? tauriRuntime.active_turn_id;
        break;
      case "onResponseEnd":
        tauriRuntime.active_turn_id = args[1] ?? tauriRuntime.active_turn_id;
        tauriRuntime.response_active = false;
        break;
      case "onAudioReady":
        tauriRuntime.active_turn_id = args[0]?.turn_id ?? tauriRuntime.active_turn_id;
        tauriRuntime.playback_active = true;
        break;
      case "onPlaybackStop":
        tauriRuntime.active_turn_id = args[0] ?? tauriRuntime.active_turn_id;
        tauriRuntime.response_active = false;
        tauriRuntime.playback_active = false;
        break;
      case "onError":
        tauriRuntime.response_active = false;
        tauriRuntime.playback_active = false;
        break;
      default:
        break;
    }
  }

  function handleTauriMessage(message) {
    if (message.type === "backend_ready") {
      applyRuntime(message.runtime);
      dispatchFrontendEvent("onBackendReady", { ...tauriRuntime });
      return;
    }

    if (message.type === "frontend_event") {
      trackRuntimeFromEvent(message.name, message.args || [], message.runtime);
      dispatchFrontendEvent(message.name, ...(message.args || []));
      return;
    }

    if (message.type === "command_result") {
      const pending = pendingRequests.get(message.request_id);
      if (!pending) {
        return;
      }

      pendingRequests.delete(message.request_id);
      if (message.ok) {
        applyRuntime(message.result);
        pending.resolve({ ...message.result });
      } else {
        pending.reject(new Error(message.error || `Desktop bridge command failed: ${message.name}`));
      }
    }
  }

  async function ensureTauriSocket() {
    if (tauriSocket && tauriSocket.readyState === WebSocket.OPEN) {
      return tauriSocket;
    }

    if (tauriConnectPromise) {
      return tauriConnectPromise;
    }

    tauriConnectPromise = new Promise((resolve, reject) => {
      const socket = new WebSocket(websocketUrl());
      tauriSocket = socket;

      socket.onopen = () => resolve(socket);

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleTauriMessage(message);
        } catch (error) {
          console.error("Failed to parse Tauri bridge message", error);
        }
      };

      socket.onerror = () => {
        reject(new Error(`Unable to connect to assistant bridge at ${websocketUrl()}`));
      };

      socket.onclose = () => {
        tauriSocket = null;
        tauriConnectPromise = null;
        for (const pending of pendingRequests.values()) {
          pending.reject(new Error("Desktop bridge disconnected"));
        }
        pendingRequests.clear();
      };
    });

    return tauriConnectPromise;
  }

  async function sendCommand(name, payload = {}) {
    await ensureTauriSocket();
    const requestId = `${Date.now().toString(36)}_${requestCounter++}`;

    return new Promise((resolve, reject) => {
      pendingRequests.set(requestId, { resolve, reject });
      tauriSocket.send(JSON.stringify({
        type: "command",
        name,
        request_id: requestId,
        ...payload,
      }));
    });
  }

  window.DesktopBridge = {
    async kind() {
      return ensureBridgeReady();
    },

    async sendText(text) {
      const kind = await ensureBridgeReady();
      if (kind === "tauri") {
        return sendCommand("send_text", { text });
      }
      return window.pywebview.api.send_text(text);
    },

    async interrupt() {
      const kind = await ensureBridgeReady();
      if (kind === "tauri") {
        return sendCommand("interrupt");
      }
      return window.pywebview.api.interrupt();
    },

    async toggleMute() {
      const kind = await ensureBridgeReady();
      if (kind === "tauri") {
        return sendCommand("toggle_mute");
      }
      return window.pywebview.api.toggle_mute();
    },

    async getRuntimeState() {
      const kind = await ensureBridgeReady();
      if (kind === "tauri") {
        return sendCommand("get_runtime_state");
      }
      return window.pywebview.api.get_runtime_state();
    },

    async toggleDebug() {
      const kind = await ensureBridgeReady();
      if (kind === "tauri") {
        return sendCommand("toggle_debug");
      }
      return window.pywebview.api.toggle_debug();
    },
  };
})();
