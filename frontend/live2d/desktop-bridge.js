(function bootstrapDesktopBridge() {
  const url = new URL(window.location.href);
  const queryBackendPort = Number(url.searchParams.get("backendPort")) || 8765;
  let qtBridge = null;
  let qtBridgePromise = null;
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

  function loadQtWebChannelScript() {
    return new Promise((resolve, reject) => {
      if (typeof window.QWebChannel !== "undefined") {
        resolve();
        return;
      }

      const existing = document.querySelector('script[data-qt-webchannel="true"]');
      if (existing) {
        existing.addEventListener("load", () => resolve(), { once: true });
        existing.addEventListener("error", () => reject(new Error("Failed to load qwebchannel.js")), { once: true });
        return;
      }

      const script = document.createElement("script");
      script.src = "qrc:///qtwebchannel/qwebchannel.js";
      script.async = false;
      script.dataset.qtWebchannel = "true";
      script.addEventListener("load", () => resolve(), { once: true });
      script.addEventListener("error", () => reject(new Error("Failed to load qwebchannel.js")), { once: true });
      document.head.appendChild(script);
    });
  }

  function hasPywebview() {
    return !!window.pywebview?.api;
  }

  function hasQt() {
    return !!window.qt?.webChannelTransport;
  }

  function hasTauri() {
    return !!window.__TAURI__;
  }

  async function ensureBridgeReady() {
    if (hasQt()) {
      await ensureQtBridge();
      return "qt";
    }

    if (hasPywebview()) {
      return "pywebview";
    }

    if (hasTauri()) {
      await ensureTauriSocket();
      return "tauri";
    }

    for (let attempt = 0; attempt < 100; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 100));
      if (hasQt()) {
        await ensureQtBridge();
        return "qt";
      }
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

  async function ensureQtBridge() {
    if (qtBridge) {
      return qtBridge;
    }

    if (qtBridgePromise) {
      return qtBridgePromise;
    }

    qtBridgePromise = (async () => {
      await loadQtWebChannelScript();

      return new Promise((resolve, reject) => {
        if (typeof window.QWebChannel === "undefined") {
          reject(new Error("QWebChannel is not available"));
          return;
        }

        new window.QWebChannel(window.qt.webChannelTransport, (channel) => {
          const bridge = channel.objects?.desktopBridge;
          if (!bridge) {
            reject(new Error("Qt desktop bridge object is missing"));
            return;
          }

          qtBridge = bridge;
          resolve(bridge);
        });
      });
    })();

    return qtBridgePromise;
  }

  function websocketUrl() {
    return `ws://127.0.0.1:${queryBackendPort}`;
  }

  function dispatchFrontendEvent(handlerName, ...args) {
    window[handlerName]?.(...args);
  }

  function normalizeLayoutMode(layout) {
    return layout === "expanded" ? "expanded" : "compact";
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

  async function callQt(method, ...args) {
    const bridge = await ensureQtBridge();
    return new Promise((resolve, reject) => {
      const callback = (payload) => {
        try {
          const parsed = payload ? JSON.parse(payload) : null;
          if (parsed?.status === "error") {
            reject(new Error(parsed.message || `${method} failed`));
            return;
          }
          resolve(parsed);
        } catch (error) {
          reject(error);
        }
      };

      bridge[method](...args, callback);
    });
  }

  window.DesktopBridge = {
    async kind() {
      return ensureBridgeReady();
    },

    async sendText(text) {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("sendText", text);
      }
      if (kind === "tauri") {
        return sendCommand("send_text", { text });
      }
      return window.pywebview.api.send_text(text);
    },

    async interrupt() {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("interrupt");
      }
      if (kind === "tauri") {
        return sendCommand("interrupt");
      }
      return window.pywebview.api.interrupt();
    },

    async toggleMute() {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("toggleMute");
      }
      if (kind === "tauri") {
        return sendCommand("toggle_mute");
      }
      return window.pywebview.api.toggle_mute();
    },

    async getRuntimeState() {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("getRuntimeState");
      }
      if (kind === "tauri") {
        return sendCommand("get_runtime_state");
      }
      return window.pywebview.api.get_runtime_state();
    },

    async toggleDebug() {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("toggleDebug");
      }
      if (kind === "tauri") {
        return sendCommand("toggle_debug");
      }
      return window.pywebview.api.toggle_debug();
    },

    async setLayoutMode(layout) {
      const normalizedLayout = normalizeLayoutMode(layout);
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("setLayoutMode", normalizedLayout);
      }
      if (kind === "pywebview" && typeof window.pywebview.api.set_layout_mode === "function") {
        return window.pywebview.api.set_layout_mode(normalizedLayout);
      }
      return { status: "ok", layout: normalizedLayout };
    },

    async setHudInteractiveRect(rect) {
      const kind = await ensureBridgeReady();
      const normalized = {
        x: Math.round(Number(rect?.x) || 0),
        y: Math.round(Number(rect?.y) || 0),
        width: Math.max(0, Math.round(Number(rect?.width) || 0)),
        height: Math.max(0, Math.round(Number(rect?.height) || 0)),
      };
      if (kind === "qt") {
        return callQt(
          "setHudInteractiveRect",
          normalized.x,
          normalized.y,
          normalized.width,
          normalized.height,
        );
      }
      return { status: "ignored", rect: normalized };
    },

    async startDrag(screenX, screenY) {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("startDrag", Number(screenX) || 0, Number(screenY) || 0);
      }
      return { status: "ignored" };
    },

    async dragMove(screenX, screenY) {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("dragMove", Number(screenX) || 0, Number(screenY) || 0);
      }
      return { status: "ignored" };
    },

    async endDrag() {
      const kind = await ensureBridgeReady();
      if (kind === "qt") {
        return callQt("endDrag");
      }
      return { status: "ignored" };
    },
  };
})();
