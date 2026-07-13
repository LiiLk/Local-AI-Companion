/**
 * Thin Live2D bridge for the browser web UI (frontend/web).
 *
 * Wraps Live2DManager so the companion avatar can run in a normal Windows
 * browser with GPU WebGL while the Python backend runs in WSL (or Windows).
 *
 * LIL-49: preferred "avatar on the PC" path under WSL = browser, not Qt/WSLg.
 */
(function (global) {
  "use strict";

  class Live2DIntegration {
    constructor() {
      this.ready = false;
      this.onPlaybackStart = null;
      this.onPlaybackEnd = null;
      this._audioContext = null;
      this._currentSource = null;
      this._playing = false;
      this._audioQueue = [];
      this._playbackGeneration = 0;
      this._sequenceActive = false;
    }

    async init(config = {}) {
      if (typeof global.Live2DManager === "undefined") {
        console.error("[Live2DIntegration] Live2DManager is not loaded");
        return false;
      }

      const canvasId = config.canvasId || "live2d-canvas";
      const canvas = document.getElementById(canvasId);
      if (!canvas) {
        console.error("[Live2DIntegration] canvas not found:", canvasId);
        return false;
      }

      // Size canvas to container / window for browser overlay
      const resize = () => {
        const host = canvas.parentElement || document.body;
        const w = Math.max(host.clientWidth || window.innerWidth || 400, 1);
        const h = Math.max(host.clientHeight || window.innerHeight || 600, 1);
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      };
      resize();
      window.addEventListener("resize", resize);

      try {
        await global.Live2DManager.init({
          canvasId,
          modelPath: config.modelPath || "/assets/models/march7th/",
          modelName: config.modelName || "march 7th.model3.json",
          scale: config.scale ?? 0.85,
          position: config.position || { x: 0.5, y: -0.2 },
          debug: !!config.debug,
          defaultExpression: config.defaultExpression || "neutral",
        });
        this.ready = true;
        return true;
      } catch (err) {
        console.error("[Live2DIntegration] init failed:", err);
        this.ready = false;
        return false;
      }
    }

    handleExpressionChange(message) {
      if (!this.ready || !global.Live2DManager) return;
      const name = message?.expression || message?.name || message?.value;
      if (!name) return;
      try {
        global.Live2DManager.setExpression(String(name));
      } catch (err) {
        console.warn("[Live2DIntegration] setExpression failed:", err);
      }
    }

    /**
     * Play TTS audio from a websocket message and drive lip-sync.
     * message: { buffer|data: ArrayBuffer|base64, lip_sync?: number[], expression?: string }
     */
    handleAudioMessage(message) {
      if (!this.ready) return;

      if (!message?.buffer && !message?.data) return;
      this._audioQueue.push(message);
      if (!this._sequenceActive) {
        this._sequenceActive = true;
        if (typeof this.onPlaybackStart === "function") {
          this.onPlaybackStart();
        }
      }
      if (!this._playing) {
        this._playNextAudio();
      }
    }

    async _playNextAudio() {
      if (this._playing || this._audioQueue.length === 0) return;

      const message = this._audioQueue.shift();
      const generation = this._playbackGeneration;
      this._playing = true;

      const expression = message?.expression;
      if (expression) {
        this.handleExpressionChange({ expression });
      }

      let arrayBuffer = message?.buffer || message?.data;
      if (typeof arrayBuffer === "string") {
        arrayBuffer = this._base64ToArrayBuffer(arrayBuffer);
      }

      try {
        if (!this._audioContext) {
          const Ctx = window.AudioContext || window.webkitAudioContext;
          this._audioContext = new Ctx();
        }
        if (this._audioContext.state === "suspended") {
          await this._audioContext.resume();
        }

        const audioBuffer = await this._audioContext.decodeAudioData(arrayBuffer.slice(0));
        if (generation !== this._playbackGeneration) {
          this._playing = false;
          return;
        }

        const source = this._audioContext.createBufferSource();
        source.buffer = audioBuffer;

        // Analyser for lip-sync
        const analyser = this._audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        analyser.connect(this._audioContext.destination);

        const data = new Uint8Array(analyser.frequencyBinCount);
        const tick = () => {
          if (!this._playing || generation !== this._playbackGeneration) return;
          analyser.getByteFrequencyData(data);
          let sum = 0;
          for (let i = 0; i < Math.min(22, data.length); i++) sum += data[i];
          const avg = sum / Math.min(22, data.length);
          const lip = Math.min(1, Math.max(0, (avg / 128 - 0.1) * 1.5));
          global.Live2DManager?.setLipSync?.(lip);
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);

        source.onended = () => {
          if (generation !== this._playbackGeneration) return;
          this._playing = false;
          this._currentSource = null;
          global.Live2DManager?.setLipSync?.(0);
          if (this._audioQueue.length > 0) {
            this._playNextAudio();
          } else {
            this._sequenceActive = false;
            if (typeof this.onPlaybackEnd === "function") {
              this.onPlaybackEnd();
            }
          }
        };

        this._currentSource = source;
        source.start(0);
      } catch (err) {
        console.error("[Live2DIntegration] audio playback failed:", err);
        this._playing = false;
        global.Live2DManager?.setLipSync?.(0);
        if (this._audioQueue.length > 0) {
          this._playNextAudio();
        } else {
          this._sequenceActive = false;
          if (typeof this.onPlaybackEnd === "function") {
            this.onPlaybackEnd();
          }
        }
      }
    }

    stopPlayback() {
      this._playbackGeneration += 1;
      this._audioQueue = [];
      this._playing = false;
      const wasActive = this._sequenceActive;
      this._sequenceActive = false;
      const source = this._currentSource;
      this._currentSource = null;
      if (source) {
        source.onended = null;
        try {
          source.stop(0);
        } catch (_) {
          /* source already stopped */
        }
      }
      global.Live2DManager?.setLipSync?.(0);
      if (wasActive && typeof this.onPlaybackEnd === "function") {
        this.onPlaybackEnd();
      }
    }

    _base64ToArrayBuffer(base64) {
      const cleaned = base64.includes(",") ? base64.split(",")[1] : base64;
      const binary = atob(cleaned);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }
  }

  global.Live2DIntegration = Live2DIntegration;
})(typeof window !== "undefined" ? window : globalThis);
