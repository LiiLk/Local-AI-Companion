/* ===========================================
   Audio Manager
   Handles microphone recording and audio playback
   With real-time streaming to server for VAD
   =========================================== */

class AudioManager {
    constructor() {
        this.stream = null;
        this.audioContext = null;
        this.playbackContext = null;
        this.processor = null;
        this.source = null;
        this.isRecording = false;
        this.audioQueue = [];
        this.isPlaying = false;

        // Target sample rate for Silero VAD (16kHz)
        this.targetSampleRate = 16000;

        // Callbacks
        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onAudioSamples = null;  // Called with audio samples for streaming
        this.onVolumeChange = null;
        this.onPlaybackStart = null;
        this.onPlaybackEnd = null;
    }

    async initMicrophone() {
        try {
            // Request microphone at native sample rate
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            console.log(`Microphone initialized (native: ${this.audioContext.sampleRate}Hz, target: ${this.targetSampleRate}Hz)`);
            return true;
        } catch (error) {
            console.error('Failed to access microphone:', error);
            return false;
        }
    }

    async startRecording() {
        if (this.isRecording) {
            console.warn('Already recording');
            return;
        }

        if (!this.stream) {
            const initialized = await this.initMicrophone();
            if (!initialized) {
                throw new Error('Microphone not available');
            }
        }

        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Create source from microphone
        this.source = this.audioContext.createMediaStreamSource(this.stream);

        // Keep capture chunks small enough to reduce end-to-end latency.
        const bufferSize = 1024;
        this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

        // Resampler state
        const nativeSampleRate = this.audioContext.sampleRate;

        this.processor.onaudioprocess = (event) => {
            if (!this.isRecording) return;

            const inputData = event.inputBuffer.getChannelData(0);

            // Calculate volume for visual feedback
            let sum = 0;
            for (let i = 0; i < inputData.length; i++) {
                sum += Math.abs(inputData[i]);
            }
            const volume = sum / inputData.length;

            if (this.onVolumeChange) {
                this.onVolumeChange(volume);
            }

            // Resample to 16kHz
            const resampled = this._resample(inputData, nativeSampleRate, this.targetSampleRate);

            // Send PCM16 bytes to the server to avoid JSON float array overhead.
            if (this.onAudioSamples) {
                this.onAudioSamples(this._floatToPcm16Buffer(resampled));
            }
        };

        // Connect: source -> processor -> destination
        this.source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);

        this.isRecording = true;

        if (this.onRecordingStart) {
            this.onRecordingStart();
        }

        console.log('Recording started (streaming to server)');
    }

    _resample(inputData, fromRate, toRate) {
        if (fromRate === toRate) {
            return inputData;
        }

        const ratio = toRate / fromRate;
        const newLength = Math.round(inputData.length * ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const position = i / ratio;
            const index = Math.floor(position);
            const fraction = position - index;

            if (index + 1 < inputData.length) {
                // Linear interpolation
                result[i] = inputData[index] * (1 - fraction) + inputData[index + 1] * fraction;
            } else {
                result[i] = inputData[index] || 0;
            }
        }

        return result;
    }

    _floatToPcm16Buffer(floatData) {
        const pcm16 = new Int16Array(floatData.length);
        for (let i = 0; i < floatData.length; i++) {
            const sample = Math.max(-1, Math.min(1, floatData[i]));
            pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }
        return pcm16.buffer;
    }

    stopRecording() {
        if (!this.isRecording) {
            console.warn('Not recording');
            return;
        }

        this.isRecording = false;

        // Disconnect audio nodes
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        if (this.source) {
            this.source.disconnect();
            this.source = null;
        }

        if (this.onRecordingStop) {
            this.onRecordingStop();
        }

        console.log('Recording stopped');
    }

    toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }

    async playAudio(audioInput) {
        // Add to queue
        this.audioQueue.push(audioInput);

        // Start playing if not already
        if (!this.isPlaying) {
            this._playNext();
        }
    }

    async _playNext() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            // Notify playback end
            if (this.onPlaybackEnd) {
                this.onPlaybackEnd();
            }
            return;
        }

        // Notify start if this is the first item (or we just started playing)
        // Wait, logic: isPlaying was false at start of playAudio, called _playNext.
        // So checking if we are starting a sequence:
        if (!this.isPlaying && this.onPlaybackStart) {
            this.onPlaybackStart();
        }

        this.isPlaying = true;
        const audioInput = this.audioQueue.shift();

        try {
            const playbackContext = this._ensurePlaybackContext();
            const arrayBuffer = this._toArrayBuffer(audioInput);

            // Decode and play
            const audioBuffer = await playbackContext.decodeAudioData(arrayBuffer);
            const source = playbackContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(playbackContext.destination);

            source.onended = () => {
                this._playNext();
            };

            // Trigger start callback if specifically defined for *chunk* or just *sequence*?
            // User wants to know "AI is speaking".
            // So onPlaybackStart should fire once when set becomes true.
            // My logic above fires it IF !isPlaying. But I set isPlaying = true right after.
            // But _playNext is recursive.
            // If I set isPlaying=true, the next recursive call will see isPlaying=true.
            // So onPlaybackStart only fires for the first chunk. Correct.

            if (this.audioQueue.length === 0 && this.audioQueue.length === 0) {
                // Trigger start if this is the very first one?
                // I moved the check up.
            }

            source.start(0);

        } catch (error) {
            console.error('Failed to play audio:', error);
            this._playNext();
        }
    }

    stopPlayback() {
        this.audioQueue = [];
    }

    _ensurePlaybackContext() {
        if (!this.playbackContext) {
            this.playbackContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.playbackContext.state === 'suspended') {
            this.playbackContext.resume();
        }
        return this.playbackContext;
    }

    _toArrayBuffer(audioInput) {
        if (audioInput instanceof ArrayBuffer) {
            return audioInput.slice(0);
        }

        if (ArrayBuffer.isView(audioInput)) {
            return audioInput.buffer.slice(
                audioInput.byteOffset,
                audioInput.byteOffset + audioInput.byteLength
            );
        }

        const audioData = atob(audioInput);
        const arrayBuffer = new ArrayBuffer(audioData.length);
        const uint8Array = new Uint8Array(arrayBuffer);
        for (let i = 0; i < audioData.length; i++) {
            uint8Array[i] = audioData.charCodeAt(i);
        }
        return arrayBuffer;
    }

    cleanup() {
        this.stopRecording();
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        if (this.playbackContext) {
            this.playbackContext.close();
            this.playbackContext = null;
        }
    }
}

/* ===========================================
   Streaming Audio Player
   Handles sequential playback of audio chunks
   from omni mode (MiniCPM-o streaming responses)
   =========================================== */

class StreamingAudioPlayer {
    constructor() {
        this.audioContext = null;
        this.queue = [];
        this.isPlaying = false;

        // Callbacks
        this.onPlaybackStart = null;
        this.onPlaybackEnd = null;
        this.onLipSync = null;
    }

    _ensureContext() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
        return this.audioContext;
    }

    _toArrayBuffer(audioInput) {
        if (audioInput instanceof ArrayBuffer) {
            return audioInput.slice(0);
        }

        if (ArrayBuffer.isView(audioInput)) {
            return audioInput.buffer.slice(
                audioInput.byteOffset,
                audioInput.byteOffset + audioInput.byteLength
            );
        }

        const binaryString = atob(audioInput);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    enqueue(audioInput, lipSync, expression) {
        this.queue.push({ audioInput, lipSync, expression });
        if (!this.isPlaying) {
            this._playNext();
        }
    }

    async _playNext() {
        if (this.queue.length === 0) {
            this.isPlaying = false;
            if (this.onPlaybackEnd) {
                this.onPlaybackEnd();
            }
            return;
        }

        if (!this.isPlaying && this.onPlaybackStart) {
            this.onPlaybackStart();
        }
        this.isPlaying = true;

        const item = this.queue.shift();

        try {
            const ctx = this._ensureContext();

            // Decode base64
            const audioBuffer = await ctx.decodeAudioData(this._toArrayBuffer(item.audioInput));
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);

            // Lip-sync via volumes array
            if (item.lipSync && item.lipSync.volumes && this.onLipSync) {
                const volumes = item.lipSync.volumes;
                const chunkMs = item.lipSync.chunk_ms || 50;
                let index = 0;

                const interval = setInterval(() => {
                    if (index < volumes.length) {
                        this.onLipSync(volumes[index]);
                        index++;
                    } else {
                        clearInterval(interval);
                        this.onLipSync(0);
                    }
                }, chunkMs);

                source.onended = () => {
                    clearInterval(interval);
                    this.onLipSync(0);
                    this._playNext();
                };
            } else {
                source.onended = () => {
                    this._playNext();
                };
            }

            source.start(0);
        } catch (error) {
            console.error('StreamingAudioPlayer: playback error:', error);
            this._playNext();
        }
    }

    stop() {
        this.queue = [];
        this.isPlaying = false;
    }

    cleanup() {
        this.stop();
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}

// Export
window.AudioManager = AudioManager;
window.StreamingAudioPlayer = StreamingAudioPlayer;
