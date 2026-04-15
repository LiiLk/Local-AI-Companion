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
        this.stopFlushDelayMs = 120;
        this.currentSource = null;
        this.playbackGeneration = 0;
        this.captureMode = 'stream';

        // Client-side speech segmentation for stable pipeline mode.
        this.noiseFloor = 0.004;
        this.speechMinThreshold = 0.014;
        this.speechThresholdMultiplier = 3.0;
        this.speechReleaseFactor = 0.7;
        this.speechStartFrames = 4;
        this.speechEndFrames = 18;
        this.preRollFrames = 6;
        this.maxUtteranceMs = 12000;
        this.speechDetected = false;
        this.speechActiveFrames = 0;
        this.speechSilenceFrames = 0;
        this.preSpeechBuffers = [];
        this.speechBuffers = [];
        this.utteranceStartedAt = 0;

        // Target sample rate for Silero VAD (16kHz)
        this.targetSampleRate = 16000;

        // Callbacks
        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onAudioSamples = null;  // Called with audio samples for streaming
        this.onVolumeChange = null;
        this.onPlaybackStart = null;
        this.onPlaybackEnd = null;
        this.onSpeechStart = null;
        this.onSpeechEnd = null;
        this.onSpeechSegment = null;
    }

    setCaptureMode(mode) {
        this.captureMode = mode === 'client_vad' ? 'client_vad' : 'stream';
        this._resetSpeechDetection();
    }

    async initMicrophone() {
        try {
            // Request the rawest microphone signal possible.
            // Browser DSP is good for calls, but it can eat short word endings
            // and destabilize local VAD/ASR.
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: this.targetSampleRate,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
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

        this._resetSpeechDetection();

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
            const pcmBuffer = this._floatToPcm16Buffer(resampled);

            if (this.captureMode === 'client_vad') {
                const frameDurationMs = (resampled.length / this.targetSampleRate) * 1000;
                this._processClientSpeechFrame(pcmBuffer, volume, frameDurationMs);
                return;
            }

            // Send PCM16 bytes to the server to avoid JSON float array overhead.
            if (this.onAudioSamples) {
                this.onAudioSamples(pcmBuffer);
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

    async stopRecording() {
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

        if (this.captureMode === 'client_vad') {
            this._finalizeSpeechSegment('manual_stop');
        }

        // Give the last browser audio callback / websocket send a short chance
        // to flush before we force-close the utterance server-side.
        await new Promise(resolve => setTimeout(resolve, this.stopFlushDelayMs));

        if (this.onRecordingStop) {
            await this.onRecordingStop();
        }

        console.log('Recording stopped');
    }

    _getSpeechThreshold() {
        return Math.max(this.speechMinThreshold, this.noiseFloor * this.speechThresholdMultiplier);
    }

    _updateNoiseFloor(volume) {
        const clamped = Math.max(0.001, Math.min(volume, 0.05));
        this.noiseFloor = (this.noiseFloor * 0.92) + (clamped * 0.08);
    }

    _isSpeechFrame(volume) {
        return volume >= this._getSpeechThreshold();
    }

    _processClientSpeechFrame(buffer, volume, frameDurationMs) {
        const frameCopy = buffer.slice(0);
        const now = performance.now();
        const speechThreshold = this._getSpeechThreshold();
        const releaseThreshold = speechThreshold * this.speechReleaseFactor;

        if (!this.speechDetected) {
            if (!this._isSpeechFrame(volume)) {
                this._updateNoiseFloor(volume);
            }

            this.preSpeechBuffers.push(frameCopy);
            if (this.preSpeechBuffers.length > this.preRollFrames) {
                this.preSpeechBuffers.shift();
            }

            if (volume >= speechThreshold) {
                this.speechActiveFrames += 1;
            } else {
                this.speechActiveFrames = 0;
            }

            if (this.speechActiveFrames >= this.speechStartFrames) {
                this.speechDetected = true;
                this.speechActiveFrames = 0;
                this.speechSilenceFrames = 0;
                this.utteranceStartedAt = now;
                this.speechBuffers = this.preSpeechBuffers.slice();
                this.preSpeechBuffers = [];
                if (this.onSpeechStart) {
                    this.onSpeechStart();
                }
            }
            return;
        }

        this.speechBuffers.push(frameCopy);

        if (volume >= releaseThreshold) {
            this.speechSilenceFrames = 0;
        } else {
            this.speechSilenceFrames += 1;
        }

        if ((now - this.utteranceStartedAt) >= this.maxUtteranceMs) {
            this._finalizeSpeechSegment('max_duration');
            return;
        }

        if (this.speechSilenceFrames * frameDurationMs >= 380) {
            this._finalizeSpeechSegment('silence');
        }
    }

    _finalizeSpeechSegment(reason) {
        const hadSpeech = this.speechDetected || this.speechBuffers.length > 0;
        const utterance = hadSpeech ? this._concatBuffers(this.speechBuffers) : null;
        const durationMs = utterance ? (utterance.byteLength / 2 / this.targetSampleRate) * 1000 : 0;

        this._resetSpeechDetection();

        if (!hadSpeech) {
            return;
        }

        if (this.onSpeechEnd) {
            this.onSpeechEnd(reason);
        }

        if (utterance && durationMs >= 250 && this.onSpeechSegment) {
            this.onSpeechSegment(utterance, this.targetSampleRate, { durationMs, reason });
        }
    }

    _concatBuffers(buffers) {
        const totalLength = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
        const merged = new Uint8Array(totalLength);
        let offset = 0;
        for (const buffer of buffers) {
            merged.set(new Uint8Array(buffer), offset);
            offset += buffer.byteLength;
        }
        return merged.buffer;
    }

    _resetSpeechDetection() {
        this.speechDetected = false;
        this.speechActiveFrames = 0;
        this.speechSilenceFrames = 0;
        this.preSpeechBuffers = [];
        this.speechBuffers = [];
        this.utteranceStartedAt = 0;
    }

    async toggleRecording() {
        if (this.isRecording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
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
            this.currentSource = null;
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
        const generation = this.playbackGeneration;

        try {
            const playbackContext = this._ensurePlaybackContext();
            const arrayBuffer = this._toArrayBuffer(audioInput);

            // Decode and play
            const audioBuffer = await playbackContext.decodeAudioData(arrayBuffer);
            if (generation !== this.playbackGeneration) {
                return;
            }
            const source = playbackContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(playbackContext.destination);
            this.currentSource = source;

            source.onended = () => {
                if (this.currentSource === source) {
                    this.currentSource = null;
                }
                if (generation !== this.playbackGeneration) {
                    return;
                }
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
        this.playbackGeneration += 1;
        this.audioQueue = [];
        const source = this.currentSource;
        this.currentSource = null;
        this.isPlaying = false;
        if (source) {
            try {
                source.stop(0);
            } catch (error) {
                console.debug('Audio source already stopped:', error);
            }
        }
        if (this.onPlaybackEnd) {
            this.onPlaybackEnd();
        }
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
