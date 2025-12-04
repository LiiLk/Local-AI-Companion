/* ===========================================
   Audio Manager
   Handles microphone recording and audio playback
   With real-time streaming to server for VAD
   =========================================== */

class AudioManager {
    constructor() {
        this.stream = null;
        this.audioContext = null;
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
        
        // Create script processor for raw audio access
        // Buffer size of 4096 gives us ~85ms chunks at 48kHz
        const bufferSize = 4096;
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
            
            // Send samples to server
            if (this.onAudioSamples) {
                // Convert Float32Array to regular array for JSON
                this.onAudioSamples(Array.from(resampled));
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
    
    async playAudio(base64Audio) {
        // Add to queue
        this.audioQueue.push(base64Audio);
        
        // Start playing if not already
        if (!this.isPlaying) {
            this._playNext();
        }
    }
    
    async _playNext() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }
        
        this.isPlaying = true;
        const base64Audio = this.audioQueue.shift();
        
        try {
            // Decode base64 to array buffer
            const audioData = atob(base64Audio);
            const arrayBuffer = new ArrayBuffer(audioData.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            
            for (let i = 0; i < audioData.length; i++) {
                uint8Array[i] = audioData.charCodeAt(i);
            }
            
            // Create separate audio context for playback
            const playbackContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Decode and play
            const audioBuffer = await playbackContext.decodeAudioData(arrayBuffer);
            const source = playbackContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(playbackContext.destination);
            
            source.onended = () => {
                playbackContext.close();
                this._playNext();
            };
            
            source.start(0);
            
        } catch (error) {
            console.error('Failed to play audio:', error);
            this._playNext();
        }
    }
    
    stopPlayback() {
        this.audioQueue = [];
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
    }
}

// Export
window.AudioManager = AudioManager;
