/* ===========================================
   WebSocket Manager
   Handles connection, reconnection, and messages
   =========================================== */

class WebSocketManager {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        // Event callbacks
        this.onStatusChange = null;
        this.onMessage = null;
        this.onStreamStart = null;
        this.onStreamChunk = null;
        this.onStreamEnd = null;
        this.onAudio = null;
        this.onError = null;
        
        // Live2D callbacks
        this.onAudioWithLipSync = null;  // For Live2D integration
        this.onExpressionChange = null;  // For Live2D expressions
    }
    
    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('Already connected');
            return;
        }
        
        this._updateStatus('connecting');
        
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this._updateStatus('connected');
        };
        
        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            this._updateStatus('disconnected');
            this._attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (this.onError) {
                this.onError(error);
            }
        };
        
        this.ws.onmessage = (event) => {
            this._handleMessage(event.data);
        };
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close(1000, 'User disconnected');
            this.ws = null;
        }
    }
    
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
            return true;
        }
        console.error('WebSocket not connected');
        return false;
    }
    
    sendText(text) {
        return this.send({
            type: 'text',
            content: text
        });
    }
    
    sendAudioStream(samples) {
        // Send raw PCM samples for server-side VAD
        return this.send({
            type: 'audio_stream',
            samples: samples
        });
    }
    
    sendMicStop() {
        // Notify server that user manually stopped mic
        return this.send({
            type: 'mic_stop'
        });
    }
    
    sendAudio(base64Audio) {
        // Fallback: send WebM blob
        return this.send({
            type: 'audio',
            data: base64Audio
        });
    }
    
    sendClear() {
        return this.send({
            type: 'clear'
        });
    }
    
    sendPreloadModels() {
        // Request server to preload VAD, ASR, TTS models
        return this.send({
            type: 'preload_models'
        });
    }
    
    ping() {
        return this.send({
            type: 'ping'
        });
    }
    
    _handleMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'text_start':
                    if (this.onTextStart) {
                        this.onTextStart();
                    }
                    break;
                    
                case 'text_chunk':
                    if (this.onTextChunk) {
                        this.onTextChunk(message.content);
                    }
                    break;
                    
                case 'text_end':
                    if (this.onTextEnd) {
                        this.onTextEnd(message.full_text);
                    }
                    break;
                    
                case 'audio_start':
                    console.log('Audio generation started');
                    break;
                    
                case 'audio_data':
                    // Check if we have Live2D lip-sync data
                    if (this.onAudioWithLipSync && message.lip_sync) {
                        this.onAudioWithLipSync(message);
                    } else if (this.onAudioData) {
                        this.onAudioData(message.data);
                    }
                    break;
                    
                case 'expression_change':
                    // Live2D expression change
                    console.log('Expression change:', message.expression);
                    if (this.onExpressionChange) {
                        this.onExpressionChange(message);
                    }
                    break;
                    
                case 'audio_end':
                    console.log('Audio generation complete');
                    break;
                    
                case 'transcription':
                    if (this.onTranscription) {
                        this.onTranscription(message.text);
                    }
                    break;
                    
                case 'transcribing':
                    console.log('Transcribing audio...');
                    if (this.onTranscribing) {
                        this.onTranscribing();
                    }
                    break;
                    
                case 'vad_start':
                    console.log('VAD: Speech started');
                    if (this.onVadStart) {
                        this.onVadStart();
                    }
                    break;
                    
                case 'vad_end':
                    console.log('VAD: Speech ended');
                    if (this.onVadEnd) {
                        this.onVadEnd();
                    }
                    break;
                    
                case 'error':
                    console.error('Server error:', message.message);
                    if (this.onError) {
                        this.onError(new Error(message.message));
                    }
                    break;
                    
                case 'pong':
                    console.log('Pong received');
                    break;
                    
                case 'cleared':
                    console.log('Conversation cleared');
                    break;
                    
                case 'models_loading':
                    console.log('🔄 Models loading...', message.progress || 0, '%');
                    if (this.onModelsLoading) {
                        this.onModelsLoading(message.message, message.progress || 0);
                    }
                    break;
                    
                case 'model_loading':
                    console.log(`📦 Loading ${message.model}...`, message.progress || 0, '%');
                    if (this.onModelLoading) {
                        this.onModelLoading(message.model, message.message, message.progress || 0);
                    }
                    break;
                    
                case 'model_loaded':
                    console.log(`✅ ${message.model} loaded!`, message.progress || 0, '%');
                    if (this.onModelLoaded) {
                        this.onModelLoaded(message.model, message.message, message.progress || 0);
                    }
                    break;
                    
                case 'models_ready':
                    console.log('✅ All models ready!');
                    if (this.onModelsReady) {
                        this.onModelsReady(message.message);
                    }
                    break;
                    
                case 'models_error':
                    console.error('❌ Model loading error:', message.message);
                    if (this.onModelsError) {
                        this.onModelsError(message.message);
                    }
                    break;
                    
                default:
                    console.warn('Unknown message type:', message.type);
            }
            
        } catch (error) {
            console.error('Failed to parse message:', error);
        }
    }
    
    _updateStatus(status) {
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
    }
    
    _attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnect attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
                this.connect();
            }
        }, delay);
    }
    
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}

// Export for use in other modules
window.WebSocketManager = WebSocketManager;
