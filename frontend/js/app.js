/* ===========================================
   Main Application
   Ties everything together
   =========================================== */

class App {
    constructor() {
        // Managers
        this.ws = null;
        this.audio = new AudioManager();
        
        // State
        this.isStreaming = false;
        this.currentStreamContent = '';
        this._modelsPreloaded = false;
        
        // DOM Elements
        this.elements = {
            status: document.getElementById('connection-status'),
            messages: document.getElementById('messages'),
            input: document.getElementById('message-input'),
            sendBtn: document.getElementById('send-button'),
            micBtn: document.getElementById('mic-button'),
            recordingIndicator: document.getElementById('recording-indicator')
        };
        
        this._init();
    }
    
    _init() {
        // Setup WebSocket with unique client ID
        const clientId = 'client_' + Math.random().toString(36).substring(2, 15);
        const wsUrl = `ws://${window.location.host}/ws/${clientId}`;
        this.ws = new WebSocketManager(wsUrl);
        
        // WebSocket callbacks
        this.ws.onStatusChange = (status) => this._updateConnectionStatus(status);
        this.ws.onTextStart = () => this._handleStreamStart();
        this.ws.onTextChunk = (chunk) => this._handleStreamChunk(chunk);
        this.ws.onTextEnd = (fullText) => this._handleStreamEnd(fullText);
        this.ws.onAudioData = (base64) => this.audio.playAudio(base64);
        this.ws.onTranscription = (text) => this._handleTranscription(text);
        this.ws.onTranscribing = () => this._showTranscribingIndicator();
        this.ws.onVadStart = () => this._handleVadStart();
        this.ws.onVadEnd = () => this._handleVadEnd();
        this.ws.onModelsLoading = (msg) => this._showModelsLoading(msg);
        this.ws.onModelsReady = (msg) => this._showModelsReady(msg);
        this.ws.onError = (error) => this._handleError(error);
        
        // Audio callbacks
        this.audio.onRecordingStart = () => this._showRecordingIndicator(true);
        this.audio.onRecordingStop = () => {
            this._showRecordingIndicator(false);
            this.elements.micBtn.classList.remove('recording');
            // Notify server that mic stopped (in case VAD hasn't triggered yet)
            this.ws.sendMicStop();
        };
        this.audio.onAudioSamples = (samples) => {
            // Stream audio samples to server for VAD processing
            this.ws.sendAudioStream(samples);
        };
        this.audio.onVolumeChange = (volume) => this._updateVolumeIndicator(volume);
        
        // Event listeners
        this._setupEventListeners();
        
        // Connect
        this.ws.connect();
    }
    
    _setupEventListeners() {
        // Send button
        this.elements.sendBtn.addEventListener('click', () => this._sendMessage());
        
        // Enter key to send
        this.elements.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this._sendMessage();
            }
        });
        
        // Microphone button
        this.elements.micBtn.addEventListener('click', async () => {
            try {
                // Preload models on first mic click
                if (!this._modelsPreloaded) {
                    this.ws.sendPreloadModels();
                    this._modelsPreloaded = true;
                }
                
                await this.audio.toggleRecording();
                this.elements.micBtn.classList.toggle('recording', this.audio.isRecording);
            } catch (error) {
                this._showError('Microphone access denied');
            }
        });
    }
    
    _sendMessage() {
        const text = this.elements.input.value.trim();
        if (!text) return;
        
        // Add user message to UI
        this._addMessage('user', text);
        
        // Send via WebSocket
        this.ws.sendText(text);
        
        // Clear input
        this.elements.input.value = '';
    }
    
    _handleStreamStart() {
        this.isStreaming = true;
        this.currentStreamContent = '';
        
        // Add placeholder message for streaming
        this._addMessage('assistant', '', true);
    }
    
    _handleStreamChunk(chunk) {
        if (!this.isStreaming) return;
        
        this.currentStreamContent += chunk;
        
        // Update the streaming message
        const streamingMsg = this.elements.messages.querySelector('.message.streaming .message-content');
        if (streamingMsg) {
            streamingMsg.innerHTML = this._formatMessage(this.currentStreamContent);
            this._scrollToBottom();
        }
    }
    
    _handleStreamEnd(fullResponse) {
        this.isStreaming = false;
        
        // Remove streaming class
        const streamingMsg = this.elements.messages.querySelector('.message.streaming');
        if (streamingMsg) {
            streamingMsg.classList.remove('streaming');
        }
    }
    
    _handleTranscription(text) {
        if (text && text.trim()) {
            // Show the transcribed user message
            this._addMessage('user', text);
        }
    }
    
    _showTranscribingIndicator() {
        // Update recording indicator to show transcribing
        const indicator = this.elements.recordingIndicator;
        indicator.classList.remove('hidden');
        indicator.querySelector('span:last-child').textContent = 'Transcription en cours...';
    }
    
    _handleVadStart() {
        // VAD detected speech start - show visual feedback
        const indicator = this.elements.recordingIndicator;
        indicator.querySelector('span:last-child').textContent = 'Parole détectée...';
        this.elements.micBtn.classList.add('active');
    }
    
    _handleVadEnd() {
        // VAD detected speech end - will transcribe now
        const indicator = this.elements.recordingIndicator;
        indicator.querySelector('span:last-child').textContent = 'Transcription...';
        this.elements.micBtn.classList.remove('active');
    }
    
    _handleError(error) {
        console.error('Error:', error);
        this._showError(error.message || 'An error occurred');
    }
    
    _showModelsLoading(message) {
        // Show loading indicator
        this.elements.recordingIndicator.classList.remove('hidden');
        this.elements.recordingIndicator.querySelector('span:last-child').textContent = '🔄 ' + message;
        this.elements.micBtn.classList.add('loading');
    }
    
    _showModelsReady(message) {
        // Update indicator briefly then hide if not recording
        this.elements.recordingIndicator.querySelector('span:last-child').textContent = '✅ ' + message;
        this.elements.micBtn.classList.remove('loading');
        
        // Hide after 1 second if not actively recording
        setTimeout(() => {
            if (!this.audio.isRecording) {
                this.elements.recordingIndicator.classList.add('hidden');
            }
        }, 1000);
    }
    
    _addMessage(role, content, isStreaming = false) {
        const message = document.createElement('div');
        message.className = `message ${role}${isStreaming ? ' streaming' : ''}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🤖';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isStreaming) {
            // Show typing indicator while waiting
            contentDiv.innerHTML = `
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
        } else {
            contentDiv.innerHTML = this._formatMessage(content);
        }
        
        message.appendChild(avatar);
        message.appendChild(contentDiv);
        
        this.elements.messages.appendChild(message);
        this._scrollToBottom();
    }
    
    _formatMessage(text) {
        // Basic formatting - escape HTML and convert newlines
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // Convert newlines to paragraphs
        const paragraphs = escaped.split('\n\n');
        return paragraphs.map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
    }
    
    _scrollToBottom() {
        this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    }
    
    _updateConnectionStatus(status) {
        const statusEl = this.elements.status;
        statusEl.className = `status ${status}`;
        
        const statusTexts = {
            'connected': '🟢 Connected',
            'disconnected': '🔴 Disconnected',
            'connecting': '🟡 Connecting...'
        };
        
        statusEl.textContent = statusTexts[status] || status;
    }
    
    _showRecordingIndicator(show) {
        if (show) {
            this.elements.recordingIndicator.classList.remove('hidden');
        } else {
            this.elements.recordingIndicator.classList.add('hidden');
            // Reset volume indicator
            const pulse = this.elements.recordingIndicator.querySelector('.pulse');
            if (pulse) pulse.style.transform = 'scale(1)';
        }
    }
    
    _updateVolumeIndicator(volume) {
        // Scale the pulse based on volume (0-1)
        const pulse = this.elements.recordingIndicator.querySelector('.pulse');
        if (pulse) {
            const scale = 1 + volume * 3;  // Scale from 1 to 4 based on volume
            pulse.style.transform = `scale(${scale})`;
        }
    }
    
    _showError(message) {
        // You could use a toast notification system here
        alert(message);
    }
    
    // Public method to clear conversation
    clearConversation() {
        this.elements.messages.innerHTML = '';
        this.ws.sendClear();
    }
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
