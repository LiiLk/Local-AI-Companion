/* ===========================================
   Main Application
   Ties everything together - Professional UI Version
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

        // DOM Elements - Updated for new UI structure
        this.elements = {
            // Messages
            messagesContainer: document.getElementById('messages-container'),
            messages: document.getElementById('messages'),

            // Input
            input: document.getElementById('message-input'),
            sendBtn: document.getElementById('send-btn'),
            voiceBtn: document.getElementById('voice-btn'),

            // Status
            statusBar: document.getElementById('status-bar'),
            statusMessage: document.getElementById('status-message'),
            progressFill: document.getElementById('progress-fill'),

            // Connection (sidebar)
            connectionStatus: document.querySelector('.connection-status'),
            statusText: document.querySelector('.connection-status .status-text'),

            // Model status items
            modelVad: document.querySelector('.model-item[data-model="vad"]'),
            modelAsr: document.querySelector('.model-item[data-model="asr"]'),
            modelTts: document.querySelector('.model-item[data-model="tts"]')
        };

        this._init();
    }

    _init() {
        // Setup WebSocket with unique client ID
        const clientId = 'client_' + Math.random().toString(36).substring(2, 15);
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
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
        this.ws.onModelsLoading = (msg, progress) => this._showModelsLoading(msg, progress);
        this.ws.onModelLoading = (model, msg, progress) => this._showModelLoading(model, msg, progress);
        this.ws.onModelLoaded = (model, msg, progress) => this._showModelLoaded(model, msg, progress);
        this.ws.onModelsReady = (msg) => this._showModelsReady(msg);
        this.ws.onModelsError = (msg) => this._showModelsError(msg);
        this.ws.onError = (error) => this._handleError(error);

        // Audio callbacks
        this.audio.onRecordingStart = () => this._handleRecordingStart();
        this.audio.onRecordingStop = () => this._handleRecordingStop();
        this.audio.onAudioSamples = (samples) => {
            // Stream audio samples to server for VAD processing
            this.ws.sendAudioStream(samples);
        };
        this.audio.onVolumeChange = (volume) => this._updateVolumeIndicator(volume);

        // Event listeners
        this._setupEventListeners();

        // Connect
        this.ws.connect();

        // Listen for settings changes from UIController
        window.addEventListener('settings-changed', (e) => this._handleSettingsChange(e.detail));
    }

    _setupEventListeners() {
        // Send button
        this.elements.sendBtn?.addEventListener('click', () => this._sendMessage());

        // Enter key to send (Shift+Enter for new line)
        this.elements.input?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this._sendMessage();
            }
        });

        // Voice button (microphone)
        this.elements.voiceBtn?.addEventListener('click', async () => {
            try {
                // Preload models on first mic click
                if (!this._modelsPreloaded) {
                    this.ws.sendPreloadModels();
                    this._modelsPreloaded = true;
                }

                await this.audio.toggleRecording();
            } catch (error) {
                console.error('[App] Microphone error:', error);
                this._showToast('Accès au microphone refusé', 'error');
            }
        });
    }

    _sendMessage() {
        const text = this.elements.input?.value.trim();
        if (!text) return;

        // Add user message to UI
        this._addMessage('user', text);

        // Send via WebSocket
        this.ws.sendText(text);

        // Clear input and reset height
        this.elements.input.value = '';
        this.elements.input.style.height = 'auto';

        // Disable send button
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = true;
        }
    }

    _handleRecordingStart() {
        // Update voice button state
        this.elements.voiceBtn?.classList.add('recording');

        // Show status bar
        this._showStatus('🎤 Écoute en cours...', 0);
    }

    _handleRecordingStop() {
        // Update voice button state
        this.elements.voiceBtn?.classList.remove('recording');
        this.elements.voiceBtn?.classList.remove('active');

        // Hide status bar
        this._hideStatus();

        // Notify server
        this.ws.sendMicStop();
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
        const streamingMsg = this.elements.messages?.querySelector('.message.streaming .message-content');
        if (streamingMsg) {
            streamingMsg.innerHTML = this._formatMessage(this.currentStreamContent);
            this._scrollToBottom();
        }
    }

    _handleStreamEnd(fullResponse) {
        this.isStreaming = false;

        // Remove streaming class
        const streamingMsg = this.elements.messages?.querySelector('.message.streaming');
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
        this._showStatus('⏳ Transcription en cours...', 50);
    }

    _handleVadStart() {
        // VAD detected speech start - show visual feedback
        this._showStatus('🗣️ Parole détectée...', 25);
        this.elements.voiceBtn?.classList.add('active');
    }

    _handleVadEnd() {
        // VAD detected speech end - will transcribe now
        this._showStatus('⏳ Transcription...', 75);
        this.elements.voiceBtn?.classList.remove('active');
    }

    _handleError(error) {
        console.error('[App] Error:', error);
        this._showToast(error.message || 'Une erreur est survenue', 'error');
    }

    _showModelsLoading(message, progress = 0) {
        this._showStatus(`🔄 ${message}`, progress);
        this.elements.voiceBtn?.classList.add('loading');
    }

    _showModelLoading(model, message, progress = 0) {
        this._updateModelStatus(model, 'loading', message);
        this._showStatus(`⏳ ${message}`, progress);
    }

    _showModelLoaded(model, message, progress = 0) {
        this._updateModelStatus(model, 'loaded', '✓ Prêt');
        this._showStatus(`✅ ${message}`, progress);
    }

    _showModelsReady(message) {
        this._showStatus(`✅ ${message}`, 100);
        this.elements.voiceBtn?.classList.remove('loading');
        this._modelsPreloaded = true;

        // Hide status bar after 2 seconds
        setTimeout(() => {
            if (!this.audio.isRecording) {
                this._hideStatus();
            }
        }, 2000);
    }

    _showModelsError(message) {
        this._showStatus(`❌ ${message}`, 0);
        this.elements.voiceBtn?.classList.remove('loading');
        this.elements.voiceBtn?.classList.add('error');

        // Update model status
        this._updateModelStatus('vad', 'error');
        this._updateModelStatus('asr', 'error');
        this._updateModelStatus('tts', 'error');

        // Remove error class after 3 seconds
        setTimeout(() => {
            this.elements.voiceBtn?.classList.remove('error');
        }, 3000);
    }

    _updateModelStatus(model, status, info = '') {
        if (window.uiController) {
            window.uiController.updateModelStatus(model, status, info);
        }
    }

    _addMessage(role, content, isStreaming = false) {
        if (!this.elements.messages) return;

        const message = document.createElement('div');
        message.className = `message ${role}${isStreaming ? ' streaming' : ''}`;

        // Avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '✨';

        // Message bubble
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        // Content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (isStreaming) {
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

        // Meta (time)
        const meta = document.createElement('div');
        meta.className = 'message-meta';
        const time = document.createElement('span');
        time.className = 'message-time';
        time.textContent = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
        meta.appendChild(time);

        bubble.appendChild(contentDiv);
        bubble.appendChild(meta);

        message.appendChild(avatar);
        message.appendChild(bubble);

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
        if (this.elements.messagesContainer) {
            this.elements.messagesContainer.scrollTo({
                top: this.elements.messagesContainer.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    _updateConnectionStatus(status) {
        // Delegate to UIController
        const connected = status === 'connected';
        if (window.uiController) {
            window.uiController.updateConnectionStatus(connected);
        }
    }

    _showStatus(message, progress = 0) {
        if (window.uiController) {
            window.uiController.showStatusBar(message, progress);
        }
    }

    _hideStatus() {
        if (window.uiController) {
            window.uiController.hideStatusBar();
        }
    }

    _updateVolumeIndicator(volume) {
        // Could add visual feedback for volume if needed
        // For now, the voice button animation handles this
    }

    _showToast(message, type = 'info') {
        // Simple toast notification
        console.log(`[Toast ${type}]`, message);

        // Create toast element
        let toast = document.getElementById('toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'toast';
            toast.style.cssText = `
                position: fixed;
                bottom: 100px;
                left: 50%;
                transform: translateX(-50%);
                padding: 12px 24px;
                background: var(--color-bg-elevated);
                border: 1px solid var(--color-border);
                border-radius: var(--radius-lg);
                color: var(--color-text-primary);
                font-size: var(--font-size-sm);
                z-index: 9999;
                box-shadow: var(--shadow-lg);
                opacity: 0;
                transition: opacity 0.3s ease;
            `;
            document.body.appendChild(toast);
        }

        // Style based on type
        if (type === 'error') {
            toast.style.borderColor = 'var(--color-error)';
        } else if (type === 'success') {
            toast.style.borderColor = 'var(--color-success)';
        }

        toast.textContent = message;
        toast.style.opacity = '1';

        // Auto hide
        setTimeout(() => {
            toast.style.opacity = '0';
        }, 3000);
    }

    _handleSettingsChange(settings) {
        console.log('[App] Settings changed:', settings);
        // Send settings update to server if needed
        // this.ws.sendSettings(settings);
    }

    // Public method to clear conversation
    clearConversation() {
        if (this.elements.messages) {
            // Keep welcome message
            const welcomeMsg = this.elements.messages.querySelector('.message.assistant:first-child');
            this.elements.messages.innerHTML = '';
            if (welcomeMsg) {
                this.elements.messages.appendChild(welcomeMsg);
            }
        }
        this.ws.sendClear();
    }
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
