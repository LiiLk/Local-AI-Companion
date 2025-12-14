/**
 * UI Controller - Manages user interface interactions
 * 
 * This module handles:
 * - Sidebar toggling (desktop/mobile)
 * - The settings panel
 * - Theme switching (dark/light)
 * - Model status updates
 * - Animations and transitions
 */

class UIController {
    constructor() {
        // DOM elements
        this.sidebar = document.getElementById('sidebar');
        this.settingsPanel = document.getElementById('settings-panel');
        this.overlay = document.getElementById('overlay');
        this.themeToggle = document.getElementById('theme-toggle');
        this.mobileMenuBtn = document.getElementById('mobile-menu-btn');
        this.settingsBtn = document.getElementById('settings-btn');
        this.closeSettingsBtn = document.getElementById('close-settings-btn');
        this.scrollToBottomBtn = document.getElementById('scroll-to-bottom');
        this.messagesContainer = document.getElementById('messages-container');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.newChatBtn = document.getElementById('new-chat-btn');

        // Status elements
        this.connectionStatus = document.querySelector('.connection-status');
        this.statusText = this.connectionStatus?.querySelector('.status-text');
        this.modelItems = document.querySelectorAll('.model-item');

        // Status bar elements
        this.statusBar = document.getElementById('status-bar');
        this.statusMessage = document.getElementById('status-message');
        this.progressFill = document.getElementById('progress-fill');

        // State
        this.theme = localStorage.getItem('theme') || 'dark';
        this.isSidebarOpen = false;
        this.isSettingsOpen = false;

        // Initialization
        this.init();
    }

    /**
     * Initialization of events and initial state
     */
    init() {
        // Apply saved theme
        this.applyTheme(this.theme);

        // Event listeners
        this.bindEvents();

        // Scroll observer for "scroll to bottom" button
        this.setupScrollObserver();

        // Auto-resize textarea
        this.setupTextareaResize();

        console.log('[UI] Controller initialisé');
    }

    /**
     * Event binding
     */
    bindEvents() {
        // Theme toggle
        this.themeToggle?.addEventListener('click', () => this.toggleTheme());

        // Mobile menu
        this.mobileMenuBtn?.addEventListener('click', () => this.toggleSidebar());

        // Settings panel
        this.settingsBtn?.addEventListener('click', () => this.openSettings());
        this.closeSettingsBtn?.addEventListener('click', () => this.closeSettings());

        // Overlay (closes mobile sidebar and settings)
        this.overlay?.addEventListener('click', () => {
            this.closeSidebar();
            this.closeSettings();
        });

        // Scroll to bottom
        this.scrollToBottomBtn?.addEventListener('click', () => this.scrollToBottom());

        // New chat
        this.newChatBtn?.addEventListener('click', () => this.handleNewChat());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Settings form
        document.getElementById('save-settings-btn')?.addEventListener('click', () => this.saveSettings());
        document.getElementById('reset-settings-btn')?.addEventListener('click', () => this.resetSettings());

        // Clear chat
        document.getElementById('clear-chat-btn')?.addEventListener('click', () => this.handleNewChat());

        // Enable/disable send button based on input
        this.messageInput?.addEventListener('input', () => {
            const hasContent = this.messageInput.value.trim().length > 0;
            this.sendBtn.disabled = !hasContent;
        });
    }

    /**
     * Toggle dark/light theme
     */
    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme(this.theme);
        localStorage.setItem('theme', this.theme);
        console.log(`[UI] Thème changé: ${this.theme}`);
    }

    /**
     * Apply theme
     * @param {string} theme - 'dark' or 'light'
     */
    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);

        // Update toggle icon
        const icon = this.themeToggle?.querySelector('svg');
        if (icon) {
            if (theme === 'dark') {
                icon.innerHTML = `
                    <circle cx="12" cy="12" r="5"></circle>
                    <line x1="12" y1="1" x2="12" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="23"></line>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                    <line x1="1" y1="12" x2="3" y2="12"></line>
                    <line x1="21" y1="12" x2="23" y2="12"></line>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                `;
            } else {
                icon.innerHTML = `
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                `;
            }
        }
    }

    /**
     * Toggle sidebar (mobile)
     */
    toggleSidebar() {
        this.isSidebarOpen = !this.isSidebarOpen;
        this.sidebar?.classList.toggle('open', this.isSidebarOpen);
        this.overlay?.classList.toggle('visible', this.isSidebarOpen);
    }

    /**
     * Close sidebar
     */
    closeSidebar() {
        this.isSidebarOpen = false;
        this.sidebar?.classList.remove('open');
        if (!this.isSettingsOpen) {
            this.overlay?.classList.remove('visible');
        }
    }

    /**
     * Open settings panel
     */
    openSettings() {
        this.isSettingsOpen = true;
        this.settingsPanel?.classList.add('open');
        this.overlay?.classList.add('visible');
        this.closeSidebar();
    }

    /**
     * Close settings panel
     */
    closeSettings() {
        this.isSettingsOpen = false;
        this.settingsPanel?.classList.remove('open');
        if (!this.isSidebarOpen) {
            this.overlay?.classList.remove('visible');
        }
    }

    /**
     * Configure scroll observer
     */
    setupScrollObserver() {
        if (!this.messagesContainer) return;

        this.messagesContainer.addEventListener('scroll', () => {
            const { scrollTop, scrollHeight, clientHeight } = this.messagesContainer;
            const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

            this.scrollToBottomBtn?.classList.toggle('hidden', isNearBottom);
        });
    }

    /**
     * Scroll to bottom of messages
     */
    scrollToBottom() {
        if (this.messagesContainer) {
            this.messagesContainer.scrollTo({
                top: this.messagesContainer.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    /**
     * Configure auto-resize textarea
     */
    setupTextareaResize() {
        if (!this.messageInput) return;

        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
        });
    }

    /**
     * Handle keyboard shortcuts
     * @param {KeyboardEvent} e 
     */
    handleKeyboard(e) {
        // Escape closes panels
        if (e.key === 'Escape') {
            this.closeSettings();
            this.closeSidebar();
        }

        // Ctrl+Enter sends message
        if (e.ctrlKey && e.key === 'Enter') {
            this.sendBtn?.click();
        }

        // Ctrl+, opens settings
        if (e.ctrlKey && e.key === ',') {
            e.preventDefault();
            this.openSettings();
        }
    }

    /**
     * Update connection status
     * @param {boolean} connected 
     */
    updateConnectionStatus(connected) {
        if (!this.connectionStatus) return;

        this.connectionStatus.classList.remove('connected', 'disconnected');
        this.connectionStatus.classList.add(connected ? 'connected' : 'disconnected');

        if (this.statusText) {
            this.statusText.textContent = connected ? 'Connecté' : 'Déconnecté';
        }
    }

    /**
     * Update model status
     * @param {string} modelType - 'vad', 'asr', or 'tts'
     * @param {string} status - 'idle', 'loading', 'loaded', 'error'
     * @param {string} [info] - Additional information
     */
    updateModelStatus(modelType, status, info = '') {
        const modelItem = document.querySelector(`.model-item[data-model="${modelType}"]`);
        if (!modelItem) return;

        modelItem.setAttribute('data-status', status);

        const badge = modelItem.querySelector('.model-badge');
        if (badge) {
            switch (status) {
                case 'loading':
                    badge.textContent = 'Chargement...';
                    break;
                case 'loaded':
                    badge.textContent = info || 'Prêt';
                    break;
                case 'error':
                    badge.textContent = 'Erreur';
                    break;
                default:
                    badge.textContent = 'Inactif';
            }
        }
    }

    /**
     * Show status bar
     * @param {string} message 
     * @param {number} [progress] - 0-100
     */
    showStatusBar(message, progress = null) {
        if (!this.statusBar) return;

        this.statusBar.classList.remove('hidden');
        if (this.statusMessage) {
            this.statusMessage.textContent = message;
        }

        if (progress !== null && this.progressFill) {
            this.progressFill.style.width = `${progress}%`;
        }
    }

    /**
     * Hide status bar
     */
    hideStatusBar() {
        this.statusBar?.classList.add('hidden');
    }

    /**
     * Handle new chat
     */
    handleNewChat() {
        // Clear messages (keep welcome)
        const messages = document.getElementById('messages');
        if (messages) {
            const welcomeMessage = messages.querySelector('.message.assistant');
            messages.innerHTML = '';
            if (welcomeMessage) {
                messages.appendChild(welcomeMessage);
            }
        }

        // Close mobile sidebar
        this.closeSidebar();

        // Focus on input
        this.messageInput?.focus();

        console.log('[UI] Nouveau chat');
    }

    /**
     * Save settings
     */
    saveSettings() {
        const settings = {
            asr: {
                provider: document.getElementById('asr-provider')?.value,
                language: document.getElementById('asr-language')?.value
            },
            tts: {
                provider: document.getElementById('tts-provider')?.value,
                autoDetect: document.getElementById('auto-detect-language')?.checked,
                streaming: document.getElementById('stream-tts')?.checked
            },
            character: {
                name: document.getElementById('character-name')?.value,
                personality: document.getElementById('character-prompt')?.value
            }
        };

        // Save to localStorage
        localStorage.setItem('aria-settings', JSON.stringify(settings));

        // Emit custom event for app.js to react
        window.dispatchEvent(new CustomEvent('settings-changed', { detail: settings }));

        // Close settings panel
        this.closeSettings();

        console.log('[UI] Settings saved:', settings);
    }

    /**
     * Reset settings to default
     */
    resetSettings() {
        // Default values
        document.getElementById('asr-provider').value = 'parakeet';
        document.getElementById('asr-language').value = '';
        document.getElementById('tts-provider').value = 'f5tts';
        document.getElementById('auto-detect-language').checked = true;
        document.getElementById('stream-tts').checked = true;
        document.getElementById('character-name').value = 'Juri';
        document.getElementById('character-prompt').value = 'You are Juri Han from Street Fighter. You are sarcastic, sadistic, but helpful.';

        // Remove from localStorage
        localStorage.removeItem('aria-settings');

        console.log('[UI] Settings reset');
    }

    /**
     * Load settings from localStorage
     */
    loadSettings() {
        const saved = localStorage.getItem('aria-settings');
        if (!saved) return;

        try {
            const settings = JSON.parse(saved);

            // Apply to fields
            if (settings.asr) {
                const provider = document.getElementById('asr-provider');
                const language = document.getElementById('asr-language');
                if (provider) provider.value = settings.asr.provider || 'parakeet';
                if (language) language.value = settings.asr.language || '';
            }

            if (settings.tts) {
                const provider = document.getElementById('tts-provider');
                const autoDetect = document.getElementById('auto-detect-language');
                const streaming = document.getElementById('stream-tts');
                if (provider) provider.value = settings.tts.provider || 'f5tts';
                if (autoDetect) autoDetect.checked = settings.tts.autoDetect !== false;
                if (streaming) streaming.checked = settings.tts.streaming !== false;
            }

            if (settings.character) {
                const name = document.getElementById('character-name');
                const personality = document.getElementById('character-prompt');
                if (name) name.value = settings.character.name || 'Juri';
                if (personality) personality.value = settings.character.personality || '';
            }

            console.log('[UI] Settings loaded:', settings);
        } catch (e) {
            console.error('[UI] Error loading settings:', e);
        }
    }
}

// Export for use in app.js
window.UIController = UIController;

// Initialization on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.uiController = new UIController();
    window.uiController.loadSettings();
});
