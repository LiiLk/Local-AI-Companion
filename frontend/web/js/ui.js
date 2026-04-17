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
        this.settingsDefaults = null;
        this.settingsLoaded = false;

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
        document.getElementById('test-settings-btn')?.addEventListener('click', () => this.testSettings());
        document.getElementById('llm-provider')?.addEventListener('change', () => this.updateProviderFields());

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
    async saveSettings() {
        const settings = this.collectSettingsPayload();
        this.setSettingsBusy(true);
        this.showSettingsFeedback('Enregistrement en cours...', 'info');

        try {
            const response = await fetch('/api/settings/llm', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || result.message || 'Impossible de sauvegarder les paramètres.');
            }

            this.settingsDefaults = result;
            this.settingsLoaded = true;
            this.applySettings(result);
            this.showSettingsFeedback(result.message || 'Paramètres sauvegardés.', 'success');
            window.dispatchEvent(new CustomEvent('settings-changed', { detail: result }));

            window.setTimeout(() => {
                window.location.reload();
            }, 900);
        } catch (error) {
            this.showSettingsFeedback(error.message || 'Impossible de sauvegarder les paramètres.', 'error');
            console.error('[UI] Error saving settings:', error);
        } finally {
            this.setSettingsBusy(false);
        }
    }

    /**
     * Reset settings to default
     */
    resetSettings() {
        if (!this.settingsDefaults) {
            return;
        }

        this.applySettings(this.settingsDefaults);
        this.showSettingsFeedback('Formulaire réinitialisé à la dernière configuration chargée.', 'info');
        console.log('[UI] Settings reset to loaded defaults');
    }

    /**
     * Test current settings without saving them
     */
    async testSettings() {
        const settings = this.collectSettingsPayload();
        this.setSettingsBusy(true);
        this.showSettingsFeedback('Test de connexion en cours...', 'info');

        try {
            const response = await fetch('/api/settings/llm/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || result.message || 'Test de connexion impossible.');
            }

            this.showSettingsFeedback(result.message || 'Connexion réussie.', result.ok ? 'success' : 'error');
        } catch (error) {
            this.showSettingsFeedback(error.message || 'Test de connexion impossible.', 'error');
            console.error('[UI] Error testing settings:', error);
        } finally {
            this.setSettingsBusy(false);
        }
    }

    /**
     * Load settings from backend
     */
    async loadSettings() {
        try {
            const response = await fetch('/api/settings/llm');
            const settings = await response.json();

            if (!response.ok) {
                throw new Error(settings.detail || settings.message || 'Impossible de charger les paramètres.');
            }

            this.settingsDefaults = settings;
            this.settingsLoaded = true;
            this.applySettings(settings);
            console.log('[UI] Settings loaded from backend:', settings);
        } catch (error) {
            this.showSettingsFeedback(error.message || 'Impossible de charger les paramètres.', 'error');
            console.error('[UI] Error loading settings:', error);
        }
    }

    collectSettingsPayload() {
        const provider = document.getElementById('llm-provider')?.value || 'ollama';
        const payload = {
            provider,
            ollama: {
                base_url: document.getElementById('ollama-base-url')?.value?.trim() || '',
                model: document.getElementById('ollama-model')?.value?.trim() || ''
            },
            openrouter: {
                base_url: document.getElementById('openrouter-base-url')?.value?.trim() || '',
                model: document.getElementById('openrouter-model')?.value?.trim() || ''
            }
        };

        const apiKey = document.getElementById('openrouter-api-key')?.value?.trim();
        if (apiKey) {
            payload.openrouter.api_key = apiKey;
        }

        return payload;
    }

    applySettings(settings) {
        const provider = settings?.provider || 'ollama';
        const providerField = document.getElementById('llm-provider');
        const ollama = settings?.ollama || {};
        const openrouter = settings?.openrouter || {};
        const openrouterHint = document.getElementById('openrouter-key-hint');

        if (providerField) providerField.value = provider;
        document.getElementById('ollama-base-url').value = ollama.base_url || 'http://localhost:11434';
        document.getElementById('ollama-model').value = ollama.model || 'llama3.2:3b';
        document.getElementById('openrouter-base-url').value = openrouter.base_url || 'https://openrouter.ai/api/v1';
        document.getElementById('openrouter-model').value = openrouter.model || 'openai/gpt-4.1-mini';
        document.getElementById('openrouter-api-key').value = '';

        if (openrouterHint) {
            if (openrouter.api_key_configured) {
                const source = openrouter.api_key_source || 'saved';
                openrouterHint.textContent = `API key disponible (${source}). Laisse le champ vide pour conserver la clé actuelle.`;
            } else {
                openrouterHint.textContent = "Aucune clé API détectée. Elle sera nécessaire pour activer OpenRouter.";
            }
        }

        this.updateProviderFields();
    }

    updateProviderFields() {
        const provider = document.getElementById('llm-provider')?.value || 'ollama';
        document.getElementById('ollama-fields')?.classList.toggle('hidden', provider !== 'ollama');
        document.getElementById('openrouter-fields')?.classList.toggle('hidden', provider !== 'openrouter');
    }

    showSettingsFeedback(message, type = 'info') {
        const feedback = document.getElementById('settings-feedback');
        if (!feedback) return;

        feedback.textContent = message;
        feedback.classList.remove('hidden', 'is-info', 'is-success', 'is-error');
        feedback.classList.add(`is-${type}`);
    }

    setSettingsBusy(isBusy) {
        const buttonIds = ['save-settings-btn', 'reset-settings-btn', 'test-settings-btn'];
        buttonIds.forEach((id) => {
            const element = document.getElementById(id);
            if (element) {
                element.disabled = isBusy;
            }
        });
    }
}

// Export for use in app.js
window.UIController = UIController;

// Initialization on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.uiController = new UIController();
    window.uiController.loadSettings();
});
