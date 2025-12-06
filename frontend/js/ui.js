/**
 * UI Controller - Gestion des interactions de l'interface utilisateur
 * 
 * Ce module gère :
 * - Le toggle sidebar (desktop/mobile)
 * - Le panneau de paramètres
 * - Le switch de thème (dark/light)
 * - Les mises à jour du statut des modèles
 * - Les animations et transitions
 */

class UIController {
    constructor() {
        // Éléments du DOM
        this.sidebar = document.getElementById('sidebar');
        this.settingsPanel = document.getElementById('settings-panel');
        this.overlay = document.getElementById('overlay');
        this.themeToggle = document.getElementById('theme-toggle');
        this.mobileMenuBtn = document.getElementById('mobile-menu-btn');
        this.settingsBtn = document.getElementById('settings-btn');
        this.closeSettingsBtn = document.getElementById('close-settings');
        this.scrollToBottomBtn = document.getElementById('scroll-to-bottom');
        this.messagesContainer = document.getElementById('messages-container');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.newChatBtn = document.getElementById('new-chat-btn');
        
        // Éléments de statut
        this.connectionStatus = document.querySelector('.connection-status');
        this.statusText = this.connectionStatus?.querySelector('.status-text');
        this.modelItems = document.querySelectorAll('.model-item');
        
        // Éléments de la barre de statut
        this.statusBar = document.getElementById('status-bar');
        this.statusMessage = document.getElementById('status-message');
        this.progressFill = document.getElementById('progress-fill');
        
        // État
        this.theme = localStorage.getItem('theme') || 'dark';
        this.isSidebarOpen = false;
        this.isSettingsOpen = false;
        
        // Initialisation
        this.init();
    }
    
    /**
     * Initialisation des événements et de l'état initial
     */
    init() {
        // Appliquer le thème sauvegardé
        this.applyTheme(this.theme);
        
        // Event listeners
        this.bindEvents();
        
        // Observer le scroll pour le bouton "scroll to bottom"
        this.setupScrollObserver();
        
        // Auto-resize du textarea
        this.setupTextareaResize();
        
        console.log('[UI] Controller initialisé');
    }
    
    /**
     * Liaison des événements
     */
    bindEvents() {
        // Theme toggle
        this.themeToggle?.addEventListener('click', () => this.toggleTheme());
        
        // Mobile menu
        this.mobileMenuBtn?.addEventListener('click', () => this.toggleSidebar());
        
        // Settings panel
        this.settingsBtn?.addEventListener('click', () => this.openSettings());
        this.closeSettingsBtn?.addEventListener('click', () => this.closeSettings());
        
        // Overlay (ferme sidebar mobile et settings)
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
        document.getElementById('save-settings')?.addEventListener('click', () => this.saveSettings());
        document.getElementById('cancel-settings')?.addEventListener('click', () => this.closeSettings());
        
        // Nav items sidebar
        document.getElementById('nav-settings')?.addEventListener('click', () => this.openSettings());
    }
    
    /**
     * Toggle du thème dark/light
     */
    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme(this.theme);
        localStorage.setItem('theme', this.theme);
        console.log(`[UI] Thème changé: ${this.theme}`);
    }
    
    /**
     * Applique le thème
     * @param {string} theme - 'dark' ou 'light'
     */
    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        
        // Mettre à jour l'icône du toggle
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
     * Ferme la sidebar
     */
    closeSidebar() {
        this.isSidebarOpen = false;
        this.sidebar?.classList.remove('open');
        if (!this.isSettingsOpen) {
            this.overlay?.classList.remove('visible');
        }
    }
    
    /**
     * Ouvre le panneau de paramètres
     */
    openSettings() {
        this.isSettingsOpen = true;
        this.settingsPanel?.classList.add('open');
        this.overlay?.classList.add('visible');
        this.closeSidebar();
    }
    
    /**
     * Ferme le panneau de paramètres
     */
    closeSettings() {
        this.isSettingsOpen = false;
        this.settingsPanel?.classList.remove('open');
        if (!this.isSidebarOpen) {
            this.overlay?.classList.remove('visible');
        }
    }
    
    /**
     * Configure l'observer de scroll
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
     * Scroll vers le bas des messages
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
     * Configure l'auto-resize du textarea
     */
    setupTextareaResize() {
        if (!this.messageInput) return;
        
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
        });
    }
    
    /**
     * Gestion des raccourcis clavier
     * @param {KeyboardEvent} e 
     */
    handleKeyboard(e) {
        // Escape ferme les panneaux
        if (e.key === 'Escape') {
            this.closeSettings();
            this.closeSidebar();
        }
        
        // Ctrl+Enter envoie le message
        if (e.ctrlKey && e.key === 'Enter') {
            this.sendBtn?.click();
        }
        
        // Ctrl+, ouvre les paramètres
        if (e.ctrlKey && e.key === ',') {
            e.preventDefault();
            this.openSettings();
        }
    }
    
    /**
     * Met à jour le statut de connexion
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
     * Met à jour le statut d'un modèle
     * @param {string} modelType - 'vad', 'asr', ou 'tts'
     * @param {string} status - 'idle', 'loading', 'loaded', 'error'
     * @param {string} [info] - Information supplémentaire
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
     * Affiche la barre de statut
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
     * Cache la barre de statut
     */
    hideStatusBar() {
        this.statusBar?.classList.add('hidden');
    }
    
    /**
     * Nouveau chat
     */
    handleNewChat() {
        // Vider les messages (garder le welcome)
        const messages = document.getElementById('messages');
        if (messages) {
            const welcomeMessage = messages.querySelector('.message.assistant');
            messages.innerHTML = '';
            if (welcomeMessage) {
                messages.appendChild(welcomeMessage);
            }
        }
        
        // Fermer la sidebar mobile
        this.closeSidebar();
        
        // Focus sur l'input
        this.messageInput?.focus();
        
        console.log('[UI] Nouveau chat');
    }
    
    /**
     * Sauvegarde les paramètres
     */
    saveSettings() {
        const settings = {
            asr: {
                provider: document.getElementById('setting-asr-provider')?.value,
                language: document.getElementById('setting-asr-language')?.value
            },
            tts: {
                provider: document.getElementById('setting-tts-provider')?.value,
                autoDetect: document.getElementById('setting-tts-autodetect')?.checked,
                streaming: document.getElementById('setting-tts-streaming')?.checked
            },
            character: {
                name: document.getElementById('setting-char-name')?.value,
                personality: document.getElementById('setting-char-personality')?.value
            }
        };
        
        // Sauvegarder en localStorage
        localStorage.setItem('aria-settings', JSON.stringify(settings));
        
        // Émettre un événement custom pour que app.js puisse réagir
        window.dispatchEvent(new CustomEvent('settings-changed', { detail: settings }));
        
        // Fermer le panneau
        this.closeSettings();
        
        console.log('[UI] Paramètres sauvegardés:', settings);
    }
    
    /**
     * Charge les paramètres depuis localStorage
     */
    loadSettings() {
        const saved = localStorage.getItem('aria-settings');
        if (!saved) return;
        
        try {
            const settings = JSON.parse(saved);
            
            // Appliquer aux champs
            if (settings.asr) {
                const provider = document.getElementById('setting-asr-provider');
                const language = document.getElementById('setting-asr-language');
                if (provider) provider.value = settings.asr.provider || 'whisper';
                if (language) language.value = settings.asr.language || 'auto';
            }
            
            if (settings.tts) {
                const provider = document.getElementById('setting-tts-provider');
                const autoDetect = document.getElementById('setting-tts-autodetect');
                const streaming = document.getElementById('setting-tts-streaming');
                if (provider) provider.value = settings.tts.provider || 'kokoro';
                if (autoDetect) autoDetect.checked = settings.tts.autoDetect !== false;
                if (streaming) streaming.checked = settings.tts.streaming !== false;
            }
            
            if (settings.character) {
                const name = document.getElementById('setting-char-name');
                const personality = document.getElementById('setting-char-personality');
                if (name) name.value = settings.character.name || 'Aria';
                if (personality) personality.value = settings.character.personality || '';
            }
            
            console.log('[UI] Paramètres chargés:', settings);
        } catch (e) {
            console.error('[UI] Erreur chargement paramètres:', e);
        }
    }
}

// Export pour utilisation dans app.js
window.UIController = UIController;

// Initialisation au chargement du DOM
document.addEventListener('DOMContentLoaded', () => {
    window.uiController = new UIController();
    window.uiController.loadSettings();
});
