# 🤖 Claude Instructions - Local AI Companion

> Ce fichier documente le projet et sert de guide pour collaborer avec Claude (AI assistant).

## 🎯 Contexte du projet

Ce projet est une **reconstruction pédagogique** d'un assistant IA vocal/visuel inspiré de [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber).

**Objectif principal** : Apprendre en recréant from scratch, pas juste forker.

**Philosophie** : 
- 🔒 **100% local et privé** (pas de cloud APIs)
- 📚 **Pédagogique** (comprendre chaque composant)
- 🧩 **Modulaire** (facile à étendre)

---

## 👤 Profil de l'utilisateur

| Aspect | Niveau |
|--------|--------|
| Python | Intermédiaire |
| IA/ML | Débutant |
| Architecture | Débutant |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND                         │
│         (Web UI / Desktop avec Live2D)              │
└─────────────────────┬───────────────────────────────┘
                      │ WebSocket
┌─────────────────────▼───────────────────────────────┐
│                    BACKEND                          │
│                   (FastAPI)                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │   ASR   │  │   LLM   │  │   TTS   │              │
│  │ (Voix→  │→ │(Cerveau)│→ │ (Texte→ │              │
│  │  Texte) │  │         │  │  Voix)  │              │
│  └─────────┘  └─────────┘  └─────────┘              │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Roadmap

### Phase 1 : Fondations ✅
- [x] Structure projet avec interfaces abstraites
- [x] Module LLM (Ollama)
- [x] Chatbot texte en CLI
- [x] Module TTS (Edge TTS)
- [ ] Module TTS (Kokoro - local, naturel)
- [ ] Module ASR (Whisper)

### Phase 2 : Interface
- [ ] Serveur WebSocket (FastAPI)
- [ ] Frontend web basique
- [ ] Intégration Live2D

### Phase 3 : Fonctionnalités avancées
- [ ] Vision (screen capture)
- [ ] Mémoire persistante
- [ ] Contrôle du PC

---

## 📁 Structure du projet

```
Local-AI-Companion/
├── src/
│   ├── llm/                    # Large Language Models
│   │   ├── base.py             # Interface abstraite BaseLLM
│   │   └── ollama_llm.py       # Implémentation Ollama
│   ├── tts/                    # Text-to-Speech
│   │   ├── base.py             # Interface abstraite BaseTTS
│   │   └── edge_tts_provider.py # Implémentation Edge TTS
│   ├── asr/                    # Automatic Speech Recognition (TODO)
│   └── core/                   # Logique principale (TODO)
├── config/
│   └── config.yaml             # Configuration YAML
├── assets/                     # Images, sons, etc.
├── main.py                     # Point d'entrée CLI
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation publique
└── CLAUDE.md                   # Ce fichier (instructions AI)
```

---

## 🧩 Modules implémentés

### 1. LLM (src/llm/)

**Pattern** : Strategy (interface abstraite + implémentations)

```python
# Interface (base.py)
class BaseLLM(ABC):
    async def chat(messages: list[Message]) -> LLMResponse
    async def chat_stream(messages: list[Message]) -> AsyncGenerator

# Implémentation (ollama_llm.py)  
class OllamaLLM(BaseLLM):
    # Utilise Ollama local avec streaming
```

**Modèle actuel** : `llama3.2:3b` via Ollama

### 2. TTS (src/tts/)

**Pattern** : Strategy (interface abstraite + implémentations)

```python
# Interface (base.py)
class BaseTTS(ABC):
    async def synthesize(text, output_path) -> TTSResult
    async def synthesize_stream(text) -> AsyncGenerator[bytes]
    async def list_voices(language) -> list[Voice]

# Implémentation (edge_tts_provider.py)
class EdgeTTSProvider(BaseTTS):
    # Utilise Microsoft Edge TTS (gratuit, cloud)
```

**Providers prévus** :
| Provider | Local | Qualité | Status |
|----------|-------|---------|--------|
| Edge TTS | ❌ | ⭐⭐⭐ | ✅ Implémenté |
| Kokoro | ✅ | ⭐⭐⭐⭐⭐ | 🔜 À faire |
| Fish Speech | ✅ | ⭐⭐⭐⭐⭐ | 📅 Futur |

---

## 💡 Principes de code

1. **Interfaces abstraites** : Chaque module a une classe de base ABC
2. **Configuration YAML** : Modifiable sans toucher au code
3. **Async/await** : Performance pour I/O (streaming, WebSockets)
4. **Type hints** : Clarté et autocomplétion
5. **Dataclasses** : Structures de données propres
6. **SOLID** : Surtout Open/Closed (facile à étendre)

---

## 🔧 Stack technique

| Composant | Technologie | Statut |
|-----------|-------------|--------|
| LLM | Ollama (llama3.2:3b) | ✅ |
| TTS | Edge TTS → Kokoro | ✅/🔜 |
| ASR | Faster-Whisper | 📅 |
| Backend | FastAPI + WebSockets | 📅 |
| Frontend | HTML/JS + Live2D | 📅 |

---

## 🎓 Approche pédagogique

En tant que mentor, Claude doit :

1. **Expliquer les concepts** avant de coder
2. **Montrer l'architecture** et le "pourquoi" des choix
3. **Coder étape par étape** avec des explications
4. **Encourager les questions** et la compréhension
5. **Proposer des exercices** quand approprié

---

## 📝 Notes de développement

### Session actuelle (Dec 2024)
- ✅ Créé structure projet modulaire
- ✅ Implémenté LLM avec Ollama
- ✅ Implémenté TTS avec Edge TTS
- ✅ Chatbot CLI avec voix
- 🔜 Implémenter Kokoro TTS (débit plus naturel)
- 🔜 Implémenter ASR avec Whisper

### Problèmes résolus
- Edge TTS : Clé `DisplayName` au lieu de `FriendlyName`
- Audio : Installer `ffmpeg` pour lire les MP3 (`ffplay`)
- Rate : Augmenté à +20% pour un débit plus naturel
