---
applyTo: '**'
---

# 🤖 Local AI Companion - Instructions pour Claude

## 📋 Vision du Projet

Ce projet est une **reconstruction pédagogique** d'un assistant IA vocal/visuel inspiré de [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber).

### Objectifs Principaux
- **100% local et privé** : Aucune donnée ne quitte la machine de l'utilisateur (sauf Edge TTS en fallback)
- **Pédagogique** : Comprendre chaque composant en profondeur
- **Modulaire** : Architecture extensible avec interfaces abstraites

### Philosophie
> "Apprendre en recréant from scratch, pas juste forker."

---

## 🏗️ Architecture du Projet

```
┌─────────────────────────────────────────────────────────────┐
│                 CONVERSATION VOCALE COMPLÈTE                 │
│                                                              │
│  🎤 Input vocal → Whisper/Canary (ASR)                      │
│       │                                                      │
│       ▼                                                      │
│  🧠 LLM (Ollama - llama3.2:3b) - Personnalité: Aria         │
│       │                                                      │
│       ▼                                                      │
│  🔊 TTS (Kokoro local / Edge cloud)                         │
│       │                                                      │
│       ▼                                                      │
│  🎧 Output vocal                                             │
│                                                              │
│  🔒 100% LOCAL - Rien ne quitte le PC !                     │
└─────────────────────────────────────────────────────────────┘
```

### Modules
| Module | Localisation | Pattern | Technologies |
|--------|--------------|---------|--------------|
| **LLM** | `src/llm/` | Strategy | Ollama (llama3.2:3b) |
| **TTS** | `src/tts/` | Strategy | Kokoro (local), Edge TTS (cloud) |
| **ASR** | `src/asr/` | Strategy | Faster-Whisper, Canary (NVIDIA) |
| **VAD** | `src/vad/` | - | Silero VAD |
| **Server** | `src/server/` | - | FastAPI + WebSocket |

---

## ✅ Bonnes Pratiques à Respecter

### 1. Architecture & Design Patterns

- **Interfaces abstraites (ABC)** : Chaque module DOIT avoir une classe de base abstraite
- **Pattern Strategy** : Permettre le changement d'implémentation sans modifier le code client
- **SOLID** :
  - **S**ingle Responsibility : Une classe = une responsabilité
  - **O**pen/Closed : Ouvert à l'extension, fermé à la modification
  - **L**iskov Substitution : Les sous-classes doivent être substituables
  - **I**nterface Segregation : Interfaces spécifiques plutôt que générales
  - **D**ependency Inversion : Dépendre des abstractions, pas des implémentations

- **Lazy Loading** : Les modèles lourds (Whisper, Kokoro, LLM) sont chargés au premier usage
- **Configuration externalisée** : Utiliser `config/config.yaml`, jamais de valeurs hardcodées

### 2. Software Engineering

- **Type hints** : TOUJOURS utiliser les annotations de type Python
- **Dataclasses** : Pour les structures de données propres
- **Async/await** : Pour toutes les opérations I/O (streaming, WebSocket, HTTP)
- **Docstrings** : Documenter les fonctions et classes publiques
- **Logging** : Utiliser `logging` avec des niveaux appropriés (DEBUG, INFO, WARNING, ERROR)
- **Gestion des erreurs** : Try/except avec messages explicites, jamais de `pass` silencieux
- **Tests** : Écrire des tests unitaires pour les nouveaux modules

### 3. AI Engineering

- **Modularité des providers** : Chaque provider ASR/TTS/LLM est interchangeable
- **Streaming** : Privilégier le streaming pour une meilleure UX (réponse progressive)
- **Gestion mémoire** : Libérer les ressources GPU/CPU après usage si possible
- **Anti-hallucination ASR** : Utiliser VAD filter, thresholds, prompts guidés
- **Prompt Engineering** : Le system prompt définit la personnalité (config.yaml)

### 4. Code Style

```python
# ✅ BON : Type hints, docstring, async
async def transcribe(self, audio_path: Path, language: str = "fr") -> ASRResult:
    """Transcrit un fichier audio en texte.
    
    Args:
        audio_path: Chemin vers le fichier audio
        language: Code langue ISO (default: fr)
        
    Returns:
        ASRResult avec le texte transcrit et métadonnées
    """
    ...

# ❌ MAUVAIS : Pas de types, pas de doc
def transcribe(self, audio_path, language):
    ...
```

---

## 📚 Ressources de Référence

Consulter ces ressources pour les décisions techniques :

### LLM & AI
- [Ollama Documentation](https://ollama.com/) - LLM local
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - ASR optimisé
- [Kokoro TTS](https://github.com/hexgrad/kokoro) - TTS local 82M params
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection

### Backend
- [FastAPI](https://fastapi.tiangolo.com/) - Documentation officielle
- [WebSockets](https://websockets.readthedocs.io/) - Protocole temps réel
- [Starlette](https://www.starlette.io/) - ASGI framework

### Python Best Practices
- [PEP 8](https://peps.python.org/pep-0008/) - Style guide
- [PEP 484](https://peps.python.org/pep-0484/) - Type hints
- [Real Python](https://realpython.com/) - Tutoriels avancés

### Architecture
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) - Robert C. Martin
- [Design Patterns Python](https://refactoring.guru/design-patterns/python) - Refactoring Guru

---

## 🗺️ Roadmap

### Phase 1 : Fondations ✅ COMPLÈTE
- [x] Structure projet avec interfaces abstraites
- [x] Module LLM (Ollama)
- [x] Module TTS (Edge + Kokoro)
- [x] Module ASR (Whisper + Canary)
- [x] Conversation vocale complète CLI

### Phase 2 : Interface Web 🚧 EN COURS
- [x] Serveur WebSocket (FastAPI)
- [x] Frontend web basique (HTML/CSS/JS)
- [x] Streaming audio bidirectionnel
- [x] VAD automatique (Silero VAD)
- [ ] Intégration Live2D avatar

### Phase 3 : Fonctionnalités avancées 📅 FUTUR
- [ ] Voice Cloning (XTTS v2)
- [ ] Vision (screen capture, camera)
- [ ] Mémoire persistante
- [ ] Contrôle du PC
- [ ] Desktop Pet mode

---

## 🎓 Profil Utilisateur

| Aspect | Niveau |
|--------|--------|
| Python | Intermédiaire |
| IA/ML | Débutant → Intermédiaire |
| Architecture | Débutant → Intermédiaire |
| Git | Intermédiaire |

**Style préféré** : Explications détaillées, comprendre le "pourquoi" avant le code.

---

## 💡 Directives pour Claude

### Approche Pédagogique
1. **Expliquer les concepts** avant de coder
2. **Montrer l'architecture** et le "pourquoi" des choix techniques
3. **Coder étape par étape** avec des commentaires explicatifs
4. **Proposer des alternatives** quand pertinent
5. **Mettre à jour la documentation** (README, CLAUDE.md) si nécessaire
6. **Toujours pusher les modifications** sur une branche dédiée avant merge
7. **Vérifier la cohérence** avec le style et l'architecture existants

### Avant de Coder
- Analyser le contexte existant (lire les fichiers pertinents)
- Vérifier la cohérence avec l'architecture existante
- S'assurer de respecter les patterns établis (Strategy, ABC, etc.)

### Qualité du Code
- Respecter les conventions de nommage existantes
- Utiliser les mêmes bibliothèques que le reste du projet
- Ajouter des logs pour le debugging
- Gérer les erreurs proprement

### Quand Hésiter
- **Consulter les ressources web** (documentation officielle, Stack Overflow, GitHub issues)
- Proposer plusieurs solutions avec pros/cons
- Demander clarification si la demande est ambiguë

---

## 🔧 Stack Technique Actuelle

| Composant | Technologie | Statut |
|-----------|-------------|--------|
| Language | Python 3.11+ | ✅ |
| LLM | Ollama (llama3.2:3b) | ✅ |
| TTS | Kokoro (local) + Edge (cloud) | ✅ |
| ASR | Faster-Whisper + Canary | ✅ |
| VAD | Silero VAD | ✅ |
| Backend | FastAPI + WebSocket | ✅ |
| Frontend | HTML/CSS/JS vanilla | ✅ |
| Config | PyYAML | ✅ |
| HTTP | httpx (async) | ✅ |

---

## ⚠️ Points d'Attention

1. **cuDNN** : Incompatibilité connue avec faster-whisper sur CUDA → CPU par défaut
2. **Modèles français** : Préférer `french-distil-dec2/dec4` pour le français
3. **Canary** : Requiert GPU NVIDIA avec 6GB+ VRAM
4. **Edge TTS** : Seul composant cloud (fallback), préférer Kokoro