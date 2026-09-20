# Veille conversation naturelle (fin de tour, AEC, barge-in) — 20 septembre 2026

**Périmètre** : rendre naturel un assistant vocal desktop local (Windows/Python, VAD Silero → ASR → LLM → TTS, RTX 4070 12 Go déjà chargée), cible FR+EN, tout en CPU ou presque. Chiffres constructeurs ≠ mesures indépendantes.

## 1. Détection de fin de tour (end-of-turn)

| Modèle | Date | Entrée | Langues | Taille / latence CPU | Licence | Hors framework ? |
|---|---|---|---|---|---|---|
| **Smart Turn v3.2** (Daily/Pipecat) | 07/01/2026 | audio brut | **23 dont FR** | 8 Mo int8, ~8M params, ~12 ms | BSD-2-Clause | **Oui** (ONNX, `predict_endpoint()`) |
| LiveKit Turn Detector v1.0 | 17/06/2026 | audio+sémantique | 14 dont FR | v1 Cloud ; v1-mini CPU | code Apache-2.0, poids LiveKit Model License | v1 = Cloud ; v1-mini lié à `livekit-agents` |
| LiveKit text multilingual (legacy) | 12/12/2025 | **texte** | 14 dont FR | Qwen2.5-0.5B INT8, ~400 Mo, ~25 ms | open-weights | déprécié, plugin LiveKit |
| Krisp Turn Prediction v3 | 06/05/2026 | audio | 12+ dont FR | ~9M, 30 Mo | **propriétaire** (VIVA SDK) | non open-weights |
| Krisp Interruption Prediction v1 | 06/05/2026 | audio | EN only | ~6M, 24 Mo | propriétaire | non open-weights |
| TEN Turn Detection | 2025 | **texte** | EN/ZH | Qwen2.5-7B | « Other » | HF, sans FR |
| VAP / MaAI 0.2.0 | 17/04/2026 | audio **stéréo** | EN/JA/ZH | CPU, MIT | MIT | oui, mais 2 micros, sans FR |

Précision publique Smart Turn v3.2 : **FR 94,09 % / F1 0,943** (v3.1 : 95,29 %). LiveKit v1.0 : 9,9 % de fausses coupures à 300 ms. Krisp v3 : +47 % de vrais turn-shifts <200 ms vs v2.

**Le plus simple en Python custom : Smart Turn v3.2** — un ONNX + `onnxruntime`, entrée waveform 16 kHz (≤8 s), sortie binaire, sans transcript, licence permissive. À combiner avec Silero VAD (`stop_secs≈0.2 s`) et un timeout de silence en secours.

## 2. Annulation d'écho (AEC) Python / Windows

| Option | Paquet pip | État 2026 | Double-talk | Licence |
|---|---|---|---|---|
| **WebRTC APM (AEC3)** | `pywebrtc-audio` 0.2.0 (03/09/2026) | wheels **Windows x86_64**, py3.10–3.14, AEC3+NS+AGC+VAD, int16/float32 | ~154× temps réel (M3, pas de bench Windows) ; beta | Apache-2.0 |
| WebRTC APM (LiveKit) | `livekit` (ex livekit-rtc) | `rtc.AudioProcessingModule` + `MediaDevices` (sounddevice), frames 10 ms | AEC3 | Apache-2.0 |
| SpeexDSP | `speexdsp` 0.1.1 | **pas de wheel Windows** | < AEC3 | BSD |
| AEC neuronal | `DTLN-aec` (TFLite/ONNX) | 1,8–10,4M params, robuste aux délais | <8 ms/frame visé | code dispo |
| Windows natif | WASAPI « Communications » (APO) ; Voice Capture DSP `CLSID_CWMAudioAEC` ; `IAcousticEchoCancellationControl` Win11 22621+ | dépend du pilote | non exposé par PortAudio | — |

## 3. Barge-in / interruptions (frameworks 2026)

- **Pipecat** : `InterruptionFrame` annule le LLM, vide la file TTS/lecture, **commit le texte réellement prononcé**. Anti-backchannel : `MinWordsUserTurnStartStrategy(min_words=3)` (actif seulement quand le bot parle) ou `KrispVivaIPUserTurnStartStrategy`.
- **LiveKit** : `interruption.mode` `"adaptive"` vs `"vad"` ; `min_duration=0.5 s`, `min_words`, `false_interruption_timeout=2.0 s`, `resume_false_interruption=True`, `backchannel_boundary=(1.0,1.0)` ; historique tronqué à la portion entendue.
- **Kyutai Unmute** : VAD (`pause_prediction<0.4`) + mot STT, cooldown 3 s « pour éviter les problèmes d'écho » — sans AEC, le barge-in est fragile.

## 4. Full-duplex natif

- **Moshi 7B** : full-duplex, 160/200 ms, CC-BY-4.0, mais backbone Helium entraîné **en anglais uniquement** → pas de FR.
- **MiniCPM-o 4.5** (03/02/2026) : 9B full-duplex, <12 Go RAM, RTF 0.21 (RTX 4090 quantifié), mais parole temps réel **officiellement EN/ZH**.

**Verdict Q4 : non** — pas de FR en parole et les 12 Go sont déjà occupés.

## 5. Verdict — combinaison recommandée

1. **AEC** : `pip install pywebrtc-audio` (AEC3), `far` = exactement le PCM envoyé aux haut-parleurs. Repli : `pip install livekit` + `sounddevice`.
2. **Fin de tour** : Silero VAD (`stop_secs=0.2`) **+ Smart Turn v3.2 ONNX CPU** (`onnxruntime`, ≤8 s) + timeout de silence.
3. **Barge-in** : `min_duration≈0.4 s` **et** `min_words≥2` sur le partiel ASR (rejette « mm-hm ») ; sur vraie interruption → annuler LLM+TTS, vider la file, **tronquer l'historique au texte prononcé** ; reprise après ~2 s de silence si fausse interruption.
4. **Full-duplex** : rejeté.

**Pièges Windows** : frames de 10 ms obligatoires pour l'APM ; `stream_delay`/`set_stream_delay_ms` correct sinon annulation mauvaise (surtout Bluetooth) ; PortAudio/sounddevice **n'expose pas** le mode WASAPI « Communications » ; `pywebrtc-audio` jeune (beta 0.2.0, ~12 étoiles) ; Smart Turn exige VAD court et contexte ≤8 s.

**Non vérifié** : double-talk réel de `pywebrtc-audio` (aucun benchmark indépendant) ; LiveKit v1 hors Cloud et v1-mini hors `livekit-agents` ; mode Communications WASAPI depuis Python ; licence exacte TEN ; conditions/pricing Krisp VIVA.

Sources : daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms ; daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses ; huggingface.co/pipecat-ai/smart-turn-v3 ; livekit.com/blog/solving-end-of-turn-detection ; docs.livekit.io/agents/logic/turns/turn-detector ; huggingface.co/livekit/turn-detector ; krisp.ai/blog/voice-ai-turn-taking-interruption-prediction ; github.com/TEN-framework/ten-turn-detection ; github.com/MaAI-Kyoto/MaAI ; pypi.org/project/pywebrtc-audio ; docs.livekit.io/reference/python/livekit/rtc/apm ; docs.livekit.io/reference/python/livekit/rtc/media_devices ; github.com/breizhn/DTLN-aec ; learn.microsoft.com/windows/win32/coreaudio/aecmicarray ; docs.pipecat.ai/pipecat/fundamentals/interruptions ; docs.livekit.io/agents/logic/turns/tuning ; github.com/kyutai-labs/moshi ; arxiv.org/abs/2410.00037 ; huggingface.co/openbmb/MiniCPM-o-4_5.
