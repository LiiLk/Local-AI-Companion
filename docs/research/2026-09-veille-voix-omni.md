# Veille voix temps réel et modèles omnimodaux locaux — 20 septembre 2026

**Périmètre** : S2S/omni open-weights locaux, latence cascade, TTS expressifs FR+EN, petits LLM, verdict 12 Go VRAM. Sources publiques de 2025–2026. Les chiffres constructeurs ne sont pas des mesures indépendantes : signalés comme tels.

## 1. Omnimodaux / speech-to-speech locaux

| Modèle | Date | Taille | VRAM / exécution | TTFA revendiqué | Langues | Licence | Full-duplex |
|---|---|---|---|---|---|---|---|
| Qwen3-Omni-30B-A3B | 22/09/2025 | MoE 30B (3B actifs) | ~19 Go Q4 (secondaire), ~69 Go BF16/vidéo (officiel) — **ne tient pas 12 Go** | 234 ms théorique, 1er paquet (arxiv) | 19 compréh. / 10 génération | Apache 2.0 | Non |
| MiniCPM-o 4.5 | 03/02/2026 | 9B | PyTorch ≥28 Go ; llama.cpp-omni INT4 ≥12 Go | ~0,5–0,9 s/unité (A100, PyTorch) | speech EN/ZH | (voir carte HF) | **Oui** |
| Moshi 7B | 2024 | 7B | Q4 ~5–9 Go (communauté) | 160 ms théorique / 200 ms pratique | EN only | CC-BY 4.0 | **Oui** |
| Kyutai Unmute | 05/2025 | cascade STT 1B + TTS 1.6B | config locale | <1 s revendiqué ; TTS 450–750 ms | EN/FR (STT) | MIT/Apache | Non |
| Step-Audio 2 mini | 07/2025 | 8B | — | — | surtout ZH/EN | Apache 2.0 | Non |
| GLM-4-Voice-9B | 2025 | 9B | int4 ~12 Go (communauté) | streaming | ZH/EN only | Model License | Non |
| Sesame CSM-1B | 13/03/2025 | 1B | — | — | EN only | Apache 2.0 | Non |

Sources : github.com/QwenLM/Qwen3-Omni ; arxiv.org/abs/2509.17765 ; huggingface.co/openbmb/MiniCPM-o-4_5 ; arxiv.org/abs/2604.27393 ; github.com/kyutai-labs/moshi ; arxiv.org/abs/2410.00037 ; kyutai.org/unmute ; huggingface.co/stepfun-ai/Step-Audio-2-mini ; github.com/THUDM/GLM-4-Voice ; github.com/SesameAILabs/csm.

**Point dur** : aucun omni natif ne confirme la **génération vocale en français**. Moshi/CSM/GLM/Step-Audio sont EN (ou ZH/EN). C'est le blocage principal pour FR+EN.

## 2. Latence cascade streaming

Budget mesuré (repo NVIDIA/Pipecat « nemotron-january-2026 », ~500–700 ms voix-à-voix) : VAD 200 ms + STT 30–50 ms + LLM 100–150 ms + TTS 185–370 ms. Cerebrium (05/08/2026) : STT local 110 ms vs 250 ms API, LLM auto-hébergé ~300 ms vs 700 ms–1,5 s API, TTS 80 ms vs 150 ms, co-localisation −150 ms → 600 ms E2E. Mesure terrain ai-box.eu (17/06/2026) : ASR Parakeet ~32 ms, TTS ~73 ms, le LLM est le seul poste coûteux (7 s avec un 26B → viser 7–8B).

Techniques : ASR incrémental (chunks 160 ms), flush trick Kyutai, génération préemptive (défaut LiveKit), TTS par chunk sur frontière de phrase, KV cache 100 %, TTS préemptive. Turn detection : LiveKit Turn Detector v1 (17/06/2026, 14 langues dont FR, min_delay 0,3 s) ; smart-turn v3 ONNX CPU ~50–85 ms, 23 langues.

Sources : github.com/pipecat-ai/nemotron-january-2026 ; cerebrium.ai/resources/pipecat-voice-agent-gpu-deployment ; ai-box.eu/en/news/local-voice-agent... ; docs.livekit.io/agents/logic/turns/turn-detector ; livekit.com/blog/solving-end-of-turn-detection ; pypi.org/project/smart-turn-livekit.

## 3. TTS expressifs basse latence (FR+EN)

| Modèle | Taille | FR | Émotion | Clonage | Licence |
|---|---|---|---|---|---|
| Kokoro-82M | 82M | 1 voix, <11 h data | non | non | Apache 2.0 |
| Chatterbox Multilingual V3 | 0,5B | oui | exagération/réglable | oui (quelques s) | MIT |
| Orpheus 3B | 3B | multilingue (recherche) | tags `<laugh>`… | oui | Apache 2.0 |
| Qwen3-TTS 0.6/1.7B | 0,6–1,7B | oui | instruct/langage naturel | oui (3 s) | (voir repo) |
| Fish Audio S1/S2 | 0,5B/4B | oui (tier 2) | 64+ tags `(angry)`… | oui | Fish Research |
| IndexTTS-2.5 | — | **non listé** (zh/en/ja/es/ar) | vecteur/texte, durée | oui | Apache 2.0 |
| MOSS-TTS-Realtime | 1,7B | tag `language="French"` | contrôle | oui | (voir repo) |
| VibeVoice-Realtime-0.5B | 0,5B | EN only | expressive | non | MIT |

Latences : Qwen3-TTS TTFP ~64–97 ms (vLLM-Omni, H200) ; MOSS-TTS-Realtime TTFB 180 ms (02/2026) ; Orpheus ~200 ms ; VibeVoice-Realtime ~300 ms. **Dépassent Kokoro en expressivité+FR** : Chatterbox V3 (10/06/2026, MIT, 23+ langues), Fish S1-mini, Qwen3-TTS.

Sources : huggingface.co/hexgrad/Kokoro-82M ; github.com/resemble-ai/chatterbox ; resemble.ai (10/06/2026) ; github.com/canopyai/Orpheus-TTS ; github.com/QwenLM/Qwen3-TTS ; github.com/fishaudio/fish-speech ; docs.fish.audio ; github.com/index-tts/index-tts ; github.com/OpenMOSS/MOSS-TTS ; github.com/microsoft/VibeVoice.

## 4. Petits LLM (≤14B) pour personnage + tool-calling

- **Qwen3.5-9B** (02–09/03/2026, Apache 2.0, HF) : 262K contexte, tool-calling natif (parser `qwen3_coder`), Q6_K ~9 Go / Q8_0 ~12 Go sur 4070. Meilleur compromis FR+EN+latence.
- Qwen3-14B Q4_K ~9,5 Go — alternative dense.
- Distil labs (02/08/2026) : Qwen et Llama éligibles tool-calling ; Gemma 3 exclu, seuls Gemma 4 E2B/E4B et FunctionGemma 270M conviennent.

Sources : huggingface.co/Qwen/Qwen3.5-9B ; github.com/QwenLM/Qwen3.5 ; unsloth.ai/docs/models/qwen3.5 ; distillabs.ai/learn/qwen-vs-llama-vs-gemma-for-tool-calling.

## 5. VERDICT

**Objectif <1–1,5 s TTFA sur 12 Go : la cascade optimisée gagne aujourd'hui.** Qwen3-Omni (~19 Go Q4) ne rentre pas. MiniCPM-o 4.5 est le seul full-duplex ≤12 Go (llama.cpp-omni INT4) avec ~0,5–0,9 s/unité, mais sa génération vocale FR n'est pas prouvée (EN/ZH probable). Une cascade `Parakeet-TDT-0.6b-v3` (FR: WER 5,15 % Fleurs, CC-BY 4.0) + `Qwen3.5-9B` + `Chatterbox Multilingual V3` (MIT, FR, clone) + turn detector atteint le budget visé.

**Coût cloud** (ordres de grandeur, USD→EUR approximatif) : cascade ~0,02–0,04 $/min ≈ **1,2–2,4 €/h** ; OpenAI Realtime mini 0,02–0,05 $/min ≈ 1,2–3 €/h ; flagship gpt-realtime-2.1 0,06–0,11 $/min ≈ 3,5–6,5 €/h. (Fora Soft 13/07/2026 ; Burki 14/02/2026.) En local : coût ≈ électricité.

**Non vérifié** : TTFA de MiniCPM-o 4.5/Qwen3-Omni sur 4070 réel ; support FR en génération pour les omnis ; TTFP constructeurs (H200/A100, non transposables) ; fiches VRAM secondaires (bestllmfor, fitmyllm) non confirmées par les éditeurs.