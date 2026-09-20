# Veille 2026-09 — Computer Use & perception d'écran en local

Périmètre : agents GUI open-weights exécutables localement sur RTX 4070 12 Go. Date de rédaction : 20/09/2026. Sources vérifiées le 20/09/2026.

## 1. Modèles / agents computer-use open-weights (local)

- **UI-TARS-2** (ByteDance, 04/09/2025) : OSWorld 47.5, WindowsAgentArena 50.6 ; MoE ~23B actifs/230B. Le 7B réellement ouvert est **UI-TARS-1.5-7B** (OSWorld 27.5, ScreenSpot-Pro 49.6, Apache-2.0, ~14 Go FP16 / ~4 Go Q4). App desktop : Windows officiellement « WIP ». https://github.com/bytedance/UI-TARS + https://arxiv.org/abs/2509.02544
- **Fara-7B** (Microsoft, 24/11/2025, MIT) : WebVoyager 73.5, Online-Mind2Web 34.1 ; perçoit par captures, purement web/browser. Q4 GGUF ~4.5–6 Go, ~1–3 s/action sur GPU. **Fara1.5** (22/07/2026) : 4B/9B/27B Qwen3.5 ; 9B = 63.4 Online-Mind2Web / 86.6 WebVoyager. https://huggingface.co/microsoft/Fara-7B + https://github.com/microsoft/fara
- **Holo3 / Holo3.1** (H Company, 31/03/2026 puis 01/06/2026, Apache-2.0) : **Holo3-35B-A3B = 77.8 OSWorld-Verified** (122B = 78.85). Holo3.1 livre des quantifications FP8/Q4-GGUF/NVFP4 et des tailles 0.8B/4B/9B. Le 4B tiendrait sur 12 Go (~10 Go FP8, ~0.8–1.2 s/step ; source tierce). Step moyen 6.8 s→3.3 s sur DGX Spark. https://hcompany.ai/holo3.1
- **OpenCUA** (xLANG, 10/2025, open) : 7B = ScreenSpot-Pro 50.0 ; 72B = OSWorld-Verified 45.0. https://opencua.xlang.ai
- **OmniParser V2** (Microsoft, 12/02/2025) : parseur d'écran (détection + caption), 39.6 ScreenSpot-Pro, **0.6–0.8 s/frame sur RTX 4090** ; YOLOv9-E ajouté 07/2026. OmniTool pilote un Windows 11 VM. https://github.com/microsoft/OmniParser
- **Qwen3-VL** (Qwen, 2025) : agent computer-use, grounding 2D en coordonnées **relatives 1000×1000** (erreurs systématiques si le resize n'est pas synchronisé). Tailles 8B/30B-A3B/235B-A22B. https://github.com/QwenLM/Qwen3-VL
- Grounding historiques : **OS-Atlas-Base-4B/7B** (ICLR 2025, 4B/7B), **ShowUI**, **UGround-V1-7B**.

**Non vérifié** : publication des poids de UI-TARS-2 (230B) ; chiffres per-step du Holo3.1-4B (article tiers).

## 2. « Jev » = V-JEPA / world models

Aucun modèle connu nommé « Jev ». Lecture la plus probable : **V-JEPA**. État réel :
- **V-JEPA 2** (Meta, 11/06/2025) et **V-JEPA 2.1** (16/03/2026, arXiv 2603.14482) : encodeurs vidéo 80M→2B, utilisés en robotique/navigation, **pas pour piloter une GUI**.
- Suites : **JEPA-WMs** (Meta FAIR), **JEPA-WAM** (08/2026), **ThinkJEPA** (03/2026) — ce dernier couple bien « branche JEPA dense + thinker VLM », mais reste recherche pure (trajectoires de mains), sans application computer-use.
- **Je n'ai trouvé AUCUN world model JEPA appliqué au computer-use / desktop en 2026.** L'architecture « petit perceptif rapide + LLM planificateur » existe surtout sous forme UIA/OCR + VLM (voir §3), pas JEPA.

## 3. Hybride UIA + OCR + petit VLM vs pure vision

Sur Windows, l'arbre d'accessibilité (UIA) est **plus fiable et plus rapide** que la pure vision. Plusieurs serveurs MCP 2026 sont **UIA-first** : `desktop-touch-mcp` (UIA en ~2 ms, arbre ~100 ms), `fastcua`, `Desktop-Computer-Use`, `mcp-windows`, `FlaUI-MCP`, `windows-mcp-server`. La vision est le fallback pour Electron/games/RDP. OmniParser+GPT-4o montre que la vision seule reste faible (39.6 ScreenSpot-Pro). Recommandation : **UIA → OCR → petit VLM de grounding** (Qwen3-VL-8B / OS-Atlas-7B), coordonnées pixel en dernier recours.

## 4. Perception continue / proactive

- **screenpipe** (YC S26, source-available) : capture événementielle écran + arbre d'accessibilité + fallback OCR, Whisper local, ~300 Mo/8 h, 100 % local, MCP. https://github.com/screenpipe/screenpipe
- **Open-LLM-VTuber** : Live2D, perception écran, MCP, offline. **AIRI** (MIT) : WebGPU, Live2D/VRM, agent Minecraft.
- Streaming VLM : **StreamingVLM** 8 FPS sur H100 ; **ViCoStream** 134 FPS sur A100 ; échantillonnage simple ~1.77 s/frame sur W7900 48 Go. Sur **12 Go**, la perception continue réaliste = échantillonnage (1 frame / 5–30 s) + UIA, pas un VLM continu.

## 5. Sécurité (2026)

- **CSA note (15/04/2026)** : pop-ups adversariaux = **86 % de réussite** d'attaque ; les défenses par prompt échouent. https://labs.cloudsecurityalliance.org
- **VPI-Bench** (2026) : jusqu'à 51 % (CUA) / 100 % (browser). **AgentHijack** (arXiv 2609.09212, 06/09/2026) : patch visuel local, 84.5 % T-ASR.
- Défense architecturale : **single-shot planning / Dual-LLM** (arXiv 2601.09923) — sépare planner privilégié et perception mise en quarantaine, garde ~57 % de la perf.
- Bonnes pratiques (OWASP AI Agent Cheat Sheet) : allowlist d'outils/domaines, sandbox, confirmation liée aux paramètres, contrôle d'egress, aucun secret dans le contexte, audit. À traiter comme non résolu.

## 6. VERDICT

| Besoin | Local 12 Go viable | Cloud nécessaire |
|---|---|---|
| Contrôle desktop généraliste | Holo3.1-4B (Apache-2.0, ~10 Go), Fara-7B Q4 | Oui pour >55 % OSWorld : Holo3-35B/122B, Fara1.5-27B, GPT-5.4, Claude (Mythos/Opus 4.6) |
| Grounding d'élément | OS-Atlas-7B, Qwen3-VL-8B, OmniParser V2 | Peu utile |
| Perception écran continue | UIA + OCR + VLM échantillonné (screenpipe) | Rarement |
| World model JEPA | Non applicable au desktop | — |

**Viabilité 100 % locale sur 12 Go** : *oui mais partielle*. Un CUA 4–9B quantifié (Holo3.1-4B, Fara1.5-4B, UI-TARS-1.5-7B Q4) exécute des tâches unitaires avec une précision réelle de l'ordre de ~50–70 % selon le type de tâche — insuffisant pour de l'autonomie non surveillée. Le grounding reste plus sûr via UIA. Coût cloud indicatif : Holo3 API 0.25 $/1M in – 1.80 $/1M out ; Fara/Qwen ~0.20 $/1M ; ~0.025 $/tâche pour Fara-7B ; Claude/GPT frontier ~10–50 $/1M. **Conclusion : local pour perception, grounding et sous-tâches ; cloud seulement pour les tâches longues à fort enjeu.**

**Non vérifié** : poids/openness de UI-TARS-2 ; latence Holo3.1-4B (source secondaire) ; absence JEPA-GUI confirmée par recherche, pas par exhaustivité.