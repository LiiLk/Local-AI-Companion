# Analyse comparative : AIRI vs Open-LLM-VTuber (P1–P8)

Date : 2026-09. Méthode : lecture statique des deux dépôts, aucun code exécuté. Chemins relatifs à la racine de chaque dépôt.

## 1. Tableau de synthèse

| Problème | AIRI (TS/Vue/Electron) | Open-LLM-VTuber (Python) |
|---|---|---|
| P1 Tour de parole | Oui — Silero VAD worker, hystérésis 0.3/0.1, silence 400 ms, padding 80 ms | Oui — VAD à compteurs hits/misses, silence fixe ~0,8 s, pre-buffer 20 chunks |
| P2 Barge-in / écho | Non (half-duplex) mais interruption explicite (`stopByIntent`, tour marqué `interrupted`) | Oui côté protocole : `interrupt-signal` + troncature au `heard_response` ; pas d'AEC |
| P3 Latence streaming | Oui, abouti — chunker « boost » + file de lecture priorisée | Oui — phrases pysbd + première phrase à la virgule + livraison ordonnée |
| P4 Émotions | Oui — token structuré `<\|ACT {emotion,motion}\|>` normalisé, retiré avant TTS | Oui — tags texte `[joy]`, extraits puis filtrés avant TTS |
| P5 Avatar / lip-sync | Oui — wLipSync (visèmes AEIOU) + drivers Live2D/VRM/MMD | Partiel — Live2D seul ; RMS envoyé au frontend, pas de visèmes en Python |
| P6 Écran / proactivité | Oui — capture Electron + agent proactif `spark:notify` décisionnel | Partiel — images vers LLM multimodal ; proactivité = trigger externe |
| P7 Outils / MCP | Oui — SDK MCP stdio dans le main Electron + tool-calls | Oui — client MCP stdio + tool-calling natif et repli JSON |
| P8 Mémoire long terme | Partiel — compaction de contexte + résumé ; pgvector = squelette vide | Faible — mémoire en contexte + historique JSON, sans résumé |

## 2. Détail par problème

### P1 — Tour de parole
- **AIRI** : VAD Silero ONNX en worker ; `speechThreshold 0.3` / `exitThreshold 0.1` (hystérésis), `minSilenceDurationMs 400`, `speechPadMs 80`, `minSpeechDurationMs 250` ; émet `speech-start/end/ready`. `apps/stage-web/src/workers/vad/vad.ts:23-41,143-171,204-238`.
- **OLV** : machine `IDLE/ACTIVE/INACTIVE`, `required_hits=3` (~0,1 s) et `required_misses=24` (~0,8 s) → silence fixe ; pre-buffer 20 chunks ; émet `<|PAUSE|>`/`<|RESUME|>`. `src/open_llm_vtuber/vad/silero.py:14-21,103,135-185`.

### P2 — Barge-in et écho
- **OLV** : le client envoie `interrupt-signal` (`src/open_llm_vtuber/websocket_handler.py:44,86,369-392`) ; le serveur annule la tâche asyncio (`conversations/conversation_handler.py:112-127`) et tronque l'historique à ce qui a été entendu : dernière réponse réécrite en `heard_response + "..."` + message `[Interrupted by user]` (`agent/agents/basic_memory_agent.py:195-223`), persisté (`conversation_handler.py:129-143`). Le VAD relaie `<|PAUSE|>` en `control:interrupt` (`websocket_handler.py:497-500`). **Pas d'annulation d'écho** en Python.
- **AIRI** : pas de barge-in micro ; le micro est coupé pendant la parole puis réarmé après cooldown (`apps/stage-tamagotchi/src/renderer/pages/index.vue:756-770`). Aucune contrainte `echoCancellation` trouvée. Mais l'audio a un vrai modèle d'interruption : `PlaybackManager` avec `AbortController`, `stopByIntent`, `stopByOwner`, politiques `steal-oldest`/`steal-lowest-priority` (`packages/pipelines-audio/src/managers/playback-manager.ts:11-13,280-391`) ; un tour partiel est marqué `interrupted: true` (`packages/core-agent/src/runtime/chat-orchestrator-runtime.ts:1004-1008`).

### P3 — Latence LLM → TTS
- **AIRI** : `chunkTtsInput` découpe par grappes de graphèmes : les `boost=2` premiers chunks finissent sur ponctuation faible dès `minimumWords=4`, coupure dure à `maximumWords=12` (`packages/pipelines-audio/src/processors/tts-chunker.ts:14-16,24-45,146-185`). Tokens `<|…|>` extraits avant le texte (`packages/core-agent/src/runtime/llm-marker-parser.ts:1-4,72-124`). Lecture ordonnée/priorisée via `playback-manager.ts`.
- **OLV** : `SentenceDivider` émet dès la ponctuation finale (pysbd) et, pour la première phrase, à la première virgule (`faster_first_response`) : `src/open_llm_vtuber/utils/sentence_divider.py:492-522`. `TTSTaskManager` synthétise en parallèle mais livre dans l'ordre via file à numéros de séquence (`conversations/tts_manager.py:16-28,92-111`).

### P4 — Émotions
- **OLV** : tags inline `[joy]` ; clés valides injectées au prompt depuis `emotionMap` (`src/open_llm_vtuber/service_context.py:457-460`, `live2d_model.py:48-53`). `extract_emotion` les mappe en indices d'expression (`live2d_model.py:146-172`), `actions_extractor` les attache par phrase (`agent/transformers.py:58-100`), retirés du TTS par `filter_brackets` (`utils/tts_preprocessor.py:141-151`) ; ` thinking` exclu (`transformers.py:189-199`).
- **AIRI** : `<|ACT {"emotion":{"name":"happy","intensity":0.8},"motion":"nod"}|>` (`packages/pipelines-audio/src/llm-streaming-control/parsers/act.ts:3,22-49`), normalisé sur un ensemble fixe (happy, sad, angry, think, surprised, awkward, question, curious, neutral) avec intensité 0–1 (`llm-streaming-control/payloads.ts:6-16,44-121`). Le marker parser sépare littéral/spécial avant TTS ; `response-categoriser` distingue parole et raisonnement (`packages/core-agent/src/runtime/response-categoriser.ts:10-35`).

### P5 — Avatar interchangeable et lip-sync
- **AIRI** : wLipSync AudioWorklet fournit poids de visèmes AEIOUS + volume ; `createWLipSyncVowelDriver` choisit gagnant/second, gère le silence (160 ms) et lisse (`packages/model-driver-lipsync/src/shared/wlipsync/vowel-driver.ts:1-126`) ; l'adaptateur Live2D remappe en AEIOU avec lissage du `mouthOpen` (`model-driver-lipsync/src/live2d/index.ts:71-157`). Drivers séparés : `model-driver-magic-live2d` (pose + filtre), `motion-driver-magic` (idle par AR-HMM, `packages/motion-driver-magic/src/ar-hmm.ts`), `model-driver-mediapipe`. VRM dans les apps de scène.
- **OLV** : Live2D uniquement. Manifeste `model_dict.json` : `url`, `kScale`, `idleMotionGroupName`, `emotionMap`, `tapMotions` (`model_dict.json:1-25`). Pas de visèmes en Python.

### P6 — Perception d'écran et proactivité
- **AIRI** : capture via `desktopCapturer`/`setDisplayMediaRequestHandler`, permissions macOS, loopback audio (`packages/electron-screen-capture/src/main/index.ts:98-288`). Agent `spark:notify` qui inspecte un événement et décide s'il répond, en émettant `spark:command` avec `priority`/`interrupt`/`guidance` (`packages/core-agent/src/agents/spark-notify/agent.ts:28-59,86-90`).
- **OLV** : images typées `CAMERA/SCREEN/CLIPBOARD/UPLOAD` vers le LLM multimodal (`src/open_llm_vtuber/agent/input_types.py:6-13,76-94`, `basic_memory_agent.py:237-248`). Proactivité = déclencheur externe `ai-speak-signal` (`conversation_handler.py:35-64`).

### P7 — Outils / MCP
- **OLV** : client MCP stdio (`src/open_llm_vtuber/mcpp/mcp_client.py:8-10,83-97`), `ToolManager`/`ToolExecutor`, tool-calling natif OpenAI/Claude + repli JSON (`agent/agents/basic_memory_agent.py:293-556,597-662`). Aucune confirmation/garde-fou dans `mcpp/`.
- **AIRI** : SDK MCP stdio dans le main Electron, timeouts et normalisation de noms (`apps/stage-tamagotchi/src/main/services/airi/mcp-servers/index.ts:115-330`), UI de config, événements `tool-call`/`tool-result` (`chat-orchestrator-runtime.ts:739-873`). Pas de confirmation par appel trouvée.

### P8 — Mémoire long terme
- **AIRI** : compaction de contexte avec limite de tours récents et hook de résumé (`packages/core-agent/src/messages/compaction.ts:5-101`). `packages/memory-pgvector/src/index.ts` est un squelette vide : mémoire vectorielle non implémentée.
- **OLV** : `self._memory` en contexte (`basic_memory_agent.py:55,135-174`), persistée en JSON par personnage et rechargée (`basic_memory_agent.py:176-193`, `chat_history_manager.py`). Aucun résumé ni bornage de tokens (Letta est optionnel/externe).

## 3. Top 5 des mécanismes à récupérer (gain/effort)

1. **Troncature d'historique sur interruption (OLV)** — portable tel quel (Python). `handle_interrupt(heard_response)` + `[Interrupted by user]` : répond à P2 et évite de mémoriser une réponse jamais entendue. Effort faible, gain élevé.
2. **Chunker TTS « boost » (AIRI)** — à réécrire (TS→Python). Premier chunk court dès `minimumWords`, plafond `maximumWords`, classes de ponctuation. Répond à P3. Effort faible/moyen, gain élevé.
3. **Émotions structurées `<|ACT …|>` + normalisation (AIRI)** — à réécrire. Remplace nos tags texte par du JSON avec intensité et validation, retiré avant TTS. Répond à P4. Effort moyen.
4. **PlaybackManager : file priorisée + `stopByIntent`/`AbortController` (AIRI)** — à réécrire / inspiration. Interruption propre et ordonnancement par `intentId`/`ownerId`. Répond à P2/P3. Effort moyen.
5. **Hystérésis VAD + padding + durée minimale (AIRI) / hits-misses + pre-buffer (OLV)** — inspiration. Réglages concrets pour fiabiliser P1 sans refonte. Effort faible.

## 4. Ce qu'ils font moins bien ou pas du tout

- **Aucun des deux ne fait de vraie annulation d'écho ni de barge-in micro** : les deux sont half-duplex (AIRI coupe le micro pendant la parole, OLV s'appuie sur un client absent). Aucun n'est une référence AEC.
- **OLV** : pas de VRM, pas de visèmes, lip-sync délégué au frontend ; fin de tour figée (~0,8 s) ; pas de résumé mémoire ; pas de garde-fou sur les tool-calls.
- **AIRI** : pas de confirmation par appel d'outil ; `memory-pgvector` non implémenté ; VAD cantonné aux apps navigateur ; architecture TS/Vue/Electron lourde, peu portable vers Python.
- **OLV** : émotions par correspondance de chaînes, moins fiables que le JSON d'AIRI ; proactivité purement externe.

## 5. Non vérifié

- `Open-LLM-VTuber/frontend/` est **vide** dans le clone : VAD client, mute micro, écho, lip-sync, interruption, idle/clignement non inspectables. Idem `web_tool/`.
- Le paquet `@proj-airi/stage-ui` (store « hearing », `autoSendEnabled`, `HearingConfigDialog`) est **absent** : logique d'envoi de tour non vérifiée.
- Le runtime VRM d'AIRI non inspecté en détail.
- Analyse statique uniquement ; aucun test exécuté, aucune commande git.
