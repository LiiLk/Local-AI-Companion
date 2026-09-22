# Veille ASR — compagnons vocaux (22 sept. 2026)

Sources web = pages datées et citées ; toute affirmation non vérifiée est marquée **[non vérifié]**. Le code des dépôts clonés a été lu en lecture seule.

## 1. Ce que font réellement les compagnons

| Projet | ASR par défaut | Cloud/local | Tour/streaming | Techniques autour du modèle |
|---|---|---|---|---|
| Open-LLM-VTuber | faster-whisper, `model_path="distil-medium.en"` par défaut | local | par tour | `initial_prompt`, `beam_size=5`, `condition_on_previous_text=False` |
| AIRI | providers cloud (OpenAI-audio, Comet, MiMo) + Web Speech API / Apple Speech | cloud surtout | streaming + fusion de fragments | Silero VAD, `transcript-buffer` |
| Pipecat | Deepgram (exemple) | cloud | streaming | Silero VAD, agrégateur contexte |
| LiveKit Agents | pipeline STT-LLM-TTS (Deepgram/AssemblyAI) | cloud | streaming | VAD + turn detector |
| ElevenLabs Agents | Scribe v2 Realtime (~150 ms) | cloud | streaming | keyterm prompting, 90+ langues |
| Kyutai Unmute | Kyutai STT `1b-en_fr` | local/open | streaming 0,5 s | VAD sémantique intégré |

**Open-LLM-VTuber (preuves):** `faster_whisper_asr.py:12,31-44` — rappelé par prompt seulement, **aucun seuil anti-hallucination** (`no_speech_prob`, `avg_logprob`, `compression_ratio`) : c'est exactement la cause de « C'est parti. » sur 1 s. `sherpa_onnx_asr.py:27-28,36` : `hotwords_file` + `hotwords_score=1.5` existent mais `decoding_method="greedy_search"` par défaut — or les hotwords sherpa **exigent** `modified_beam_search` (doc sherpa). Donc, tel quel, le hotword est inerte. `fun_asr.py:42,44-45` : SenseVoiceSmall + `fsmn-vad` + ponctuation (utile zh/en, **pas de français**). `groq_whisper_asr.py:50-53` : prompt commenté, `temperature=0.0`. VAD : `config_manager/vad.py:10-16` (`prob_threshold=0.4`, 3 hits, 24 misses).

**AIRI (preuves):** `transcript-buffer.ts:1-13,94-106` fusionne les fragments ASR (jointure CJK/latin, `flushDelayMs`, max 80 car.). VAD navigateur : `apps/stage-web/src/workers/vad/vad.ts:50` charge `onnx-community/silero-vad` (transformers.js) avec **hystérésis** `speechThreshold=0.3` / `exitThreshold=0.1` (:27-28). Aucun Whisper local, aucun keyword boosting dans le code.

Les cadres « production » (LiveKit, Pipecat, ElevenLabs) gagnent en latence par **streaming STT cloud** ; les transcripts Realtime (OpenAI) sont retardés et sans partiels.

## 2. Modèles locaux open-weights pour le FR (sept. 2026)

| Modèle (date) | FR WER (benchmark) | Stream | Licence | Prompt/hotwords | Notes |
|---|---|---|---|---|---|
| Parakeet-TDT-0.6b-v3 (08/2025) | Fleurs 5,15 / MLS 4,97 / CoVoST 6,05 (carte HF) | simulé (chunks) | CC-BY-4.0 | non natif ; **NeMo GPU-PB** phrase boosting | non-autorégressif → peu d'hallucinations |
| Whisper large-v3-turbo | Fleurs 6,7 (vocova, 07/2026) | non (offline) | MIT | `initial_prompt`+`hotwords` | large-v3 : 5,8 |
| whisper-large-v3-french-distil-dec8 | Fleurs 5,40 (carte HF) | non | MIT | comme Whisper | FR spécialisé |
| distil-whisper | — | non | MIT | — | **anglais uniquement** |
| Kyutai stt-1b-en_fr | Fleurs non publié **[non vérifié]** | **oui, 0,5 s** | CC-BY-4.0 | non | VAD sémantique |
| Voxtral Mini 4B Realtime 2602 (02/2026) | Fleurs 6,42 @480 ms | oui | Apache-2.0 | context biasing 100 termes (EN-optimisé) | 4B |
| Qwen3-ASR 0.6B/1.7B (01/2026) | Fleurs moy. 7,57/4,90 | oui (TTFT 92 ms) | Apache-2.0 | non | 52 langues |
| Moonshine v2 (02/2026) | — | oui | permissive | non | **anglais exclusivement** |
| Canary-1B v2 | 25 langues EU dont FR | non | CC-BY-4.0 | GPU-PB | — |
| SenseVoiceSmall | **non supporté** | non | — | — | zh/en/yue/ja/ko |
| FireRedASR | **non supporté** | non | — | — | zh + anglais |

SenseVoice/FireRedASR sont **hors jeu pour le français**. Moonshine FR n'existe qu'en fine-tune communautaire (21,8 % WER, MLS).

## 3. Vos deux problèmes

**(a) Sigles/années (« l'AGI », « 2027 »).** Whisper `initial_prompt` **biaise sans forcer** (~224 tokens max) ; un prompt trop long fait **disparaître le début de phrase** (bug documenté whisper.cpp, 06/2026). Pour Parakeet, NeMo **GPU-PB** (CTC/RNN-T/TDT/AED, jusqu'à 20 000 phrases, surcoût 2–5 % RTFx, +8–10 F-score en greedy) est la voie propre, mais le modèle est **sensible à la casse** : écrire `AGI`, `2027`, `LLM`, `GPU`. Voxtral : context biasing. Filet réel = **post-correction** (regex + LLM principal) — le prompt ne garantit rien.

**(b) Hallucinations sur clips courts.** Défauts faster-whisper : `no_speech_threshold=0.6`, `logprob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `vad_filter=False`. Le piège structurel : l'hallucination boilerplate est **haute probabilité**, donc les filtres logprob la laissent passer. Correctifs : VAD **en amont** (Silero), `no_speech_threshold` abaissé à 0,4–0,5 (MetaWhisp), `condition_on_previous_text=False`, `hallucination_silence_threshold` (exige `word_timestamps`), filtre segments `whisper-guard` (`no_speech>0.8`, `avg_logprob<-1.5`, `compression>3.0`). Piste 2026 : *Hallucination Space Projection* (arXiv 2609.04561, 09/2026) réduit le taux d'hallucination de 31,3 % → 2,4 % sans réentraînement. **Les modèles CTC/TDT (Parakeet) n'ont pas ce problème** — argument décisif.

## 4. Streaming local

Kyutai STT (Delayed Streams Modeling) et Voxtral Realtime sont **nativement** streaming ; Qwen3-ASR, Moonshine v2 et sherpa-onnx aussi (zipformer, `nemotron-speech-streaming-en-0.6b` ; Parakeet v3 en « simulated streaming »). `whisper-streaming`/`simul-whisper` existent mais restent fragiles. Les compagnons réputés qui « comprennent pendant qu'on parle » s'appuient sur du **STT cloud streaming** ; côté local, AIRI ne fait que fusionner des fragments.

## 5. Verdict (FR + anglais technique, RTX 4070 12 Go, <100 ms)

**« Quasi sans erreur » en local n'est pas réaliste.** Le plancher public est ~5 % WER FR ; les projets perçus comme quasi parfaits utilisent Deepgram/AssemblyAI/Scribe. En local, on **approche**, on n'égale pas — et le résidu se règle par boosting + post-correction LLM.

| Rang | Combinaison | Gain attendu | Coût VRAM/latence | Risque | Effort |
|---|---|---|---|---|---|
| 1 | **Parakeet-TDT-0.6b-v3 int8** + NeMo GPU-PB (`AGI`, `2027`, `LLM`, `GPU`, `IA`) + VAD | FR ~5 % ; **zéro hallucination générative** ; corrige sigles/années | ~1–1,5 Go ; RTF très bas (déjà intégré) | casse des mots boostés ; streaming par chunks | moyen (activer `boosting_tree`) |
| 2 | **faster-whisper large-v3-turbo** + VAD Silero + seuils (`no_speech=0.45`, `logprob=-1.0`, `compression=2.4`, `condition_on_previous_text=False`) + `initial_prompt` + regex/LLM | moins d'hallucinations, sigles partiellement corrigés | ~2–3 Go fp16 ; 0,6–0,8 s | génératif : hallucinations réduites, pas éliminées | faible |
| 3 | **Kyutai stt-1b-en_fr** (ou Voxtral Realtime 4B) | latence perçue minimale, VAD sémantique | ~2–3 Go / ~8 Go (4B) | FR moins documenté (Kyutai) **[non vérifié]** ; Voxtral 4B lourd sur GPU partagée | élevé |

Les deux premiers sont **complémentaires** : Parakeet comme moteur principal, Whisper+seuils en secours/fallback pour les segments à faible confiance. C'est ce couple, plus un boosting de domaine et une post-correction LLM, qui donne le meilleur rapport erreurs/coût sur votre matériel.

### Non vérifié / à creuser
- WER Fleurs français de Kyutai `1b-en_fr` (aucune valeur publiée trouvée).
- Latence réelle de Parakeet v3 int8 sur RTX 4070 (non benchmarkée ici).
- « Neuro-sama » n'a **aucune** spec ASR publique.
---

## Vérifications du tech lead (2026-09-22)

- Parakeet-TDT-0.6b-v3 : WER FR 5,15 (Fleurs, `fr_fr`) confirmé sur la fiche Hugging Face.
- **Correction** : la recommandation n° 1 (Parakeet + NeMo GPU-PB) ne s'applique pas telle quelle. Notre intégration Parakeet passe par `onnx-asr` (ONNX int8), qui n'a aucun mécanisme de hotwords / context biasing ; GPU-PB exige le runtime NeMo PyTorch.
- Test comparatif sur voix synthétique (Kokoro FR), même machine : Parakeet 2× plus rapide (CPU, 120-350 ms vs 300-650 ms GPU), **zéro hallucination** sur bruit de 1 s et parole tronquée de 0,5 s (Whisper : « Obrigado. », « Is »), mais « l'AGI » → « l'AJ » et « VRAM » → « VAM » ; Whisper juste sur AGI mais « GPU » → « GP ». Aucun des deux n'est propre sur les sigles ; à trancher sur la voix réelle de l'utilisateur.
- Le prompt de vocabulaire Whisper (français) ne fait pas basculer les phrases anglaises en français (détection de langue avant décodage, vérifié).
