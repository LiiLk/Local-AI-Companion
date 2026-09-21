# Veille Avatar — Incarnation (septembre 2026)

Portée : projet public, avatars par défaut librement redistribuables, swap facile, RTX 4070 12 Go partagé. Chaque point clé = nom exact + date + URL. Les éléments non vérifiables sont signalés **[NON VÉRIFIÉ]**.

## 1. Live2D vs VRM vs autres

- **Live2D** = illustration 2D découpée et déformée par Cubism ; pas de 3D, pas de VRM, tourne surtout dans VTube Studio (VTuberMe, consulté 2026 ; vtubeme.com/vs/live2d).
- **VRM 1.0** = format 3D ouvert bâti sur glTF, métadonnées de licence embarquées, spring bones, expressions. `@pixiv/three-vrm` **v3.5.5 (npm, 2026-07-09)** supporte **WebGPURenderer** depuis la v3 (`MToonNodeMaterial`), MIT (npmjs.com/package/@pixiv/three-vrm ; github.com/pixiv/three-vrm).
- **Piège de version** : VRoid exporte du VRM 1.0, VSeeFace ne lit que du 0.x (vtubeme.com/guide/vrm-versions, 2026). « Supporte VRM » ne suffit pas.
- **Inochi2D** : BSD-2, sans frais, mais instable ; format INP2 ajouté le **2026-01-29** ; retours communautaires peu favorables (r/vtubertech, 2025-11-01).
- **Gaussian splats** : `Gaussian-VRM` (OSS ~2025-10, MIT, 60 fps navigateur, github.com/naruya/gaussian-vrm) ; HyperGaussians (CVPR 2026, 300 fps de rendu, ~2 j d'entraînement).
- **Projets comparables** : **Open-LLM-VTuber** (Live2D Cubism 5, ~13,9k ★, figé ~mai 2026) ; **AIRI / moeru-ai** (Live2D+VRM+MMD, v0.12.0-beta.5 le 2026-08-29, motion « MAGIC ») ; **Amica** (dernier commit produit **2025-07**) ; **SillyTavern** (extensions Live2D et VRM officielles) ; **Neuro-sama** fermée (3D le 2025-11-15).

| Critère | Live2D | VRM 1.0 | Gaussian/3DGS |
|---|---|---|---|
| Expressions/regard | excellent (2D) | bon (blendshapes + lookAt) | moyen (selon modèle) |
| Lip-sync | MotionSync (visèmes A/I/U/E/O) | blendshapes audio | via rig externe |
| Coût GPU | très faible | faible (three-vrm) | rendu OK, capture lourde |
| Swap | dossier modèle | un `.vrm` | pipeline scan |
| Modèles libres | rares (licences Live2D) | nombreux CC0 | très rares |
| Licence moteur | SDK Cubism contraignant | three-vrm MIT | variable |

## 2. Licences

- **Live2D Cubism SDK** : individus et petites entreprises (<10 M ¥ de CA) exonérés, **sauf « Expandable Applications »** — un **système d'avatars en est une**, contrat séparé requis quel que soit le statut ; tracking et >20 M ¥ aussi (live2d.com/en/sdk/license ; help.live2d.com/en/sdk/sdk_001). → **Risque réel pour un projet open-source d'avatars.**
- **Modèles d'exemple Live2D** ne sont **pas** sous licence du projet : Live2D Free Material License, usage commercial restreint (note du dépôt Open-LLM-VTuber).
- **VRM** : licence par modèle dans les métadonnées ; VRM 1.0 ajoute `allowRedistribution`, `modification`, `commercialUsage` (vtubeme.com/guide/vrm-versions). **VRoid Hub n'autorise pas CC0**, mais **VRoid Studio permet de créer et exporter un avatar original** en choisissant ses conditions (FAQ VRoid / Pixiv).
- **Avatars libres redistribuables** : **100Avatars** de PolygonalMind (300+ VRM+FBX, **CC0**, github.com/PolygonalMind/100Avatars) ; **OpenSourceAvatars** (~4260 avatars, registre CC0/CC-BY, opensourceavatars.com, github.com/ToxSam/open-source-avatars).

## 3. Animation pilotée par IA en temps réel

- **NVIDIA Audio2Face-3D**, open-sourcé le **2025-09-24** : modèles régression v2.3 (faible latence) et diffusion v3.0 (streaming), SDK **MIT**, framework d'entraînement **Apache-2.0**, modèles sous NVIDIA Open Model License ; sortie ARKit blendshapes, >60 fps (developer.nvidia.com/blog/nvidia-open-sources-audio2face-animation-model ; github.com/NVIDIA/Audio2Face-3D).
- **Lip-sync TTS léger** : analyse RMS/FFT du flux audio → paramètres de bouche (approche `wlipsync` d'AIRI) ; **Live2D MotionSync / CRI LipSync** (royalty-free mais adossé au SDK Cubism).
- **Mouvement idle** : AIRI « MAGIC » (2026) pour un idle crédible sans moteur de gestes lourd.
- **Speech-to-motion léger depuis le texte : aucun standard local ouvert confirmé en 2026. [NON VÉRIFIÉ]**

## 4. Humain réaliste

| Modèle | VRAM | Latence | Licence |
|---|---|---|---|
| MuseTalk 1.5 | ~8 Go (256px) | 30 fps+ (V100) | MIT |
| LatentSync 1.5 / 1.6 | ~8 Go / ~18 Go (512px) | lent | Apache-2.0 |
| Ditto (antgroup) | TensorRT, GPU | RTF<1, FFD <400 ms | recherche |
| MOVA-720p | ~48 Go (12 Go offload) | clips 8 s | Apache-2.0 |

Sources : github.com/TMElyralab/MuseTalk ; github.com/antgroup/ditto-talkinghead ; news.creeta.com (2026-07-09) ; research.nvidia.com/labs/amri/projects/instant4d.
**MetaHuman** : gratuit <1 M$ de CA (metahuman.com/license) ; RigLogic/DNA open-sourcés **MIT** le **2026-06-11** (MetaHuman 5.8) ; pilotage Audio2Face via plugin ACE Unreal. **Verdict 12 Go** : un talking-head diffusion (~8–18 Go) **ne cohabite pas** avec un LLM+TTS local ; à réserver au cloud/offline.

## 5. Génération IA d'avatars VRM

- **AniGen** (SIGGRAPH 2026, MIT) : image → asset 3D riggé (github.com/VAST-AI-Research/AniGen).
- **Hamr** (2026-05-08, headless, VRM 1.0) et **Seiðr-Smiðja** (mai 2026) : alternatives ouvertes à VRoid, génération VRM pilotée par agents.
- **AutoVtuber** : formulaire → `.vrm` via SDXL+TripoSR+base VRoid (~8 min).
- **Meshy** (commercial) : texte/image → modèle riggé. **VRoid Studio** reste la voie manuelle fiable et redistribuable.

## 6. VERDICT

**Avatar par défaut : VRM 1.0 rendu par `@pixiv/three-vrm` (v3.5.5, MIT).** Motifs : format ouvert, modèles **CC0** disponibles (100Avatars, OpenSourceAvatars), aucun frais SDK, swap = un fichier, création/redistribution via VRoid Studio. **Live2D en option** (backend secondaire) mais **ne pas embarquer le SDK Cubism** sans contrat : un système d'avatars = Expandable Application.

**Architecture « avatar pluggable »** — interface commune :
`load(path)`, `setEmotion(label)`, `setViseme(viseme, weight)`, `update(dt)`, `dispose()`.
Backends : `vrm` (three-vrm), `live2d` (plugin externe non fourni), `sprite` (fallback PNG/SVG). Mapping émotions = 28 catégories SillyTavern ; lip-sync = RMS/FFT sur l'audio TTS, puis Audio2Face-3D (ARKit) si besoin 3D.

**Humain réaliste** : seulement en mode opt-in, quand l'utilisateur accepte soit le cloud, soit de sacrifier le LLM local (VRAM >18 Go), ou pour des clips pré-calculés — jamais sur le chemin temps réel 12 Go.

**Non vérifié** : VRAM Ditto/MuseTalk mesurée sur 4070 ; état d'Inochi2D et d'Ayagami.