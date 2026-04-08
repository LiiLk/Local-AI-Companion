"""
End-to-end smoke test for the current pipeline stack.

Flow:
1. Real audio input -> faster-whisper ASR
2. Transcription -> Gemma text/vision LLM
3. Gemma response -> Qwen3-TTS voice clone
4. Save generated response chunks as WAV files

Run:
    python -m scripts.smoke_test_gemma_whisper_qwen
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import yaml

from src.assistant.conversation_pipeline import ConversationConfig, ConversationPipeline
from src.asr.whisper_provider import WhisperProvider
from src.llm.gemma_text_vision_llm import GemmaTextVisionLLM
from src.omni.gemma_provider import GemmaProvider
from src.tts.qwen3_tts_provider import Qwen3TTSProvider
from src.utils.character_loader import resolve_character_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "resources" / "voices" / "march7th" / "VO_Archive_March_7th_5.wav"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "pipeline_smoke"


def load_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return resolve_character_config(config)


def prepare_audio_bytes(path: Path) -> bytes:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = soxr.resample(audio, sample_rate, 16000)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


async def main() -> None:
    print("=" * 60)
    print("PIPELINE SMOKE TEST: faster-whisper + Gemma + Qwen3-TTS")
    print("=" * 60)

    config = load_config()
    character = config["character"]
    gemma_config = config["gemma"]
    asr_config = config["asr"]
    tts_config = config["tts"]
    qwen_config = tts_config["qwen3"]
    voice_config = character.get("voice", {})

    input_path = DEFAULT_INPUT
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.wav"):
        old_file.unlink()

    audio_bytes = prepare_audio_bytes(input_path)

    print(f"\n[1/4] Input audio: {input_path.name}")
    print(f"  Raw bytes prepared: {len(audio_bytes)}")

    print("\n[2/4] Building providers...")
    started = time.time()
    gemma = GemmaProvider(
        model_id=gemma_config.get("model_id", "google/gemma-4-E2B-it"),
        device=gemma_config.get("device", "cuda"),
        quantization=gemma_config.get("quantization", "int4"),
        max_new_tokens=48,
        temperature=0.4,
        top_p=0.9,
        context_max_turns=6,
    )
    llm = GemmaTextVisionLLM(gemma=gemma, screen_config={"enabled": False})
    asr = WhisperProvider(
        model_size=asr_config.get("model_size", "large-v3-turbo"),
        device=asr_config.get("device", "cpu"),
        initial_prompt=asr_config.get("prompt"),
    )
    tts = Qwen3TTSProvider(
        model_id=qwen_config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        mode=qwen_config.get("mode", "voice_clone"),
        language=qwen_config.get("language", "auto"),
        speaker=qwen_config.get("speaker"),
        instruct=qwen_config.get("instruct"),
        ref_audio_path=voice_config.get("qwen_ref_audio") or qwen_config.get("ref_audio_path"),
        ref_text=voice_config.get("qwen_ref_text") or qwen_config.get("ref_text"),
        x_vector_only_mode=qwen_config.get("x_vector_only_mode"),
        device=qwen_config.get("device", "cuda:0"),
        dtype=qwen_config.get("dtype", "bfloat16"),
        attn_implementation=qwen_config.get("attn_implementation", "flash_attention_2"),
        backend=qwen_config.get("backend", "worker"),
        python_path=qwen_config.get("python_path"),
        site_packages_dir=qwen_config.get("site_packages_dir"),
        worker_script=qwen_config.get("worker_script"),
    )
    print(f"  Providers ready in {time.time() - started:.1f}s")

    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=asr,
        config=ConversationConfig(
            character_name=character.get("name", "March 7th"),
            system_prompt="You are March 7th. Reply in one or two short French sentences only.",
            stream_tts=True,
            auto_detect_language=tts_config.get("auto_detect_language", True),
            asr_language=asr_config.get("language", "auto"),
        ),
        rvc=None,
    )

    state = {
        "transcription": None,
        "response_chunks": [],
        "audio_payloads": [],
    }

    async def on_transcription(text: str) -> None:
        state["transcription"] = text

    async def on_response_chunk(chunk: str) -> None:
        state["response_chunks"].append(chunk)

    async def on_audio_ready(payload) -> None:
        index = len(state["audio_payloads"])
        output_path = output_dir / f"response_{index}.wav"
        output_path.write_bytes(payload.wav_bytes or b"")
        state["audio_payloads"].append(
            {
                "path": str(output_path),
                "duration_ms": payload.duration_ms,
                "sample_rate": payload.sample_rate,
                "text": payload.text,
                "size": output_path.stat().st_size,
            }
        )

    pipeline.on_transcription = on_transcription
    pipeline.on_response_chunk = on_response_chunk
    pipeline.on_audio_ready = on_audio_ready

    print("\n[3/4] Running end-to-end pipeline...")
    started = time.time()
    result = await pipeline.process_speech(audio_bytes)
    total_time = time.time() - started

    print("\n[4/4] Summary")
    print(f"  Transcription: {state['transcription']}")
    print(f"  Response:      {result}")
    print(f"  Audio chunks:  {len(state['audio_payloads'])}")
    for item in state["audio_payloads"]:
        print(
            "   - "
            f"{Path(item['path']).name}: {item['duration_ms']}ms, "
            f"{item['sample_rate']}Hz, {item['size']} bytes"
        )
    print(f"  Total time:    {total_time:.1f}s")

    if not result:
        raise SystemExit("Smoke test failed: no text response generated.")
    if not state["audio_payloads"]:
        raise SystemExit("Smoke test failed: no audio response generated.")

    tts.cleanup()
    llm.cleanup()

    print("\n" + "=" * 60)
    print("PIPELINE SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
