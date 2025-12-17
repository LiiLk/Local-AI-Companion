#!/usr/bin/env python3
"""
CosyVoice3 TTS Server - FastAPI server for CosyVoice3 voice synthesis.

This server runs in the cosyvoice conda environment (Python 3.10)
and exposes a simple HTTP API for text-to-speech synthesis.

Usage:
    # Start server (run from project root or with correct paths)
    python scripts/cosyvoice3_server.py --port 9881

    # Or via the start script:
    ./scripts/start_cosyvoice3.sh start

API Endpoints:
    POST /synthesize     - Synthesize text to audio
    GET  /health         - Health check
    GET  /               - Server info
"""

import argparse
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cosyvoice3_server")

# Add CosyVoice to path
COSYVOICE_DIR = Path(os.environ.get("COSYVOICE_DIR", "~/tools/CosyVoice")).expanduser()
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import torchaudio


# Global model instance
model = None
model_sample_rate = 24000


class SynthesizeRequest(BaseModel):
    """Request body for synthesis."""
    text: str
    prompt_text: str = ""  # Optional prompt text for zero-shot
    language: str = "fr"
    speed: float = 1.0
    stream: bool = False


class ServerInfo(BaseModel):
    """Server information."""
    name: str = "CosyVoice3 TTS Server"
    version: str = "1.0.0"
    model: str = "Fun-CosyVoice3-0.5B"
    sample_rate: int = 24000
    supported_languages: list[str] = ["en", "zh", "ja", "ko", "de", "es", "fr", "it", "ru"]


app = FastAPI(
    title="CosyVoice3 TTS Server",
    description="REST API for CosyVoice3 text-to-speech synthesis",
    version="1.0.0"
)


def load_model(model_dir: str, fp16: bool = True):
    """Load CosyVoice3 model with RL weights for better quality.

    Args:
        model_dir: Path to model directory
        fp16: Use half-precision (reduces VRAM from ~4.8GB to ~2.5GB)

    Note:
        CosyVoice's fp16 flag only enables autocast during inference, not weight conversion.
        We manually convert weights to FP16 after loading for true memory savings.
    """
    global model, model_sample_rate

    if model is not None:
        return model

    logger.info(f"Loading CosyVoice3 model from {model_dir}...")
    logger.info("Using RL-tuned weights (llm.rl.pt) for better quality (WER 0.81 vs 1.21)")

    from cosyvoice.cli.cosyvoice import AutoModel

    # Load model (will use llm.rl.pt via symlink llm.pt → llm.rl.pt)
    model = AutoModel(model_dir=model_dir, fp16=fp16)
    model_sample_rate = model.sample_rate

    # Convert weights to FP16 for actual VRAM reduction
    # CosyVoice's fp16 flag only enables autocast, weights stay FP32!
    if fp16 and torch.cuda.is_available():
        logger.info("Converting model weights to FP16 for VRAM reduction...")
        vram_before = torch.cuda.memory_allocated() / 1024**3

        # Convert LLM to FP16 (biggest component ~1.9GB -> ~950MB)
        model.model.llm.half()

        # Convert Flow Matching to FP16 (~1.3GB -> ~650MB)
        model.model.flow.half()

        # NOTE: Do NOT convert HiFi-GAN (hift) to FP16!
        # The f0_predictor inside hift uses .cpu() which converts to FP32,
        # causing type mismatch if weights are in FP16.
        # hift is only ~80MB anyway, not worth the compatibility issues.

        # Force garbage collection
        torch.cuda.empty_cache()

        vram_after = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"FP16 conversion complete! VRAM: {vram_before:.2f}GB -> {vram_after:.2f}GB")
    elif fp16:
        logger.warning("FP16 requested but no CUDA available, using FP32")

    logger.info(f"Model loaded! Sample rate: {model_sample_rate}, fp16={fp16}")
    return model


@app.get("/", response_model=ServerInfo)
async def root():
    """Get server information."""
    return ServerInfo(sample_rate=model_sample_rate)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    ref_audio: UploadFile = File(None),
    prompt_text: str = Form(""),
    language: str = Form("fr"),
    speed: float = Form(1.0)
):
    """
    Synthesize text to speech.

    Args:
        text: Text to synthesize
        ref_audio: Reference audio file for voice cloning (optional)
        prompt_text: Prompt text for zero-shot synthesis (optional)
        language: Target language (fr, en, zh, ja, etc.)
        speed: Speech speed factor (default 1.0)

    Returns:
        WAV audio file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        logger.info(f"Synthesizing: '{text[:50]}...' (lang={language}, speed={speed})")

        # Handle reference audio
        ref_audio_path = None
        temp_ref = None

        if ref_audio:
            # Save uploaded file temporarily
            temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            content = await ref_audio.read()
            temp_ref.write(content)
            temp_ref.close()
            ref_audio_path = temp_ref.name
            logger.info(f"Using uploaded reference audio: {ref_audio.filename}")

        # Choose inference method based on reference audio
        audio_chunks = []

        if ref_audio_path:
            # Cross-lingual or zero-shot synthesis with reference
            for output in model.inference_cross_lingual(
                text,
                ref_audio_path,
                stream=False,
                speed=speed
            ):
                audio_chunks.append(output['tts_speech'])
        else:
            # No reference - use inference_cross_lingual with default asset
            default_ref = COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav"
            if default_ref.exists():
                for output in model.inference_cross_lingual(
                    text,
                    str(default_ref),
                    stream=False,
                    speed=speed
                ):
                    audio_chunks.append(output['tts_speech'])
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No reference audio provided and default not found"
                )

        # Cleanup temp file
        if temp_ref:
            os.unlink(temp_ref.name)

        # Concatenate audio chunks
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        audio = torch.cat(audio_chunks, dim=1)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, model_sample_rate, format="wav")
        buffer.seek(0)

        logger.info(f"Generated audio: {audio.shape[1] / model_sample_rate:.2f}s")

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-Audio-Duration": str(audio.shape[1] / model_sample_rate),
                "X-Sample-Rate": str(model_sample_rate)
            }
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_with_ref")
async def synthesize_with_ref(
    text: str = Form(...),
    ref_audio_path: str = Form(...),
    prompt_text: str = Form(""),
    language: str = Form("fr"),
    speed: float = Form(1.0),
    mode: str = Form("auto")  # "zero_shot", "cross_lingual", "instruct", or "auto"
):
    """
    Synthesize text using a reference audio path on the server.

    Best Practices (from official CosyVoice3 documentation):
    - Reference audio: 3-10 seconds for best quality
    - Zero-shot: Use when you have exact transcription of reference
    - Cross-lingual: Use for different language than reference
    - Instruct: Use for dialect/emotion/speed control

    Args:
        text: Text to synthesize
        ref_audio_path: Path to reference audio on server (3-10s recommended)
        prompt_text: Exact transcription of reference audio (improves voice cloning)
        language: Target language (fr, en, zh, ja, etc.)
        speed: Speech speed factor (default 1.0)
        mode: Inference mode ("zero_shot", "cross_lingual", "instruct", "auto")

    Returns:
        WAV audio file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    ref_path = Path(ref_audio_path).expanduser()
    if not ref_path.exists():
        raise HTTPException(status_code=400, detail=f"Reference audio not found: {ref_audio_path}")

    try:
        # Determine inference mode
        # - zero_shot: Best when prompt_text matches reference audio exactly
        # - cross_lingual: Best for different language synthesis
        # - instruct: Best for explicit language/dialect/emotion control
        if mode == "auto":
            # Auto mode: use zero_shot if prompt_text provided, else cross_lingual
            mode = "zero_shot" if prompt_text.strip() else "cross_lingual"
        
        logger.info(f"Synthesizing: '{text[:50]}...' (mode={mode}, lang={language}, ref={ref_path.name})")

        audio_chunks = []

        if mode == "zero_shot" and prompt_text.strip():
            # Zero-shot: Use exact transcription for best voice cloning
            # Format: "You are a helpful assistant.<|endofprompt|>[transcription]"
            full_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text.strip()}"
            logger.info(f"Zero-shot with prompt: '{prompt_text[:50]}...'")
            for output in model.inference_zero_shot(
                text,
                full_prompt,
                str(ref_path),
                stream=False,
                speed=speed
            ):
                audio_chunks.append(output['tts_speech'])
        
        elif mode == "instruct":
            # Instruct: Explicit language/dialect control
            # Format: "You are a helpful assistant. <instruction>.<|endofprompt|>"
            lang_names = {
                "fr": "en français",
                "en": "in English", 
                "zh": "用中文",
                "ja": "日本語で",
                "ko": "한국어로",
                "de": "auf Deutsch",
                "es": "en español",
                "it": "in italiano",
                "ru": "по-русски"
            }
            lang_instruction = lang_names.get(language, f"in {language}")
            instruct_text = f"You are a helpful assistant. Please speak {lang_instruction}.<|endofprompt|>"
            logger.info(f"Instruct mode with: '{instruct_text}'")
            for output in model.inference_instruct2(
                text,
                instruct_text,
                str(ref_path),
                stream=False,
                speed=speed
            ):
                audio_chunks.append(output['tts_speech'])
        
        else:
            # Cross-lingual: No prompt text needed, good for different languages
            logger.info("Cross-lingual mode (no prompt text)")
            for output in model.inference_cross_lingual(
                text,
                str(ref_path),
                stream=False,
                speed=speed
            ):
                audio_chunks.append(output['tts_speech'])

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        audio = torch.cat(audio_chunks, dim=1)

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, model_sample_rate, format="wav")
        buffer.seek(0)

        duration = audio.shape[1] / model_sample_rate
        logger.info(f"Generated audio: {duration:.2f}s")

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-Audio-Duration": str(duration),
                "X-Sample-Rate": str(model_sample_rate)
            }
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CosyVoice3 TTS Server")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(COSYVOICE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
        help="Path to CosyVoice3 model directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9881,
        help="Port to listen on"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 (half-precision) to reduce VRAM from ~4.8GB to ~3GB"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16, use full precision (more VRAM but slightly better quality)"
    )

    args = parser.parse_args()
    
    # Determine fp16 setting
    use_fp16 = args.fp16 and not args.no_fp16

    # Load model at startup
    logger.info("Starting CosyVoice3 TTS Server...")
    load_model(args.model_dir, fp16=use_fp16)

    # Run server
    logger.info(f"Server running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
