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


def load_model(model_dir: str):
    """Load CosyVoice3 model with RL weights for better quality."""
    global model, model_sample_rate

    if model is not None:
        return model

    logger.info(f"Loading CosyVoice3 model from {model_dir}...")
    logger.info("Using RL-tuned weights (llm.rl.pt) for better quality (WER 0.81 vs 1.21)")

    from cosyvoice.cli.cosyvoice import AutoModel

    # Load model (will use llm.rl.pt via symlink llm.pt → llm.rl.pt)
    # RL model has better quality: WER 0.81 vs 1.21, Similarity 77.4%, Naturalness 69.5%
    model = AutoModel(model_dir=model_dir)
    model_sample_rate = model.sample_rate

    logger.info(f"Model loaded! Sample rate: {model_sample_rate}")
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
    speed: float = Form(1.0)
):
    """
    Synthesize text using a reference audio path on the server.

    This is faster than uploading when the reference is already on the server.

    Args:
        text: Text to synthesize
        ref_audio_path: Path to reference audio on server
        prompt_text: Prompt text (optional, for zero-shot)
        language: Target language
        speed: Speech speed factor

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
        logger.info(f"Synthesizing with ref: '{text[:50]}...' (ref={ref_path.name})")

        audio_chunks = []

        # Use zero-shot if prompt_text provided, otherwise cross-lingual
        if prompt_text.strip():
            # Zero-shot with prompt text
            full_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
            for output in model.inference_zero_shot(
                text,
                full_prompt,
                str(ref_path),
                stream=False,
                speed=speed
            ):
                audio_chunks.append(output['tts_speech'])
        else:
            # Cross-lingual (no prompt text needed)
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

    args = parser.parse_args()

    # Load model at startup
    logger.info("Starting CosyVoice3 TTS Server...")
    load_model(args.model_dir)

    # Run server
    logger.info(f"Server running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
