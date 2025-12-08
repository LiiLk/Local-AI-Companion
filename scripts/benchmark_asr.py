
import time
import sys
import os
from pathlib import Path
import logging

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.asr.whisper_provider import WhisperProvider

# Configuration
MODELS_TO_TEST = ["tiny", "base", "small", "distil-large-v3", "large-v3-turbo"]
TEST_FILE = "resources/voices/f5_refs/juri_neutral.wav" # Assuming this exists, verify first
# If not, we will need to find another file or generate one.

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def benchmark_model(model_size: str, audio_path: str):
    logger.info(f"\n========================================")
    logger.info(f"🚀 Testing Model: {model_size}")
    logger.info(f"========================================")
    
    try:
        # Measure Load Time
        start_load = time.time()
        asr = WhisperProvider(model_size=model_size, device="cpu", compute_type="int8")
        # Force load
        asr._get_model() 
        load_time = time.time() - start_load
        logger.info(f"⏱️  Load Time:       {load_time:.2f}s")
        
        # Measure Transcribe Time (Run 1 - Cold)
        start_transcribe = time.time()
        result = asr.transcribe(audio_path)
        transcribe_time = time.time() - start_transcribe
        logger.info(f"⏱️  Transcribe (Cold): {transcribe_time:.2f}s")
        logger.info(f"📝 Text: {result.text}")
        
        # Measure Transcribe Time (Run 2 - Warm)
        start_transcribe_warm = time.time()
        result_warm = asr.transcribe(audio_path)
        transcribe_time_warm = time.time() - start_transcribe_warm
        logger.info(f"⏱️  Transcribe (Warm): {transcribe_time_warm:.2f}s")
        
        return {
            "model": model_size,
            "load_time": load_time,
            "cold_time": transcribe_time,
            "warm_time": transcribe_time_warm,
            "text": result.text
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to test {model_size}: {e}")
        return None

def main():
    # Find a valid audio file
    audio_file = None
    possible_files = [
        "resources/voices/f5_refs/juri_neutral.wav",
        "resources/voices/samples/sample_fr.wav", 
        "test_audio.wav"
    ]
    
    project_root = Path(__file__).parent.parent
    
    for f in possible_files:
        p = project_root / f
        if p.exists():
            audio_file = str(p)
            break
            
    if not audio_file:
        logger.error("❌ No audio file found for testing.")
        return

    logger.info(f"📂 Using audio file: {audio_file}")
    
    results = []
    for model in MODELS_TO_TEST:
        res = benchmark_model(model, audio_file)
        if res:
            results.append(res)
            
    # Print Summary Table
    logger.info("\n\n📊 BENCHMARK RESULTS (CPU)")
    logger.info(f"{'Model':<20} | {'Load (s)':<10} | {'Cold (s)':<10} | {'Warm (s)':<10}")
    logger.info("-" * 60)
    for r in results:
        logger.info(f"{r['model']:<20} | {r['load_time']:<10.2f} | {r['cold_time']:<10.2f} | {r['warm_time']:<10.2f}")

if __name__ == "__main__":
    main()
