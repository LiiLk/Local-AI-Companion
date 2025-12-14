from pathlib import Path
from src.asr.whisper_provider import WhisperProvider

def transcribe():
    print("🚀 Initializing Local Whisper for Reference Transcription (CPU Mode)...")
    asr = WhisperProvider(device="cpu", compute_type="int8")
    # Force English for Juri's file if we know it's English
    # But asr.transcribe_file auto detects usually.
    # We'll stick to defaults or "en" if supported.
    asr.language = "en" 
    
    ref_file = Path("resources/voices/f5_refs/juri_neutral.wav")
    
    if not ref_file.exists():
        print(f"❌ File not found: {ref_file}")
        return

    print(f"🎤 Transcribing: {ref_file}")
    # run in thread/sync
    result = asr.transcribe(ref_file)
    print(f"\n📝 TRANSCRIPTION RESULT:\n{result.text}\n")

if __name__ == "__main__":
    transcribe()
