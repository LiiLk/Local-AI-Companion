import subprocess
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("resources/voices/f5_refs")
# Using a very standard voice that usually works
VOICE = "fr-FR-DeniseNeural" 

# Texts to synthesize
EMOTIONS = {
    "neutral": "Bonjour, je suis Aria. Je suis ton assistante virtuelle.",
    "happy": "C'est génial ! Je suis tellement contente de discuter avec toi aujourd'hui ! Quelle belle journée !",
    "sad": "Oh non... c'est vraiment dommage... je suis triste d'apprendre ça... j'espère que ça va aller.",
    "angry": "Mais c'est inacceptable ! Je ne suis pas du tout d'accord avec ça ! C'est n'importe quoi !"
}

def generate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Generating High-Quality References using Edge TTS CLI ({VOICE})...")
    
    for emotion, text in EMOTIONS.items():
        output_path = OUTPUT_DIR / f"aria_{emotion}.wav"
        print(f"   🎤 Generating {emotion} -> {output_path}")
        
        # Use CLI directly as Python API seems flaky
        # Point to venv executable explicitly
        cmd = [
            "venv/bin/edge-tts",
            "--voice", VOICE,
            "--text", text,
            "--write-media", str(output_path),
            "--rate=+0%"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate {emotion}: {e}")

    print("✅ All references generated successfully!")

if __name__ == "__main__":
    generate()
