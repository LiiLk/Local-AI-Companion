import asyncio
import os
from pathlib import Path
from src.tts.kokoro_provider import KokoroProvider

async def generate_refs():
    print("Generating reference audios with Kokoro...")
    
    # Ensure directory exists
    ref_dir = Path("resources/voices/f5_refs")
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    tts = KokoroProvider(voice="ff_siwis")
    
    prompts = {
        "neutral": "Bonjour, je suis Aria. Je suis ton assistante virtuelle.",
        "happy": "C'est génial ! Je suis tellement contente de discuter avec toi aujourd'hui !",
        "sad": "Oh non... c'est vraiment dommage... je suis triste d'apprendre ça...",
        "angry": "Mais c'est inacceptable ! Je ne suis pas du tout d'accord avec ça !"
    }
    
    for emotion, text in prompts.items():
        output_path = ref_dir / f"aria_{emotion}.wav"
        print(f"Generating {emotion} -> {output_path}")
        await tts.synthesize(text, output_path)
        
    print("Done! References generated.")

if __name__ == "__main__":
    asyncio.run(generate_refs())
