from kokoro import KPipeline
import soundfile as sf
import numpy as np

# Init pipeline
try:
    pipeline = KPipeline(lang_code='f') # French
except Exception as e:
    print(f"Init failed: {e}")
    exit(1)

text = "Salut ! C'est incroyable cette journée, je suis trop contente !"

# 1. Pure French reference
print("Generating pure...")
try:
    gen = pipeline(text, voice='ff_siwis', speed=1.0)
    for _, _, audio in gen:
        sf.write('pure.wav', audio, 24000)
except Exception as e:
    print(f"Pure failed: {e}")

# 2. Mixed: French + American (Heart = energetic?)
print("Generating mixed (50/50)...")
try:
    gen = pipeline(text, voice='ff_siwis+af_bella', speed=1.0) # Bella is allegedly energetic
    for _, _, audio in gen:
        sf.write('mixed.wav', audio, 24000)
    print("Mixed generation success!")
except Exception as e:
    print(f"Mixing failed: {e}")
