#!/usr/bin/env python3
"""
Download March 7th voice samples from Honkai Star Rail dataset.

This script downloads real voice samples from the game for better voice cloning.
The dataset is from: https://huggingface.co/datasets/simon3000/starrail-voice

Usage:
    python scripts/download_march7th_voice.py
"""

import os
import sys
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("resources/voices/march7th/real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_voices():
    """Download March 7th voice samples."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    print("Loading StarRail voice dataset (this may take a while)...")
    print("Dataset: https://huggingface.co/datasets/simon3000/starrail-voice")

    # Load only English voices to save bandwidth
    dataset = load_dataset(
        "simon3000/starrail-voice",
        split="train",
        streaming=True  # Stream to avoid downloading everything
    )

    # Filter for March 7th English voices
    march7th_samples = []
    target_count = 10  # Get 10 good samples

    print("\nSearching for March 7th English voice samples...")

    for i, sample in enumerate(dataset):
        if i % 10000 == 0:
            print(f"  Scanned {i} samples...")

        # Filter criteria
        speaker = sample.get("speaker", "")
        language = sample.get("language", "")
        transcription = sample.get("transcription", "")

        # Check if it's March 7th in English with transcription
        if (
            speaker and "march" in speaker.lower() and "7" in speaker
            and language == "English"
            and transcription  # Has transcription
            and len(transcription) > 20  # Not too short
            and len(transcription) < 200  # Not too long
        ):
            march7th_samples.append(sample)
            print(f"  Found: {transcription[:60]}...")

            if len(march7th_samples) >= target_count:
                break

    if not march7th_samples:
        print("\nNo March 7th samples found. Trying alternative search...")
        # Try with ingame_filename
        dataset = load_dataset(
            "simon3000/starrail-voice",
            split="train",
            streaming=True
        )
        for i, sample in enumerate(dataset):
            filename = sample.get("ingame_filename", "")
            language = sample.get("language", "")
            if "mar7th" in filename.lower() and language == "English":
                march7th_samples.append(sample)
                print(f"  Found by filename: {filename}")
                if len(march7th_samples) >= target_count:
                    break

    if not march7th_samples:
        print("\n❌ No March 7th samples found in dataset.")
        print("Try downloading manually from:")
        print("  https://huggingface.co/datasets/simon3000/starrail-voice")
        return

    print(f"\n✅ Found {len(march7th_samples)} March 7th samples!")

    # Save samples
    print("\nSaving audio files...")
    for i, sample in enumerate(march7th_samples):
        audio = sample.get("audio")
        transcription = sample.get("transcription", f"sample_{i}")

        if audio:
            # Save audio
            output_path = OUTPUT_DIR / f"march7th_real_{i+1}.wav"

            # Audio is usually a dict with 'array' and 'sampling_rate'
            if isinstance(audio, dict):
                import soundfile as sf
                sf.write(
                    str(output_path),
                    audio["array"],
                    audio["sampling_rate"]
                )
            else:
                # Try to save directly
                with open(output_path, "wb") as f:
                    f.write(audio)

            # Save transcription
            trans_path = OUTPUT_DIR / f"march7th_real_{i+1}.txt"
            trans_path.write_text(transcription)

            print(f"  Saved: {output_path.name}")
            print(f"    Transcription: {transcription[:60]}...")

    print(f"\n✅ Done! Files saved to: {OUTPUT_DIR}")
    print("\nTo use for voice cloning, update config.yaml:")
    print(f'  ref_audio_path: "{OUTPUT_DIR}/march7th_real_1.wav"')
    print(f'  prompt_text: "<copy from march7th_real_1.txt>"')


if __name__ == "__main__":
    download_voices()
