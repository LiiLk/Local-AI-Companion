# March 7th Voice Reference

To enable voice cloning for March 7th in omni mode (MiniCPM-o):

1. Record or find a 6-30 second clear audio clip of March 7th's voice
2. Save it as `reference.wav` in this folder (16kHz mono recommended)
3. Update `config/characters/march7th.yaml`:
   ```yaml
   voice:
     omni_ref_audio: "resources/voices/march7th/reference.wav"
   ```

## Requirements
- WAV format, 16kHz sample rate, mono channel
- 6-30 seconds of clear speech (no music/noise)
- Consistent voice throughout the clip

## Sources
- Game voice lines from Honkai: Star Rail
- Fan recordings or voice packs
