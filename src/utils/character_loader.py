"""
Character preset loader.

Loads character configurations from config/characters/*.yaml files.
Allows easy switching between different AI personalities (March 7th, Juri, Clippy, etc.)
"""

import yaml
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root (go up from src/utils/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_available_characters() -> list[str]:
    """
    List all available character presets.
    
    Returns:
        List of character preset names (without .yaml extension)
    """
    characters_dir = PROJECT_ROOT / "config" / "characters"
    if not characters_dir.exists():
        return []
    
    return [f.stem for f in characters_dir.glob("*.yaml")]


def load_character_preset(preset_name: str) -> Optional[dict]:
    """
    Load a character preset from config/characters/<preset_name>.yaml
    
    Args:
        preset_name: Name of the preset (e.g., "march7th", "juri", "clippy")
        
    Returns:
        Character configuration dict, or None if not found
    """
    preset_path = PROJECT_ROOT / "config" / "characters" / f"{preset_name}.yaml"
    
    if not preset_path.exists():
        logger.warning(f"Character preset not found: {preset_path}")
        return None
    
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            preset = yaml.safe_load(f)
            logger.info(f"🎭 Loaded character preset: {preset.get('name', preset_name)}")
            return preset
    except Exception as e:
        logger.error(f"Failed to load character preset {preset_name}: {e}")
        return None


def resolve_character_config(config: dict) -> dict:
    """
    Resolve character configuration, loading preset if specified.
    
    This function checks if a preset is specified in config["character"]["preset"].
    If so, it loads the preset and merges it with any overrides in the config.
    
    Args:
        config: Main configuration dict (from config.yaml)
        
    Returns:
        Updated config dict with resolved character settings
    """
    character_config = config.get("character", {})
    preset_name = character_config.get("preset")
    
    if not preset_name:
        # No preset, use inline config
        return config
    
    # Load preset
    preset = load_character_preset(preset_name)
    if not preset:
        logger.warning(f"Preset '{preset_name}' not found, using default config")
        return config
    
    # Merge preset into character config
    # Preset values are used, but inline config can override
    resolved_character = {
        "name": preset.get("name", "AI"),
        "subtitle": preset.get("subtitle", ""),
        "system_prompt": preset.get("system_prompt", "You are a helpful AI assistant."),
        "preset": preset_name,  # Keep track of which preset is used
    }
    
    # Allow inline overrides
    for key in ["name", "system_prompt", "subtitle"]:
        if key in character_config and character_config[key] and key != "preset":
            resolved_character[key] = character_config[key]
    
    config["character"] = resolved_character
    
    # Also update TTS voice if preset has voice config
    voice_config = preset.get("voice", {})
    if voice_config:
        tts_config = config.get("tts", {})
        
        # Update XTTS speaker_wav if specified in preset
        speaker_wav = voice_config.get("speaker_wav")
        if speaker_wav and "xtts" in tts_config:
            tts_config.setdefault("xtts", {})["speaker_wav"] = speaker_wav
            logger.info(f"🎤 Using voice: {speaker_wav}")
        
        # Update Edge TTS voice if specified
        edge_voice = voice_config.get("edge_voice")
        if edge_voice:
            tts_config["voice"] = edge_voice
        
        # Update Kokoro voice if specified
        kokoro_voice = voice_config.get("kokoro_voice")
        if kokoro_voice:
            tts_config["kokoro_voice"] = kokoro_voice
    
    # Update Live2D config if preset has it
    live2d_config = preset.get("live2d", {})
    if live2d_config:
        config_live2d = config.setdefault("live2d", {})
        
        # Update model path
        if "model_path" in live2d_config:
            config_live2d.setdefault("model", {})["path"] = live2d_config["model_path"]
        if "settings_file" in live2d_config:
            config_live2d.setdefault("model", {})["settings_file"] = live2d_config["settings_file"]
        
        # Update expressions
        if "expressions" in live2d_config:
            config_live2d["expressions"] = live2d_config["expressions"]
        if "default_expression" in live2d_config:
            config_live2d["default_expression"] = live2d_config["default_expression"]
    
    return config


def get_character_info(config: dict) -> dict:
    """
    Get a summary of the current character configuration.
    
    Args:
        config: Main configuration dict
        
    Returns:
        Dict with character info for display
    """
    character = config.get("character", {})
    return {
        "name": character.get("name", "AI"),
        "subtitle": character.get("subtitle", ""),
        "preset": character.get("preset"),
        "available_presets": get_available_characters(),
    }
