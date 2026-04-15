from src.assistant.pipeline_runtime import (
    build_pipeline_conversation_config,
    resolve_initial_tts_language,
)


def test_resolve_initial_tts_language_prefers_reply_language():
    config = {
        "pipeline": {"reply_language": "en"},
    }

    assert resolve_initial_tts_language(config, "fr") == "en"


def test_build_pipeline_conversation_config_uses_pipeline_defaults():
    config = {
        "character": {
            "name": "March 7th",
            "system_prompt": "You are March 7th.",
        },
        "tts": {
            "stream_tts": True,
            "auto_detect_language": False,
        },
        "asr": {
            "language": "auto",
        },
        "pipeline": {
            "reply_language": "en",
        },
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.character_name == "March 7th"
    assert conversation_config.system_prompt == "You are March 7th."
    assert conversation_config.stream_tts is True
    assert conversation_config.auto_detect_language is False
    assert conversation_config.asr_language == "auto"
    assert conversation_config.reply_language == "en"
