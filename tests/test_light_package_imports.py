import json
import subprocess
import sys


def _run_import_probe(script: str) -> dict[str, bool]:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_assistant_package_import_does_not_load_audio_stack():
    loaded = _run_import_probe(
        """
import json
import sys
import src.assistant
print(json.dumps({
    "audio_service": "src.assistant.audio_service" in sys.modules,
    "conversation_pipeline": "src.assistant.conversation_pipeline" in sys.modules,
    "vad": "src.vad" in sys.modules,
    "torch": "torch" in sys.modules,
}))
"""
    )

    assert loaded == {
        "audio_service": False,
        "conversation_pipeline": False,
        "vad": False,
        "torch": False,
    }


def test_tts_package_import_does_not_load_optional_providers():
    loaded = _run_import_probe(
        """
import json
import sys
import src.tts
print(json.dumps({
    "base": "src.tts.base" in sys.modules,
    "edge_provider": "src.tts.edge_tts_provider" in sys.modules,
    "kokoro_provider": "src.tts.kokoro_provider" in sys.modules,
    "qwen3_provider": "src.tts.qwen3_tts_provider" in sys.modules,
    "edge_tts": "edge_tts" in sys.modules,
}))
"""
    )

    assert loaded == {
        "base": True,
        "edge_provider": False,
        "kokoro_provider": False,
        "qwen3_provider": False,
        "edge_tts": False,
    }


def test_lazy_package_exports_still_resolve():
    from src.assistant import ConversationConfig
    from src.tts import BaseTTS, KokoroProvider

    assert ConversationConfig.__name__ == "ConversationConfig"
    assert BaseTTS.__name__ == "BaseTTS"
    assert KokoroProvider.__name__ == "KokoroProvider"
