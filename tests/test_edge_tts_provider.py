import tempfile
from pathlib import Path

import pytest

from src.tts.edge_tts_provider import EdgeTTSProvider


@pytest.mark.asyncio
async def test_edge_tts_default_output_uses_secure_temp_file(monkeypatch):
    saved_paths = []

    class FakeCommunicate:
        def __init__(self, **_kwargs):
            pass

        async def save(self, output_path: str) -> None:
            saved_paths.append(Path(output_path))
            Path(output_path).write_bytes(b"mp3")

    monkeypatch.setattr("src.tts.edge_tts_provider.edge_tts.Communicate", FakeCommunicate)

    provider = EdgeTTSProvider()
    result = await provider.synthesize("Hello")

    try:
        assert saved_paths == [result.audio_path]
        assert result.audio_path.parent == Path(tempfile.gettempdir())
        assert result.audio_path.name.startswith("local_ai_companion_edge_tts_")
        assert result.audio_path.suffix == ".mp3"
        assert result.audio_path.read_bytes() == b"mp3"
    finally:
        result.audio_path.unlink(missing_ok=True)
