import pytest

from src.utils.language_detection import normalize_language_code


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fr", "fr"),
        ("en", "en"),
        ("es", "es"),
        ("it", "it"),
        ("ja", "ja"),
        ("zh", "zh"),
        ("pt-BR", "pt"),
        ("tr", "tr"),
        ("de", "de"),
        ("ru", "ru"),
    ],
)
def test_normalize_language_code_handles_multilingual_locales(raw, expected):
    assert normalize_language_code(raw) == expected
