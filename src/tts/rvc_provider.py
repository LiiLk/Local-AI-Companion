"""
RVC (Retrieval-based Voice Conversion) post-processor.

Takes TTS audio output and converts it to a target voice using an RVC v2 model.
Pipeline: Kokoro TTS → RVC → final audio with cloned voice.

Requires: rvc-inferpy (pip install rvc-inferpy) + fairseq fork for Python 3.12
"""

import logging
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RVCConverter:
    """
    Voice converter using RVC v2 models.

    Loads a .pth model + optional .index file and converts audio
    from any voice to the target voice.

    Args:
        model_path: Path to the .pth RVC model file.
        index_path: Optional path to the .index FAISS file.
        device: "cuda:0" or "cpu".
        f0_method: Pitch extraction method ("rmvpe", "crepe", "harvest").
        index_rate: How much to use the index (0.0-1.0). Higher = more similar.
        protect: Protect voiceless consonants (0.0-0.5).
    """

    def __init__(
        self,
        model_path: str | Path,
        index_path: str | Path | None = None,
        device: str = "cuda:0",
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        protect: float = 0.33,
    ):
        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.device = device
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.protect = protect
        self._converter = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._available = None

    @staticmethod
    def is_available() -> bool:
        """Check if RVC dependencies are installed."""
        try:
            from rvc_inferpy import RVCConverter as _RVC
            return True
        except ImportError:
            return False

    def _load(self):
        """Lazy-load the RVC model."""
        if self._converter is not None:
            return

        try:
            from rvc_inferpy import RVCConverter as _RVC
        except ImportError:
            raise ImportError(
                "rvc-inferpy is not installed. Install with:\n"
                "  pip install git+https://github.com/One-sixth/fairseq.git\n"
                "  pip install rvc-inferpy"
            )

        logger.info(f"Loading RVC model: {self.model_path.name}")
        self._converter = _RVC(device=self.device, is_half=True)
        self._converter.load_model(str(self.model_path))
        logger.info("RVC model loaded")

    def convert_file(self, input_path: str | Path, output_path: str | Path) -> Path:
        """Convert an audio file to the target voice."""
        self._load()
        input_path = str(input_path)
        output_path = str(output_path)

        self._converter.infer_file(
            input_path,
            output_path,
            f0_method=self.f0_method,
            index_rate=self.index_rate,
            protect=self.protect,
        )
        return Path(output_path)

    def convert_array(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        Convert audio numpy array to target voice.

        Args:
            audio: Input audio (float32, mono).
            sr: Sample rate of input.

        Returns:
            (converted_audio, sample_rate) tuple.
        """
        self._load()

        # Write to temp file, convert, read back
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, audio, sr)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        try:
            self.convert_file(tmp_in_path, tmp_out_path)
            converted, out_sr = sf.read(tmp_out_path, dtype="float32")
            return converted, out_sr
        finally:
            import os
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

    async def convert_array_async(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Async wrapper for convert_array."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.convert_array,
            audio, sr,
        )
