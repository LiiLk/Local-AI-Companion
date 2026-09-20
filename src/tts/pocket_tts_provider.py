from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import math
import os
import queue
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath
from typing import Any, AsyncGenerator

from .base import BaseTTS, TTSResult, Voice

POCKET_TTS_VERSION = "3.1.0"
VOICE_NAMES = (
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles",
    "cosette", "eponine", "eve", "fantine", "george", "jane", "jean",
    "javert", "marius", "mary", "michael", "paul", "peter_yearsley",
    "stuart_bell", "vera",
)


MAX_VOICE_STATES = 4


class PocketTTSProvider(BaseTTS):
    def __init__(
        self,
        voice: str = "alba",
        language: str = "en",
        ref_audio_path: str | Path | None = None,
        device: str = "cpu",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ):
        if device != "cpu":
            raise ValueError("Pocket TTS provider supports only device='cpu'.")
        self._validate_language(language)
        self.set_rate(rate)
        self.set_pitch(pitch)
        self._lock = threading.Lock()
        self._model: Any = None
        self._voice_states: dict[str | Path, Any] = {}
        self._voice = self._validate_voice(voice)
        self._ref_audio_path = (
            self._validate_audio_path(ref_audio_path)
            if ref_audio_path is not None else None
        )

    @staticmethod
    def _validate_language(language: str) -> None:
        value = language.strip().lower().replace("_", "-")
        if value not in ("en", "english", "en-us", "en-gb", "en-au", "en-ca"):
            raise ValueError(
                f"Unsupported Pocket TTS provider language: {language!r}. "
                "This adapter supports English only; use language='en'."
            )

    @staticmethod
    def _validate_audio_path(value: str | Path) -> Path:
        raw = str(value)
        windows_path = PureWindowsPath(raw)
        if (
            not raw.strip()
            or "://" in raw
            or raw.startswith(("//", "\\\\"))
            or (":" in raw and not (
                len(windows_path.drive) == 2 and windows_path.is_absolute()
                and ":" not in raw[2:]
            ))
        ):
            raise ValueError("Pocket TTS reference audio must be a local file, not a URL or UNC path.")
        path = Path(raw).expanduser()
        if path.suffix.lower() not in (".wav", ".flac", ".mp3", ".ogg"):
            raise ValueError("Pocket TTS reference audio must be a local WAV, FLAC, MP3 or OGG file.")
        return path

    @classmethod
    def _validate_voice(cls, voice: str) -> str | Path:
        if voice in VOICE_NAMES:
            return voice
        if not any(character in voice for character in (".", "/", "\\", ":")):
            raise ValueError(f"Unknown Pocket TTS voice {voice!r}; select a listed voice or local audio file.")
        return cls._validate_audio_path(voice)

    @property
    def voice(self) -> str:
        with self._lock:
            return str(self._ref_audio_path or self._voice)

    @property
    def language(self) -> str:
        return "en"

    @property
    def device(self) -> str:
        return "cpu"

    def _load_model_locked(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            installed_version = version("pocket-tts")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Pocket TTS is not installed. Use requirements-optional-pocket.txt in a "
                "compatible isolated environment; pocket-tts requires NumPy >=2, "
                "which conflicts with the main runtime's NumPy <2 requirement."
            ) from exc
        if installed_version != POCKET_TTS_VERSION:
            raise RuntimeError(
                f"Pocket TTS requires pocket-tts=={POCKET_TTS_VERSION}; found {installed_version}. "
                "Use requirements-optional-pocket.txt in a compatible isolated environment."
            )
        try:
            from pocket_tts import TTSModel
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Pocket TTS dependencies could not load. Check requirements-optional-pocket.txt "
                "in a compatible isolated environment (Python 3.10-3.14, PyTorch >=2.5, NumPy >=2)."
            ) from exc
        try:
            model = TTSModel.load_model(language="english")
            model.to("cpu")
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Pocket TTS model loading failed. Check the local Hugging Face cache, "
                "network access for first-time downloads, and model access permissions. "
                "For gated weights, accept the upstream terms and authenticate locally."
            ) from exc
        self._model = model
        return model

    def _load_voice_locked(self) -> tuple[Any, Any]:
        source = self._ref_audio_path or self._voice
        if isinstance(source, Path):
            source = self._validate_audio_path(source.resolve())
            if not source.is_file():
                raise FileNotFoundError(f"Pocket TTS reference audio not found: {source}")
        model = self._load_model_locked()
        if source not in self._voice_states:
            try:
                state = model.get_state_for_audio_prompt(source)
            except Exception as exc:
                raise RuntimeError(
                    "Pocket TTS voice loading failed. Check the selected audio file or "
                    "named-voice cache and download access. Local voice cloning requires "
                    "access to cloning weights; accept the upstream terms and authenticate "
                    "locally if gated. No alternative voice was selected."
                ) from exc
            self._voice_states[source] = state
            while len(self._voice_states) > MAX_VOICE_STATES:
                self._voice_states.pop(next(iter(self._voice_states)))
        return model, self._voice_states[source]

    def _synthesize_sync(self, text: str, output_path: Path | None = None) -> TTSResult:
        if not text.strip():
            raise ValueError("Pocket TTS cannot synthesize empty text.")
        with self._lock:
            import numpy as np

            started = time.perf_counter()
            model, state = self._load_voice_locked()
            try:
                audio = model.generate_audio(state, text, copy_state=True)
                samples = np.asarray(audio.detach().cpu().numpy(), dtype=np.float32)
            except Exception as exc:
                raise RuntimeError("Pocket TTS synthesis failed on CPU.") from exc
            if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
                raise RuntimeError("Pocket TTS returned invalid or empty mono audio.")
            sample_rate = int(model.sample_rate)
            if sample_rate <= 0:
                raise RuntimeError("Pocket TTS returned an invalid sample rate.")
            pcm = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm.tobytes())
            data = buffer.getvalue()
            path = Path(output_path) if output_path is not None else None
            if path is not None:
                path.write_bytes(data)
            return TTSResult(
                audio_path=path,
                audio_data=data if path is None else None,
                duration=samples.size / sample_rate,
                metadata={
                    "provider": "pocket",
                    "voice": str(self._ref_audio_path or self._voice),
                    "language": "en",
                    "device": "cpu",
                    "sample_rate": sample_rate,
                    "synth_ms": (time.perf_counter() - started) * 1000,
                },
            )

    async def synthesize(self, text: str, output_path: Path | None = None) -> TTSResult:
        return await asyncio.to_thread(self._synthesize_sync, text, output_path)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        result = await self.synthesize(text)
        if result.audio_data is not None:
            yield result.audio_data

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        if language is not None:
            self._validate_language(language)
        return [
            Voice(id=name, name=name.replace("_", " ").title(), language="en", gender="Unknown")
            for name in VOICE_NAMES
        ]

    def set_voice(self, voice_id: str) -> None:
        with self._lock:
            voice = self._validate_voice(voice_id)
            self._voice = voice
            self._ref_audio_path = None

    def set_language(self, language: str | None) -> None:
        if language is not None:
            self._validate_language(language)

    def set_rate(self, rate: str) -> None:
        if rate not in ("+0%", "0%", "-0%"):
            raise ValueError("Pocket TTS does not support rate adjustment; use '+0%'.")

    def set_pitch(self, pitch: str) -> None:
        if pitch not in ("+0Hz", "0Hz", "-0Hz"):
            raise ValueError("Pocket TTS does not support pitch adjustment; use '+0Hz'.")

    def preload(self) -> None:
        with self._lock:
            self._load_voice_locked()

    def warmup(self) -> None:
        self._synthesize_sync("Hello.")

    def cleanup(self) -> None:
        with self._lock:
            self._voice_states.clear()
            self._model = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "pocket_tts_worker.py"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024


@dataclass
class _PocketWorker:
    process: Any
    replies: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=4))
    requests: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))
    stopped: threading.Event = field(default_factory=threading.Event)
    threads: list[threading.Thread] = field(default_factory=list)
    ready: bool = False


def _worker_reply(worker: _PocketWorker, reply: Any) -> None:
    try:
        worker.replies.put_nowait(reply)
    except queue.Full:
        worker.stopped.set()


def _read_worker(worker: _PocketWorker) -> None:
    try:
        while not worker.stopped.is_set():
            line = worker.process.stdout.readline(MAX_MESSAGE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                break
            reply = json.loads(line)
            if not isinstance(reply, dict):
                break
            _worker_reply(worker, reply)
    except (OSError, ValueError):
        pass
    finally:
        _worker_reply(worker, None)


def _write_worker(worker: _PocketWorker) -> None:
    try:
        while not worker.stopped.is_set():
            try:
                message = worker.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            worker.process.stdin.write(message)
            worker.process.stdin.flush()
    except (OSError, ValueError):
        _worker_reply(worker, None)


def _drain_closed(worker: _PocketWorker, pipe: Any) -> None:
    try:
        while not worker.stopped.is_set():
            if not pipe.read(8192):
                break
    except (OSError, ValueError):
        pass


class PocketTTSWorkerProvider(BaseTTS):
    def __init__(
        self,
        voice: str = "alba",
        language: str = "en",
        ref_audio_path: str | Path | None = None,
        device: str = "cpu",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        python_path: str | Path | None = None,
        startup_timeout_sec: float = 300,
        request_timeout_sec: float = 60,
    ):
        from src.utils.platform_compat import resolve_python_executable

        selection = PocketTTSProvider(voice, language, ref_audio_path, device, rate, pitch)
        self._voice = self._resolve_voice(selection.voice)
        self.python_path = resolve_python_executable(
            python_path or ".venv-pocket-tts/Scripts/python.exe", project_root=PROJECT_ROOT,
        )
        self.startup_timeout_sec = self._validate_timeout(startup_timeout_sec)
        self.request_timeout_sec = self._validate_timeout(request_timeout_sec)
        self._request_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._worker: _PocketWorker | None = None
        self._generation = 0
        self._request_id = 0
        self._closed = False

    @staticmethod
    def _validate_timeout(value: float) -> float:
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Pocket TTS worker timeouts must be finite and positive.")
        return timeout

    @staticmethod
    def _resolve_voice(voice: str) -> str:
        source = PocketTTSProvider._validate_voice(voice)
        if isinstance(source, Path):
            from src.utils.platform_compat import resolve_project_path

            source = resolve_project_path(source, project_root=PROJECT_ROOT)
            source = PocketTTSProvider._validate_audio_path(source)
        return str(source)

    @property
    def voice(self) -> str:
        with self._process_lock:
            return self._voice

    @property
    def language(self) -> str:
        return "en"

    @property
    def device(self) -> str:
        return "cpu"

    def _check_locked(self, generation: int) -> None:
        if self._closed:
            raise RuntimeError("Pocket TTS worker is closed; create a new provider to restart.")
        if generation != self._generation:
            raise RuntimeError("Pocket TTS worker request was cancelled.")

    def _snapshot(self) -> tuple[int, str, threading.Event]:
        with self._process_lock:
            self._check_locked(self._generation)
            return self._generation, self._voice, threading.Event()

    def _cancel_marker(self, marker: threading.Event, generation: int) -> None:
        marker.set()
        self._invalidate(generation)

    def _stop_locked(self) -> None:
        from src.utils.platform_compat import kill_process_tree

        worker = self._worker
        if worker is None:
            return
        worker.stopped.set()
        try:
            kill_process_tree(worker.process, timeout=2)
            if worker.process.poll() is None:
                worker.process.kill()
                worker.process.wait(timeout=2)
        finally:
            for pipe in (worker.process.stdin, worker.process.stdout, worker.process.stderr):
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
            self._worker = None
            for thread in worker.threads:
                thread.join(timeout=2)
            if any(thread.is_alive() for thread in worker.threads):
                raise RuntimeError(
                    "Pocket TTS worker stopped but its pipe readers did not exit."
                )

    def _invalidate(self, generation: int) -> None:
        with self._process_lock:
            if generation == self._generation:
                self._generation += 1
                self._stop_locked()

    def cancel_inflight(self) -> None:
        with self._process_lock:
            self._generation += 1
            self._stop_locked()

    def cleanup(self) -> None:
        with self._process_lock:
            self._closed = True
            self._generation += 1
            self._stop_locked()

    def _start_worker(self, generation: int, voice: str) -> _PocketWorker:
        with self._process_lock:
            self._check_locked(generation)
            if self._worker is not None:
                return self._worker
            if not self.python_path.is_file():
                raise RuntimeError(
                    "Pocket TTS worker Python is missing. Create the dedicated .venv-pocket-tts "
                    "environment using requirements-optional-pocket.txt, or set python_path. "
                    "Do not install Pocket TTS into the main runtime."
                )
            if not WORKER_SCRIPT.is_file():
                raise RuntimeError("Pocket TTS worker script is missing: scripts/pocket_tts_worker.py")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            try:
                process = subprocess.Popen(
                    [str(self.python_path), "-u", "-B", str(WORKER_SCRIPT), "--voice", voice,
                     "--language", "en"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=str(PROJECT_ROOT), env=env, shell=False,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                    if os.name == "nt" else 0,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Pocket TTS worker could not start. Check python_path and executable permissions."
                ) from exc
            worker = _PocketWorker(process)
            self._worker = worker
            threading.Thread(target=_drain_closed, args=(worker, worker.process.stderr), daemon=True).start()
            reader = threading.Thread(target=_read_worker, args=(worker,), daemon=True)
            writer = threading.Thread(target=_write_worker, args=(worker,), daemon=True)
            reader.start()
            writer.start()
            worker.threads.extend([reader, writer])
            return worker

    def _wait_reply(
        self, worker: _PocketWorker, generation: int, timeout: float, phase: str,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            with self._process_lock:
                self._check_locked(generation)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Pocket TTS worker {phase} timed out after {timeout:g}s.")
            if worker.stopped.is_set():
                raise RuntimeError(f"Pocket TTS worker {phase} stopped unexpectedly.")
            try:
                reply = worker.replies.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            with self._process_lock:
                self._check_locked(generation)
            if reply is None:
                raise RuntimeError(f"Pocket TTS worker {phase} exited or sent invalid JSON.")
            return reply

    def _request(
        self, operation: str, generation: int, voice: str, marker: threading.Event, text: str = "",
    ) -> dict:
        with self._request_lock:
            if marker.is_set():
                self._invalidate(generation)
                raise RuntimeError("Pocket TTS worker request was cancelled.")
            try:
                worker = self._start_worker(generation, voice)
                if not worker.ready:
                    reply = self._wait_reply(worker, generation, self.startup_timeout_sec, "startup")
                    if reply.get("status") != "ready" or reply.get("protocol") != 1:
                        raise RuntimeError(
                            "Pocket TTS worker startup failed. Check the dedicated environment "
                            "(requirements-optional-pocket.txt), model cache/download access, "
                            "and Hugging Face terms/authentication for cloning weights."
                        )
                    worker.ready = True
                with self._process_lock:
                    self._check_locked(generation)
                    self._request_id += 1
                    request_id = self._request_id
                payload = json.dumps({
                    "id": request_id, "operation": operation, "voice": voice, "text": text,
                }, ensure_ascii=True).encode("utf-8") + b"\n"
                if len(payload) > MAX_REQUEST_BYTES:
                    raise ValueError("Pocket TTS worker request exceeds 1 MiB.")
                worker.requests.put_nowait(payload)
                reply = self._wait_reply(worker, generation, self.request_timeout_sec, operation)
                if reply.get("id") != request_id or reply.get("status") != "ok":
                    raise RuntimeError(
                        f"Pocket TTS worker {operation} failed or returned an invalid response. "
                        "Check the selected voice, model access and isolated dependencies; no fallback was used."
                    )
                with self._process_lock:
                    self._check_locked(generation)
                return reply
            except Exception:
                self._invalidate(generation)
                raise

    def _synthesize_worker(
        self, text: str, output_path: Path | None, generation: int, voice: str,
        marker: threading.Event,
    ) -> TTSResult:
        reply = self._request("synthesize", generation, voice, marker, text)
        try:
            data = base64.b64decode(reply["audio"], validate=True)
            with wave.open(io.BytesIO(data), "rb") as wav:
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
                if (wav.getnchannels() != 1 or wav.getsampwidth() != 2 or
                        sample_rate <= 0 or frames <= 0 or len(wav.readframes(frames)) != frames * 2):
                    raise ValueError("Invalid mono WAV")
            metadata = reply.get("metadata", {})
            synth_ms = float(metadata.get("synth_ms", 0))
            if not math.isfinite(synth_ms) or synth_ms < 0:
                raise ValueError("Invalid synthesis timing")
        except (KeyError, TypeError, ValueError, AttributeError, binascii.Error, wave.Error, EOFError):
            self._invalidate(generation)
            raise RuntimeError("Pocket TTS worker returned invalid WAV audio or metadata.") from None
        with self._process_lock:
            self._check_locked(generation)
            path = Path(output_path) if output_path is not None else None
            if path is not None:
                path.write_bytes(data)
        return TTSResult(
            audio_path=path, audio_data=data if path is None else None,
            duration=frames / sample_rate,
            metadata={"provider": "pocket", "backend": "worker", "voice": voice,
                      "language": "en", "device": "cpu", "sample_rate": sample_rate,
                      "synth_ms": synth_ms},
        )

    async def synthesize(self, text: str, output_path: Path | None = None) -> TTSResult:
        if not text.strip():
            raise ValueError("Pocket TTS cannot synthesize empty text.")
        generation, voice, marker = self._snapshot()
        try:
            return await asyncio.to_thread(
                self._synthesize_worker, text, output_path, generation, voice, marker,
            )
        except asyncio.CancelledError:
            self._cancel_marker(marker, generation)
            raise

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        result = await self.synthesize(text)
        if result.audio_data is not None:
            yield result.audio_data

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        return await PocketTTSProvider(voice=self._voice).list_voices(language)

    def set_voice(self, voice_id: str) -> None:
        voice = self._resolve_voice(voice_id)
        with self._process_lock:
            self._check_locked(self._generation)
            self._voice = voice

    def set_language(self, language: str | None) -> None:
        PocketTTSProvider.set_language(self, language)

    _validate_language = staticmethod(PocketTTSProvider._validate_language)
    set_rate = PocketTTSProvider.set_rate
    set_pitch = PocketTTSProvider.set_pitch

    def preload(self) -> None:
        generation, voice, marker = self._snapshot()
        self._request("preload", generation, voice, marker)

    def warmup(self) -> None:
        generation, voice, marker = self._snapshot()
        self._request("warmup", generation, voice, marker)
