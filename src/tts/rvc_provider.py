"""
RVC (Retrieval-based Voice Conversion) post-processor.

Supported backends:
- ``worker``: persistent subprocess worker with isolated RVC dependencies
- ``inferrvc``: in-process InferRVC backend
- ``rvc_inferpy``: legacy compatibility backend

Recommended Windows setup for this project:
    powershell -ExecutionPolicy Bypass -File scripts/install_rvc_windows.ps1
"""

from __future__ import annotations

import importlib
import hashlib
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RVC_SITE_PACKAGES = PROJECT_ROOT / ".rvc-overlay"
DEFAULT_RVC_WORKER = PROJECT_ROOT / "scripts" / "rvc_worker.py"
DEFAULT_RVC_MODELS_DIR = PROJECT_ROOT / "resources" / "rvc"
DEFAULT_VOICE_MODELS_DIR = PROJECT_ROOT / "resources" / "voices"
_RVC_CWD_LOCK = threading.Lock()
_TORCH_LOAD_LOCK = threading.Lock()


def _normalize_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _slugify_model_name(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def _find_rvc_assets_in_dir(model_dir: Path) -> tuple[Path, Path | None]:
    if not model_dir.exists():
        raise FileNotFoundError(f"RVC model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"RVC model directory is not a folder: {model_dir}")

    pth_files = sorted(model_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth RVC model found in: {model_dir}")
    if len(pth_files) > 1:
        names = ", ".join(path.name for path in pth_files)
        raise RuntimeError(f"Multiple .pth files found in {model_dir}: {names}")

    model_path = pth_files[0]
    index_candidates = sorted(model_dir.glob("*.index"))
    index_path = None
    if index_candidates:
        same_stem = [path for path in index_candidates if path.stem == model_path.stem]
        if same_stem:
            index_path = same_stem[0]
        elif len(index_candidates) == 1:
            index_path = index_candidates[0]
        else:
            names = ", ".join(path.name for path in index_candidates)
            raise RuntimeError(
                f"Multiple .index files found in {model_dir} and none matches {model_path.stem}: {names}"
            )

    return model_path.resolve(), index_path.resolve() if index_path else None


def resolve_rvc_paths(
    rvc_config: dict,
    character_config: dict | None = None,
) -> tuple[Path, Path | None]:
    """
    Resolve RVC model assets from explicit paths or a drop-in directory convention.

    Supported KISS layouts:
    - explicit `model_path` / `index_path`
    - `model_dir`
    - `model_name` -> `resources/rvc/<model_name>/`
    - current character preset -> `resources/voices/<preset>/` or `resources/rvc/<preset>/`
    """
    model_path = rvc_config.get("model_path")
    if model_path:
        resolved_model = Path(model_path).resolve()
        index_path = rvc_config.get("index_path")
        resolved_index = Path(index_path).resolve() if index_path else None
        return resolved_model, resolved_index

    model_dir = rvc_config.get("model_dir")
    if model_dir:
        return _find_rvc_assets_in_dir(Path(model_dir).resolve())

    candidate_dirs: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidate_dirs.append(resolved)

    model_name = _slugify_model_name(rvc_config.get("model_name"))
    if model_name:
        add_candidate(DEFAULT_RVC_MODELS_DIR / model_name)
        add_candidate(DEFAULT_VOICE_MODELS_DIR / model_name)

    if character_config:
        preset = _slugify_model_name(character_config.get("preset"))
        name = _slugify_model_name(character_config.get("name"))
        for token in (preset, name):
            if token:
                add_candidate(DEFAULT_VOICE_MODELS_DIR / token)
                add_candidate(DEFAULT_RVC_MODELS_DIR / token)

    errors: list[str] = []
    for candidate in candidate_dirs:
        if not candidate.exists():
            continue
        try:
            return _find_rvc_assets_in_dir(candidate)
        except Exception as exc:
            errors.append(str(exc))

    search_hint = (
        "\n".join(f"  - {path}" for path in candidate_dirs)
        or "  - no candidate directories"
    )
    details = f"\nDetails:\n" + "\n".join(errors) if errors else ""
    raise FileNotFoundError(
        "Unable to resolve RVC model assets.\n"
        "Use one of: `model_path`, `model_dir`, or `model_name`.\n"
        f"Searched directories:\n{search_hint}{details}"
    )


class RVCConverter:
    """
    Voice converter using RVC v2 models.

    Args:
        model_path: Path to the .pth RVC model file.
        index_path: Optional path to the .index FAISS file.
        device: "cuda:0" or "cpu".
        f0_method: Pitch extraction method ("rmvpe", "crepe", "harvest", ...).
        index_rate: How much to use the index (0.0-1.0). Higher = more similar.
        protect: Protect voiceless consonants (0.0-0.5).
        backend: "auto", "worker", "inferrvc", or "rvc_inferpy".
        python_path: Python executable used by the worker backend.
        site_packages_dir: Optional directory containing isolated RVC dependencies.
        worker_script: Worker script path. Defaults to scripts/rvc_worker.py.
        f0_up_key: Pitch shift in semitones.
        output_freq: Optional worker output sample rate override.
    """

    def __init__(
        self,
        model_path: str | Path,
        index_path: str | Path | None = None,
        device: str = "cuda:0",
        f0_method: str = "rmvpe",
        index_rate: float = 0.0,
        protect: float = 0.33,
        backend: str = "auto",
        python_path: str | Path | None = None,
        site_packages_dir: str | Path | None = None,
        worker_script: str | Path | None = None,
        f0_up_key: float = 0.0,
        output_freq: int | None = None,
        request_timeout_sec: float = 15.0,
        model_sha256: str | None = None,
        index_sha256: str | None = None,
    ):
        self.model_path = Path(model_path).resolve()
        requested_index_path = Path(index_path).resolve() if index_path else None
        self.model_sha256 = _normalize_sha256(model_sha256)
        self.index_sha256 = _normalize_sha256(index_sha256)
        self.device = device
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.index_path = requested_index_path
        self.protect = protect
        self.backend = backend
        self.python_path = (
            Path(python_path).resolve()
            if python_path
            else Path(sys.executable).resolve()
        )
        self.site_packages_dir = (
            Path(site_packages_dir).resolve()
            if site_packages_dir
            else DEFAULT_RVC_SITE_PACKAGES.resolve()
        )
        self.worker_script = (
            Path(worker_script).resolve()
            if worker_script
            else DEFAULT_RVC_WORKER.resolve()
        )
        self.f0_up_key = f0_up_key
        self.output_freq = output_freq
        self.request_timeout_sec = float(request_timeout_sec)
        self.voice_model = self.model_path.stem
        self.models_root = PROJECT_ROOT / "models"
        self.voice_model_dir = self.models_root / self.voice_model
        self._converter = None
        self._backend_name: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._worker_process: subprocess.Popen[str] | None = None
        self._worker_lock = threading.Lock()
        self._worker_stderr: deque[str] = deque(maxlen=50)
        self._worker_stderr_thread: threading.Thread | None = None
        self._warmed_up = False

        if requested_index_path and self.index_rate <= 0:
            logger.info(
                "RVC FAISS index disabled because index_rate <= 0. "
                "This is the recommended realtime-safe default on Windows."
            )

    def _effective_index_path(self) -> Path | None:
        return self.index_path if self.index_rate > 0 else None

    @staticmethod
    def install_hint() -> str:
        return (
            "RVC is optional. On Windows, install it with:\n"
            "  powershell -ExecutionPolicy Bypass -File scripts/install_rvc_windows.ps1 -Clean"
        )

    @staticmethod
    @contextmanager
    def _sanitized_argv():
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            yield
        finally:
            sys.argv = original_argv

    @classmethod
    def _legacy_backend_class(cls):
        module = importlib.import_module("rvc_inferpy")
        backend = getattr(module, "RVCConverter", None)
        if backend is None:
            raise ImportError("rvc_inferpy.RVCConverter was not found")
        return backend

    @classmethod
    def _inferrvc_api(cls):
        with cls._sanitized_argv():
            module = importlib.import_module("inferrvc")
            config_module = importlib.import_module("inferrvc.configs.config")
        backend = getattr(module, "RVC", None)
        config_class = getattr(config_module, "Config", None)
        if backend is None or config_class is None:
            raise ImportError(
                "inferrvc.RVC or inferrvc.configs.config.Config was not found"
            )
        return backend, config_class

    @classmethod
    def _worker_import_check(
        cls,
        python_path: Path,
        worker_script: Path,
        site_packages_dir: Path | None = None,
    ) -> bool:
        command = [str(python_path), str(worker_script), "--check-imports"]
        if site_packages_dir and site_packages_dir.exists():
            command.extend(["--site-packages-dir", str(site_packages_dir)])

        try:
            result = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
        except Exception:
            return False

        if result.returncode != 0:
            return False

        stdout_lines = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        if not stdout_lines:
            return False

        try:
            payload = json.loads(stdout_lines[-1])
        except json.JSONDecodeError:
            return False

        return payload.get("status") == "ok"

    @classmethod
    def is_available(
        cls,
        backend: str = "auto",
        python_path: str | Path | None = None,
        site_packages_dir: str | Path | None = None,
        worker_script: str | Path | None = None,
    ) -> bool:
        """Check if at least one configured RVC backend is usable."""
        python_path = (
            Path(python_path).resolve()
            if python_path
            else Path(sys.executable).resolve()
        )
        site_packages_dir = (
            Path(site_packages_dir).resolve()
            if site_packages_dir
            else DEFAULT_RVC_SITE_PACKAGES.resolve()
        )
        worker_script = (
            Path(worker_script).resolve()
            if worker_script
            else DEFAULT_RVC_WORKER.resolve()
        )

        if backend == "worker":
            if not python_path.exists() or not worker_script.exists():
                return False
            return cls._worker_import_check(
                python_path, worker_script, site_packages_dir
            )
        if backend == "inferrvc":
            try:
                cls._inferrvc_api()
                return True
            except Exception:
                return False
        if backend == "rvc_inferpy":
            try:
                cls._legacy_backend_class()
                return True
            except Exception:
                return False
        if backend != "auto":
            return False

        if python_path.exists() and worker_script.exists():
            if cls._worker_import_check(python_path, worker_script, site_packages_dir):
                return True

        if (
            site_packages_dir.exists()
            and python_path.exists()
            and worker_script.exists()
        ):
            return True

        try:
            cls._inferrvc_api()
            return True
        except Exception:
            pass

        try:
            cls._legacy_backend_class()
            return True
        except Exception:
            return False

    @staticmethod
    def _sync_file(src: Path, dst: Path) -> None:
        if not dst.exists():
            shutil.copy2(src, dst)
            return
        src_stat = src.stat()
        dst_stat = dst.stat()
        if (
            src_stat.st_size != dst_stat.st_size
            or src_stat.st_mtime_ns != dst_stat.st_mtime_ns
        ):
            shutil.copy2(src, dst)

    def _ensure_model_files(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"RVC model not found: {self.model_path}")
        self._verify_expected_sha256(self.model_path, self.model_sha256, "RVC model")
        require_index = self.index_rate > 0 or self.backend == "rvc_inferpy"
        if require_index and self.index_path and not self.index_path.exists():
            raise FileNotFoundError(f"RVC index not found: {self.index_path}")
        if self.index_path and self.index_sha256:
            self._verify_expected_sha256(self.index_path, self.index_sha256, "RVC index")

    @staticmethod
    def _verify_expected_sha256(path: Path, expected_sha256: str | None, label: str) -> None:
        if not expected_sha256:
            return

        digest = _sha256_file(path)
        if digest != expected_sha256:
            raise RuntimeError(
                f"{label} SHA-256 mismatch for {path.name}: expected "
                f"{expected_sha256}, got {digest}. Refusing to load this local model file."
            )

    def _ensure_voice_model_workspace(self) -> None:
        self._ensure_model_files()
        self.voice_model_dir.mkdir(parents=True, exist_ok=True)
        self._sync_file(self.model_path, self.voice_model_dir / self.model_path.name)
        if self.index_path:
            self._sync_file(
                self.index_path, self.voice_model_dir / self.index_path.name
            )

    @contextmanager
    def _project_cwd(self):
        with _RVC_CWD_LOCK:
            previous_cwd = Path.cwd()
            os.chdir(PROJECT_ROOT)
            try:
                yield
            finally:
                os.chdir(previous_cwd)

    def _resolve_backend(self) -> str:
        if self.backend in {"worker", "inferrvc", "rvc_inferpy"}:
            return self.backend

        if self.backend != "auto":
            raise ValueError(f"Unsupported RVC backend: {self.backend}")

        if (
            self.site_packages_dir.exists()
            and self.worker_script.exists()
            and self.python_path.exists()
        ):
            return "worker"

        try:
            self._inferrvc_api()
            return "inferrvc"
        except Exception:
            pass

        try:
            self._legacy_backend_class()
            return "rvc_inferpy"
        except Exception as exc:
            raise ImportError(
                f"No supported RVC backend found.\n{self.install_hint()}"
            ) from exc

    def _build_inferrvc_config(self, config_class):
        with self._sanitized_argv():
            config = config_class()

        if self.device == "cpu":
            config.device = "cpu"
            config.is_half = False
            config.x_pad = 1
            config.x_query = 6
            config.x_center = 38
            config.x_max = 41
        else:
            config.device = self.device

        return config

    @contextmanager
    def _legacy_torch_load_context(self):
        """
        InferRVC still expects pre-PyTorch-2.6 checkpoint loading behavior.

        We keep the override scoped to RVC calls instead of changing torch.load
        globally for the whole application process.
        """
        import torch

        patch_targets: list[tuple[object, str, object]] = []

        with _TORCH_LOAD_LOCK:

            def compat_torch_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            original_torch_load = torch.load

            def patch_attr(target_obj, attr_name: str):
                if target_obj is None or not hasattr(target_obj, attr_name):
                    return
                patch_targets.append(
                    (target_obj, attr_name, getattr(target_obj, attr_name))
                )
                setattr(target_obj, attr_name, compat_torch_load)

            patch_attr(torch, "load")
            patch_attr(getattr(torch, "serialization", None), "load")

            for module_name in ("fairseq.checkpoint_utils", "inferrvc.modules"):
                module = sys.modules.get(module_name)
                module_torch = getattr(module, "torch", None) if module else None
                patch_attr(module_torch, "load")
            try:
                yield
            finally:
                while patch_targets:
                    target_obj, attr_name, original = patch_targets.pop()
                    setattr(target_obj, attr_name, original)

    def _load_inferrvc(self) -> None:
        self._ensure_model_files()
        backend_class, config_class = self._inferrvc_api()
        config = self._build_inferrvc_config(config_class)
        effective_index_path = self._effective_index_path()
        with self._legacy_torch_load_context():
            self._converter = backend_class(
                model=str(self.model_path),
                index=str(effective_index_path) if effective_index_path else None,
                config=config,
            )
        native_sr = int(getattr(self._converter, "tgt_sr", 0) or 0)
        if native_sr > 0:
            setattr(self._converter, "outputfreq", native_sr)
        self._backend_name = "inferrvc"
        logger.info("Loaded RVC backend: %s", self._backend_name)

    @staticmethod
    def _resample_audio_array(
        audio_array: np.ndarray,
        source_rate: int,
        target_rate: int | None,
    ) -> tuple[np.ndarray, int]:
        if not target_rate or target_rate == source_rate:
            return audio_array, source_rate

        from scipy.signal import resample_poly

        up = int(target_rate)
        down = int(source_rate)
        factor = math.gcd(up, down)
        up //= factor
        down //= factor
        resampled = resample_poly(audio_array, up, down).astype(np.float32, copy=False)
        return resampled, int(target_rate)

    def _create_legacy_backend(self):
        backend_class = self._legacy_backend_class()
        kwargs = {
            "device": self.device,
            "is_half": self.device != "cpu",
            "models_dir": self.models_root,
            "download_if_missing": True,
        }
        try:
            return backend_class(**kwargs)
        except TypeError:
            kwargs.pop("models_dir", None)
            kwargs.pop("download_if_missing", None)
            try:
                return backend_class(**kwargs)
            except TypeError:
                kwargs.pop("is_half", None)
                return backend_class(**kwargs)

    def _load_legacy_backend(self) -> None:
        self._ensure_model_files()
        converter = self._create_legacy_backend()

        if hasattr(converter, "infer_audio"):
            self._ensure_voice_model_workspace()
            self._converter = converter
            self._backend_name = "rvc_inferpy.infer_audio"
            logger.info("Loaded RVC backend: %s", self._backend_name)
            return

        if hasattr(converter, "load_model") and hasattr(converter, "infer_file"):
            converter.load_model(str(self.model_path))
            self._converter = converter
            self._backend_name = "rvc_inferpy.infer_file"
            logger.info("Loaded RVC backend: %s", self._backend_name)
            return

        raise RuntimeError(
            "Unsupported rvc_inferpy API. Expected infer_audio() or load_model()/infer_file()."
        )

    def _drain_worker_stderr(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                text = line.rstrip()
                if text:
                    self._worker_stderr.append(text)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _worker_command(self) -> list[str]:
        effective_index_path = self._effective_index_path()
        command = [
            str(self.python_path),
            str(self.worker_script),
            "--model-path",
            str(self.model_path),
            "--device",
            self.device,
            "--f0-method",
            self.f0_method,
            "--index-rate",
            str(self.index_rate),
            "--protect",
            str(self.protect),
            "--f0-up-key",
            str(self.f0_up_key),
        ]
        if effective_index_path:
            command.extend(["--index-path", str(effective_index_path)])
        if self.output_freq:
            command.extend(["--output-freq", str(self.output_freq)])
        if self.site_packages_dir.exists():
            command.extend(["--site-packages-dir", str(self.site_packages_dir)])
        return command

    def _worker_error_summary(self) -> str:
        if not self._worker_stderr:
            return "No worker stderr captured."
        return "\n".join(list(self._worker_stderr)[-10:])

    def _reset_worker_state(self) -> None:
        self._worker_process = None
        self._converter = None

    def _terminate_worker_process(self) -> None:
        process = self._worker_process
        if process is None:
            self._reset_worker_state()
            return

        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except Exception:
            pass
        finally:
            self._reset_worker_state()

    def _read_worker_response_line(
        self,
        timeout_sec: float,
        *,
        operation: str = "response",
    ) -> str:
        if self._worker_process is None or self._worker_process.stdout is None:
            raise RuntimeError("RVC worker stdout is not available")

        result_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)

        def _reader() -> None:
            try:
                line = self._worker_process.stdout.readline()
            except Exception as exc:
                result_queue.put(("error", str(exc)))
            else:
                result_queue.put(("ok", line))

        threading.Thread(target=_reader, daemon=True, name="RVCWorkerRead").start()

        try:
            status, payload = result_queue.get(timeout=max(timeout_sec, 0.1))
        except queue.Empty as exc:
            self._terminate_worker_process()
            raise TimeoutError(
                f"RVC worker {operation} timed out after {timeout_sec:.1f}s.\n"
                f"{self._worker_error_summary()}"
            ) from exc

        if status == "error":
            raise RuntimeError(f"RVC worker stdout read failed: {payload}")

        return payload

    def _start_worker(self) -> None:
        self._ensure_model_files()
        if not self.python_path.exists():
            raise FileNotFoundError(f"Worker python not found: {self.python_path}")
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"RVC worker script not found: {self.worker_script}"
            )

        self._worker_process = subprocess.Popen(
            self._worker_command(),
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        assert self._worker_process.stderr is not None
        self._worker_stderr_thread = threading.Thread(
            target=self._drain_worker_stderr,
            args=(self._worker_process.stderr,),
            daemon=True,
        )
        self._worker_stderr_thread.start()

        assert self._worker_process.stdout is not None
        try:
            ready_line = self._read_worker_response_line(
                self.request_timeout_sec,
                operation="startup",
            ).strip()
            if not ready_line:
                raise RuntimeError(
                    "RVC worker exited before initialization.\n"
                    f"{self._worker_error_summary()}\n{self.install_hint()}"
                )

            try:
                ready_payload = json.loads(ready_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "RVC worker returned an invalid startup response.\n"
                    f"stdout: {ready_line}\n{self._worker_error_summary()}"
                ) from exc

            if ready_payload.get("status") != "ready":
                raise RuntimeError(
                    "RVC worker failed to start.\n"
                    f"{ready_payload}\n{self._worker_error_summary()}"
                )
        except Exception:
            self._terminate_worker_process()
            raise

        self._converter = self._worker_process
        self._backend_name = "worker"
        logger.info("Loaded RVC backend: %s", self._backend_name)

    def _load(self) -> None:
        """Lazy-load the selected RVC backend."""
        if self._converter is not None:
            return

        selected_backend = self._resolve_backend()
        if selected_backend == "worker":
            self._start_worker()
            return
        if selected_backend == "inferrvc":
            self._load_inferrvc()
            return
        if selected_backend == "rvc_inferpy":
            self._load_legacy_backend()
            return
        raise RuntimeError(f"Unsupported backend selection: {selected_backend}")

    def _write_inferrvc_output(self, audio_tensor, output_path: Path) -> Path:
        if hasattr(audio_tensor, "detach"):
            audio_array = audio_tensor.detach().float().cpu().numpy()
        else:
            audio_array = np.asarray(audio_tensor, dtype=np.float32)

        audio_array = np.asarray(audio_array, dtype=np.float32).squeeze()
        if audio_array.ndim != 1:
            raise RuntimeError(f"Unexpected InferRVC output shape: {audio_array.shape}")

        sample_rate = int(
            getattr(self._converter, "tgt_sr", None)
            or getattr(self._converter, "outputfreq", None)
            or 44100
        )
        audio_array, sample_rate = self._resample_audio_array(
            audio_array,
            sample_rate,
            self.output_freq,
        )
        sf.write(output_path, audio_array, sample_rate)
        return output_path

    def _prepare_inferrvc_input(self, input_path: Path) -> tuple[Path, Path | None]:
        config = getattr(self._converter, "config", None)
        x_pad = float(getattr(config, "x_pad", 3))
        minimum_duration_sec = x_pad + 0.05

        audio, sample_rate = sf.read(input_path, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        current_duration_sec = audio.shape[0] / float(sample_rate)
        if current_duration_sec > minimum_duration_sec:
            return input_path, None

        target_samples = int(np.ceil(sample_rate * minimum_duration_sec))
        pad_samples = max(0, target_samples - audio.shape[0])
        if pad_samples == 0:
            return input_path, None

        pad_left = pad_samples // 2
        pad_right = pad_samples - pad_left
        pad_mode = "edge" if audio.shape[0] > 1 else "constant"
        padded_audio = np.pad(audio, (pad_left, pad_right), mode=pad_mode)

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        temp_path = Path(temp_file.name)
        sf.write(temp_path, padded_audio, sample_rate)
        return temp_path, temp_path

    def _convert_file_with_worker(self, input_path: Path, output_path: Path) -> Path:
        if self._worker_process is None or self._worker_process.poll() is not None:
            raise RuntimeError(
                f"RVC worker is not running.\n{self._worker_error_summary()}"
            )

        payload = {
            "command": "convert",
            "input_path": str(input_path),
            "output_path": str(output_path),
        }

        with self._worker_lock:
            assert self._worker_process.stdin is not None
            self._worker_process.stdin.write(json.dumps(payload) + "\n")
            self._worker_process.stdin.flush()

            response_line = self._read_worker_response_line(self.request_timeout_sec).strip()
            if not response_line:
                raise RuntimeError(
                    f"RVC worker returned no response.\n{self._worker_error_summary()}"
                )

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "RVC worker returned invalid JSON.\n"
                f"stdout: {response_line}\n{self._worker_error_summary()}"
            ) from exc

        if response.get("status") != "ok":
            raise RuntimeError(
                "RVC worker conversion failed.\n"
                f"{response}\n{self._worker_error_summary()}"
            )

        converted_path = Path(response.get("output_path", output_path)).resolve()
        if not converted_path.exists():
            raise RuntimeError(
                f"RVC worker reported success but output is missing: {converted_path}"
            )
        return converted_path

    def convert_file(self, input_path: str | Path, output_path: str | Path) -> Path:
        """Convert an audio file to the target voice."""
        self._load()

        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend_name == "worker":
            converted_path = self._convert_file_with_worker(input_path, output_path)
            if converted_path != output_path:
                shutil.move(str(converted_path), str(output_path))
            return output_path

        if self._backend_name == "inferrvc":
            prepared_input_path, temp_input_path = self._prepare_inferrvc_input(
                input_path
            )
            try:
                with self._legacy_torch_load_context():
                    audio_tensor = self._converter(
                        str(prepared_input_path),
                        f0_up_key=self.f0_up_key,
                        f0_method=self.f0_method,
                        index_rate=self.index_rate,
                        protect=self.protect,
                        output_volume=getattr(self._converter, "NO_CHANGE", 2),
                    )
                return self._write_inferrvc_output(audio_tensor, output_path)
            finally:
                if temp_input_path:
                    temp_input_path.unlink(missing_ok=True)

        if self._backend_name == "rvc_inferpy.infer_audio":
            with self._project_cwd():
                generated_path = self._converter.infer_audio(
                    voice_model=self.voice_model,
                    audio_path=str(input_path),
                    f0_change=self.f0_up_key,
                    f0_method=self.f0_method,
                    index_rate=self.index_rate,
                    protect=self.protect,
                    split_infer=False,
                    audio_format=output_path.suffix.lstrip(".") or "wav",
                )

            if not generated_path:
                raise RuntimeError("RVC inference returned no output path")

            generated_path = Path(generated_path).resolve()
            if generated_path != output_path:
                shutil.move(str(generated_path), str(output_path))
            return output_path

        self._converter.infer_file(
            str(input_path),
            str(output_path),
            f0_method=self.f0_method,
            index_rate=self.index_rate,
            protect=self.protect,
        )
        return output_path

    def preload(self):
        """Force backend/model initialization ahead of the first conversion."""
        self._load()
        return self

    def warmup(self) -> None:
        """
        Trigger one tiny conversion to fetch auxiliary models and JIT-heavy paths.

        InferRVC downloads supporting artifacts such as HuBERT / RMVPE lazily on the
        first real conversion. Doing it during preload makes the first spoken reply
        much less surprising.
        """
        if self._warmed_up:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            input_path = Path(tmp_in.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = Path(tmp_out.name)

        try:
            warmup_audio = np.zeros(16000, dtype=np.float32)
            sf.write(input_path, warmup_audio, 16000)
            self.convert_file(input_path, output_path)
            self._warmed_up = True
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

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
            Path(tmp_in_path).unlink(missing_ok=True)
            Path(tmp_out_path).unlink(missing_ok=True)

    async def convert_array_async(
        self, audio: np.ndarray, sr: int
    ) -> tuple[np.ndarray, int]:
        """Async wrapper for convert_array."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.convert_array, audio, sr)

    def close(self) -> None:
        """Release worker resources if needed."""
        if self._worker_process is None:
            return

        try:
            if self._worker_process.poll() is None and self._worker_process.stdin:
                self._worker_process.stdin.write(
                    json.dumps({"command": "shutdown"}) + "\n"
                )
                self._worker_process.stdin.flush()
        except Exception:
            pass
        finally:
            self._terminate_worker_process()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
