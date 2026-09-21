"""
LIL-67 diagnostic (read-only w.r.t. the project).

Goal: separate three hypotheses for the RVC worker hang/slowdown observed on
2026-09-20 between 14:42 and 14:48:

    H1 VRAM saturation / Windows sysmem fallback
    H2 worker pipe / protocol deadlock
    H3 multi-process GPU contention without memory saturation

Method:
  * generate 5 Kokoro audios at increasing durations (~1, 3, 6, 10, 15 s),
    tiled from a real Kokoro clip (fallback: synthetic amplitude-modulated tone)
  * Phase A: RVC worker alone -> convert each audio twice, 60 s hard timeout,
    record conversion time + nvidia-smi memory.used before/after
  * Phase B: additionally load Whisper large-v3-turbo float16 on CUDA in THIS
    process (same class as the app), warm it up, then alternate
    transcribe -> RVC convert for each audio

Only two project files are written: this script and logs/diag_rvc_hang.txt.
All scratch audio lives in the OS temp directory.

Run:  venv/Scripts/python.exe scripts/diag_rvc_hang.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "resources" / "voices" / "march7th" / "March-7th.pth"
INDEX_PATH = PROJECT_ROOT / "resources" / "voices" / "march7th" / "March-7th.index"
WORKER_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
RVC_OVERLAY = PROJECT_ROOT / ".rvc-overlay"
RVC_WORKER = PROJECT_ROOT / "scripts" / "rvc_worker.py"
REPORT_PATH = PROJECT_ROOT / "logs" / "diag_rvc_hang.txt"

TARGET_DURATIONS = [1.0, 3.0, 6.0, 10.0, 15.0]
REPEATS_PER_AUDIO = 2
CONVERT_TIMEOUT_SEC = 60.0
GLOBAL_DEADLINE_SEC = 7.5 * 60

_REPORT_LINES: list[str] = []
_KOKORO_KEEPALIVE = None


def log_report(line: str = "") -> None:
    _REPORT_LINES.append(line)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_REPORT_LINES) + "\n", encoding="utf-8")


def gpu_memory_mb() -> dict:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
        return {"used": float(parts[0]), "total": float(parts[1]), "free": float(parts[2])}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"used": -1.0, "total": -1.0, "free": -1.0, "error": str(exc)}


def build_kokoro_base(tmp_dir: Path) -> tuple[Path, int, str]:
    global _KOKORO_KEEPALIVE
    sr = 24000
    base_path = tmp_dir / "base_kokoro.wav"
    text = (
        "Hello there. This is a diagnostic recording used to measure the voice "
        "conversion worker. The quick brown fox jumps over the lazy dog several "
        "times, so the resulting clip is long enough to tile into longer samples."
    )
    try:
        from src.tts.kokoro_provider import KokoroProvider

        provider = KokoroProvider(voice="af_heart")
        audio = provider._synthesize_sync(text)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < sr * 1.0:
            raise RuntimeError(f"Kokoro produced only {audio.size} samples")
        sf.write(str(base_path), audio, sr)
        _KOKORO_KEEPALIVE = provider  # keep GPU/CPU pipeline resident during Phase B
        return base_path, sr, "kokoro(af_heart)"
    except Exception as exc:
        sr = 16000
        seconds = 4.0
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        carrier = 0.25 * np.sin(2 * np.pi * 180 * t)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
        audio = (carrier * envelope).astype(np.float32)
        sf.write(str(base_path), audio, sr)
        return base_path, sr, f"synthetic-fallback({exc})"


def make_audio_set(tmp_dir: Path) -> tuple[list[tuple[float, Path]], int, str]:
    base_path, sr, source = build_kokoro_base(tmp_dir)
    base, _ = sf.read(str(base_path), dtype="float32")
    base = np.asarray(base, dtype=np.float32)
    if base.ndim > 1:
        base = base.mean(axis=1)

    audios: list[tuple[float, Path]] = []
    for target in TARGET_DURATIONS:
        wanted = int(round(target * sr))
        reps = int(np.ceil(wanted / max(base.size, 1)))
        tiled = np.tile(base, max(reps, 1))[:wanted]
        out = tmp_dir / f"audio_{target:g}s.wav"
        sf.write(str(out), tiled, sr)
        audios.append((target, out))
    return audios, sr, source


def make_converter():
    from src.tts.rvc_provider import RVCConverter

    return RVCConverter(
        model_path=str(MODEL_PATH),
        index_path=str(INDEX_PATH),
        device="cuda:0",
        f0_method="rmvpe",
        index_rate=0.0,
        protect=0.33,
        backend="worker",
        python_path=str(WORKER_PYTHON),
        site_packages_dir=str(RVC_OVERLAY),
        worker_script=str(RVC_WORKER),
        request_timeout_sec=CONVERT_TIMEOUT_SEC,
    )


def make_whisper():
    from src.asr.whisper_provider import WhisperProvider

    return WhisperProvider(
        model_size="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        beam_size=5,
    )


def transcribe_once(whisper, audio_path: Path, warm: bool) -> tuple[float, str]:
    started = time.perf_counter()
    try:
        result = whisper.transcribe(str(audio_path))
        elapsed = (time.perf_counter() - started) * 1000
        return elapsed, "ok" if warm else f"warmup:{result.text[:40]!r}"
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return elapsed, f"error:{str(exc).splitlines()[0][:80]}"


def whisper_warmup_ok(status: str) -> bool:
    """Return True only when the Whisper warmup transcribe did not fail."""
    return not status.startswith("error:")


def convert_once(converter, audio_path: Path, out_path: Path) -> tuple[float, str]:
    started = time.perf_counter()
    try:
        converter.convert_file(str(audio_path), str(out_path))
        status = "ok"
    except TimeoutError:
        status = "timeout(60s)"
    except Exception as exc:
        status = f"error:{str(exc).splitlines()[0][:80]}"
    elapsed = (time.perf_counter() - started) * 1000
    return elapsed, status


def main() -> int:
    start_time = time.perf_counter()
    deadline = start_time + GLOBAL_DEADLINE_SEC
    tmp_dir = Path(tempfile.mkdtemp(prefix="diag_rvc_"))
    report_rows: list[dict] = []
    max_vram_used = 0.0
    notes: list[str] = []

    try:
        log_report("LIL-67 RVC worker hang diagnostic")
        log_report(f"timestamp_start={time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_report(f"project_root={PROJECT_ROOT}")
        log_report(f"model={MODEL_PATH.name} exists={MODEL_PATH.exists()}")
        log_report(f"worker_python={WORKER_PYTHON} exists={WORKER_PYTHON.exists()}")
        log_report(f"overlay={RVC_OVERLAY} exists={RVC_OVERLAY.exists()}")
        log_report(f"convert_timeout={CONVERT_TIMEOUT_SEC}s global_deadline={GLOBAL_DEADLINE_SEC}s")
        mem = gpu_memory_mb()
        log_report(
            f"gpu_total_mb={mem['total']:.0f} gpu_used_at_start_mb={mem['used']:.0f}"
        )
        log_report()

        audios, sr, source = make_audio_set(tmp_dir)
        log_report(f"audio_source={source} sample_rate={sr}")
        for dur, path in audios:
            info = sf.info(str(path))
            log_report(f"  audio_{dur:g}s -> {info.duration:.2f}s {info.samplerate}Hz")
        log_report()

        converter = make_converter()
        log_report("Phase A: preloading RVC worker ...")
        t_pre = time.perf_counter()
        try:
            converter.preload()
            converter.warmup()
            log_report(f"Phase A worker ready+warmed in {(time.perf_counter()-t_pre)*1000:.0f} ms")
        except Exception as exc:
            log_report(f"Phase A preload/warmup FAILED: {exc}")
            notes.append("RVC worker could not preload; Phase A aborted")
            raise
        mem = gpu_memory_mb()
        max_vram_used = max(max_vram_used, mem["used"])
        log_report(f"VRAM after RVC worker init: used={mem['used']:.0f} free={mem['free']:.0f} MB")
        log_report()

        phase_a_rows = run_phase(
            "A", converter, audios, tmp_dir, deadline, phase_b=False, whisper=None
        )
        report_rows.extend(phase_a_rows)
        for row in phase_a_rows:
            max_vram_used = max(max_vram_used, row["vram_before"], row["vram_after"])

        whisper = None
        try:
            log_report("Phase B: loading Whisper large-v3-turbo float16 on CUDA ...")
            t_load = time.perf_counter()
            whisper = make_whisper()
            w_ms, w_status = transcribe_once(whisper, audios[0][1], warm=False)
            log_report(
                f"Whisper loaded+warmup in {(time.perf_counter()-t_load)*1000:.0f} ms "
                f"(warmup transcribe {w_ms:.0f} ms, {w_status})"
            )
            mem = gpu_memory_mb()
            max_vram_used = max(max_vram_used, mem["used"])
            log_report(
                f"VRAM after Whisper init: used={mem['used']:.0f} free={mem['free']:.0f} MB"
            )
            log_report()
            if not whisper_warmup_ok(w_status):
                notes.append(
                    f"Whisper warmup failed ({w_status}); Phases B and C skipped"
                )
                log_report(
                    "Whisper warmup FAILED; skipping Phase B and Phase C - no "
                    "Whisper/RVC contention data was collected."
                )
                log_report()
            else:
                phase_b_rows = run_phase(
                    "B", converter, audios, tmp_dir, deadline, phase_b=True, whisper=whisper
                )
                report_rows.extend(phase_b_rows)
                for row in phase_b_rows:
                    max_vram_used = max(
                        max_vram_used, row["vram_before"], row["vram_after"]
                    )

                log_report()
                log_report("Phase C: TRUE concurrency - Whisper in background thread "
                           "while RVC worker converts")
                phase_c_rows = run_phase_c(
                    converter, audios, tmp_dir, deadline, whisper
                )
                report_rows.extend(phase_c_rows)
                for row in phase_c_rows:
                    max_vram_used = max(
                        max_vram_used, row["vram_before"], row["vram_after"]
                    )
        except Exception as exc:
            log_report(f"Phase B setup FAILED: {exc}")
            notes.append(f"Phase B setup failed: {exc}")
            log_report(traceback.format_exc())

        write_table(report_rows)
        log_report()
        log_report(f"max_vram_used_observed_mb={max_vram_used:.0f}")
        if mem["total"] > 0:
            log_report(
                f"max_vram_pct={100.0*max_vram_used/mem['total']:.1f}% "
                f"(device total {mem['total']:.0f} MB)"
            )
        if notes:
            log_report("notes:")
            for note in notes:
                log_report(f"  - {note}")
        log_report(f"total_elapsed_sec={time.perf_counter()-start_time:.1f}")

        try:
            converter.close()
        except Exception:
            pass
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log_report("worker_closed=yes scratch_removed=yes")

    return 0


def run_phase(
    phase: str,
    converter,
    audios,
    tmp_dir: Path,
    deadline: float,
    phase_b: bool,
    whisper,
) -> list[dict]:
    rows: list[dict] = []
    for duration, audio_path in audios:
        for rep in range(REPEATS_PER_AUDIO):
            if time.perf_counter() >= deadline:
                log_report(f"!!! global deadline reached during Phase {phase} ({duration:g}s rep{rep})")
                return rows

            whisper_ms = None
            whisper_status = None
            if phase_b and whisper is not None:
                whisper_ms, whisper_status = transcribe_once(whisper, audio_path, warm=True)

            vram_before = gpu_memory_mb()["used"]
            out_path = tmp_dir / f"out_{phase}_{duration:g}_{rep}.wav"
            rvc_ms, status = convert_once(converter, audio_path, out_path)
            vram_after = gpu_memory_mb()["used"]

            row = {
                "phase": phase,
                "duration": duration,
                "rep": rep + 1,
                "rvc_ms": rvc_ms,
                "vram_before": vram_before,
                "vram_after": vram_after,
                "whisper_ms": whisper_ms,
                "status": status,
            }
            rows.append(row)
            w_txt = f"{whisper_ms:.0f}" if whisper_ms is not None else "-"
            log_report(
                f"phase={phase} audio={duration:g}s rep={rep+1} "
                f"rvc_ms={rvc_ms:.0f} vram_before={vram_before:.0f} "
                f"vram_after={vram_after:.0f} whisper_ms={w_txt} status={status}"
            )
            if status != "ok":
                log_report(f"  detail: {status}")
    return rows


class _WhisperLoop:
    """Continuously transcribe one clip in a background thread (app-process side)."""

    def __init__(self, whisper, audio_path: Path):
        self.whisper = whisper
        self.audio_path = audio_path
        self.times: list[float] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="diag-whisper-loop", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                started = time.perf_counter()
                self.whisper.transcribe(str(self.audio_path))
                self.times.append((time.perf_counter() - started) * 1000)
            except Exception as exc:
                self.errors.append(str(exc).splitlines()[0][:80])
                time.sleep(0.05)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=90)

    def recent_ms(self) -> float | None:
        if not self.times:
            return None
        window = self.times[-4:]
        return sum(window) / len(window)


def run_phase_c(converter, audios, tmp_dir: Path, deadline: float, whisper) -> list[dict]:
    rows: list[dict] = []
    by_duration = {d: p for d, p in audios}
    loop = _WhisperLoop(whisper, by_duration[max(by_duration)])
    loop.start()
    time.sleep(1.5)
    try:
        for duration in [6.0, 10.0, 15.0]:
            for rep in range(2):
                if time.perf_counter() >= deadline:
                    log_report(f"!!! global deadline reached during Phase C ({duration:g}s rep{rep})")
                    return rows
                vram_before = gpu_memory_mb()["used"]
                out_path = tmp_dir / f"out_C_{duration:g}_{rep}.wav"
                rvc_ms, status = convert_once(converter, by_duration[duration], out_path)
                vram_after = gpu_memory_mb()["used"]
                row = {
                    "phase": "C",
                    "duration": duration,
                    "rep": rep + 1,
                    "rvc_ms": rvc_ms,
                    "vram_before": vram_before,
                    "vram_after": vram_after,
                    "whisper_ms": loop.recent_ms(),
                    "status": status,
                }
                rows.append(row)
                w_txt = f"{row['whisper_ms']:.0f}" if row["whisper_ms"] is not None else "-"
                log_report(
                    f"phase=C audio={duration:g}s rep={rep+1} "
                    f"rvc_ms={rvc_ms:.0f} vram_before={vram_before:.0f} "
                    f"vram_after={vram_after:.0f} concurrent_whisper_ms={w_txt} status={status}"
                )
    finally:
        loop.stop()
        log_report(
            f"concurrent whisper calls completed={len(loop.times)} "
            f"errors={len(loop.errors)} "
            f"min={min(loop.times) if loop.times else -1:.0f} "
            f"max={max(loop.times) if loop.times else -1:.0f} ms"
        )
        if loop.errors:
            log_report(f"  first whisper error: {loop.errors[0]}")
    return rows


def write_table(rows: list[dict]) -> None:
    log_report()
    log_report("RESULTS TABLE")
    header = (
        f"{'phase':<6} {'audio_s':>7} {'rep':>3} {'rvc_ms':>9} "
        f"{'vram_before':>11} {'vram_after':>10} {'whisper_ms':>10}  status"
    )
    log_report(header)
    log_report("-" * len(header))
    for row in rows:
        w = f"{row['whisper_ms']:.0f}" if row["whisper_ms"] is not None else "-"
        log_report(
            f"{row['phase']:<6} {row['duration']:>7g} {row['rep']:>3} "
            f"{row['rvc_ms']:>9.0f} {row['vram_before']:>11.0f} "
            f"{row['vram_after']:>10.0f} {w:>10}  {row['status']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())