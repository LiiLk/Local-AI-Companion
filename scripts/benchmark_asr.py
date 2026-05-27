"""Live voice benchmark for the local ASR and turn-taking path.

The benchmark intentionally uses the real microphone path instead of prepared
audio fixtures:

    microphone -> AudioService -> Silero VAD -> delayed commit timing -> ASR

It captures live utterances, asks for the expected transcript, then compares
selected Whisper profiles on the exact same live segments.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import time
import unicodedata
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class LiveSegment:
    id: str
    path: Path
    sample_rate: int
    bytes_len: int
    audio_duration_sec: float
    capture_start_ms: float | None
    capture_end_ms: float | None
    gap_from_previous_end_ms: float | None
    starts_before_previous_commit: bool | None
    vad_wall_ms: float | None
    commit_delay_ms: int
    commit_ready_ms: float | None
    reference: str | None = None
    was_cut_off: bool | None = None
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    segment_id: str
    provider: str
    model: str
    language: str | None
    requested_language: str | None
    reference: str | None
    hypothesis: str
    audio_duration_sec: float
    transcribe_ms: float
    vad_end_to_transcript_ready_ms: float
    realtime_factor: float | None
    wer: float | None
    cer: float | None
    confidence: float | None
    effective_device: str | None
    effective_compute_type: str | None
    was_cut_off: bool | None


@dataclass(frozen=True)
class WarmupResult:
    provider: str
    model: str
    segment_id: str
    run_index: int
    transcribe_ms: float


def configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def normalize_text(text: str, *, strip_accents: bool = True) -> str:
    clean = unicodedata.normalize("NFKC", text or "").lower()
    if strip_accents:
        clean = "".join(
            char
            for char in unicodedata.normalize("NFD", clean)
            if unicodedata.category(char) != "Mn"
        )
    clean = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in clean
    )
    return " ".join(clean.split())


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def audio_duration_from_bytes(audio_bytes: bytes, sample_rate: int) -> float:
    return len(audio_bytes) / 2.0 / sample_rate


def write_pcm16_wav(path: Path, audio_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_bytes)


def wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, OSError):
        return None


def create_whisper_provider(model: str, device: str, compute_type: str, beam_size: int):
    from src.asr.whisper_provider import WhisperProvider

    return WhisperProvider(
        model_size=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
    )


def run_whisper_case(
    asr,
    *,
    model: str,
    segment: LiveSegment,
    forced_language: str | None,
) -> BenchmarkResult:
    started = time.perf_counter()
    result = asr.transcribe(segment.path, language=forced_language)
    elapsed_ms = (time.perf_counter() - started) * 1000
    model_info = asr.get_model_info()
    return BenchmarkResult(
        segment_id=segment.id,
        provider="whisper",
        model=model,
        language=result.language,
        requested_language=forced_language,
        reference=segment.reference,
        hypothesis=result.text,
        audio_duration_sec=segment.audio_duration_sec,
        transcribe_ms=elapsed_ms,
        vad_end_to_transcript_ready_ms=segment.commit_delay_ms + elapsed_ms,
        realtime_factor=(
            segment.audio_duration_sec / (elapsed_ms / 1000)
            if segment.audio_duration_sec and elapsed_ms > 0
            else None
        ),
        wer=word_error_rate(segment.reference, result.text) if segment.reference else None,
        cer=char_error_rate(segment.reference, result.text) if segment.reference else None,
        confidence=result.confidence,
        effective_device=model_info.get("effective_device"),
        effective_compute_type=model_info.get("effective_compute_type"),
        was_cut_off=segment.was_cut_off,
    )


def run_warmup(
    asr,
    *,
    model: str,
    segment: LiveSegment,
    forced_language: str | None,
    run_index: int,
) -> WarmupResult:
    started = time.perf_counter()
    asr.transcribe(segment.path, language=forced_language)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return WarmupResult(
        provider="whisper",
        model=model,
        segment_id=segment.id,
        run_index=run_index,
        transcribe_ms=elapsed_ms,
    )


def _mean_optional(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def summarize_results(results: Iterable[BenchmarkResult]) -> list[dict]:
    grouped: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault((result.provider, result.model), []).append(result)

    summary = []
    for (provider, model), items in sorted(grouped.items()):
        uncut_items = [item for item in items if item.was_cut_off is not True]
        rtf_values = [item.realtime_factor for item in items if item.realtime_factor is not None]
        summary.append(
            {
                "provider": provider,
                "model": model,
                "segments": len(items),
                "mean_wer": _mean_optional(item.wer for item in items),
                "mean_cer": _mean_optional(item.cer for item in items),
                "mean_wer_without_cutoffs": _mean_optional(item.wer for item in uncut_items),
                "mean_cer_without_cutoffs": _mean_optional(item.cer for item in uncut_items),
                "median_ms": statistics.median(item.transcribe_ms for item in items),
                "median_vad_end_to_transcript_ready_ms": statistics.median(
                    item.vad_end_to_transcript_ready_ms for item in items
                ),
                "median_realtime_factor": statistics.median(rtf_values) if rtf_values else None,
                "cutoff_reports": sum(1 for item in items if item.was_cut_off is True),
            }
        )
    return summary


def capture_live_segments(
    *,
    output_dir: Path,
    target_segments: int,
    max_duration_sec: int,
    commit_delay_ms: int,
    audio_config: dict,
) -> list[LiveSegment]:
    from src.assistant.audio_service import AudioService, AudioServiceConfig

    events: queue.Queue[tuple[str, object]] = queue.Queue()
    pending_start: float | None = None
    pending_audio: bytes | None = None
    previous_end_ms: float | None = None

    config = AudioServiceConfig(**audio_config)
    service = AudioService(config)

    def on_speech_start() -> None:
        events.put(("start", time.perf_counter()))

    def on_speech_detected(audio_bytes: bytes) -> None:
        events.put(("audio", audio_bytes))

    def on_speech_end() -> None:
        events.put(("end", time.perf_counter()))

    service.on_speech_start = on_speech_start
    service.on_speech_detected = on_speech_detected
    service.on_speech_end = on_speech_end

    segments: list[LiveSegment] = []
    started = time.perf_counter()
    print("Listening. Speak naturally; press Ctrl+C to stop early.")
    service.start()
    try:
        while len(segments) < target_segments and (time.perf_counter() - started) < max_duration_sec:
            try:
                event_name, payload = events.get(timeout=0.2)
            except queue.Empty:
                continue

            if event_name == "start":
                pending_start = float(payload)
                pending_audio = None
                print("Speech started...")
            elif event_name == "audio":
                pending_audio = bytes(payload)
            elif event_name == "end":
                end_time = float(payload)
                if pending_audio:
                    segment_id = f"live_{len(segments) + 1:03d}"
                    path = output_dir / "segments" / f"{segment_id}.wav"
                    write_pcm16_wav(path, pending_audio, config.sample_rate)
                    vad_wall_ms = (
                        (end_time - pending_start) * 1000.0
                        if pending_start is not None
                        else None
                    )
                    capture_start_ms = (
                        (pending_start - started) * 1000.0
                        if pending_start is not None
                        else None
                    )
                    capture_end_ms = (end_time - started) * 1000.0
                    gap_from_previous_end_ms = (
                        capture_start_ms - previous_end_ms
                        if capture_start_ms is not None and previous_end_ms is not None
                        else None
                    )
                    starts_before_previous_commit = (
                        gap_from_previous_end_ms <= commit_delay_ms
                        if gap_from_previous_end_ms is not None
                        else None
                    )
                    audio_duration = audio_duration_from_bytes(pending_audio, config.sample_rate)
                    segments.append(
                        LiveSegment(
                            id=segment_id,
                            path=path,
                            sample_rate=config.sample_rate,
                            bytes_len=len(pending_audio),
                            audio_duration_sec=audio_duration,
                            capture_start_ms=capture_start_ms,
                            capture_end_ms=capture_end_ms,
                            gap_from_previous_end_ms=gap_from_previous_end_ms,
                            starts_before_previous_commit=starts_before_previous_commit,
                            vad_wall_ms=vad_wall_ms,
                            commit_delay_ms=commit_delay_ms,
                            commit_ready_ms=(vad_wall_ms + commit_delay_ms) if vad_wall_ms is not None else None,
                        )
                    )
                    previous_end_ms = capture_end_ms
                    print(f"Captured {segment_id}: {audio_duration:.2f}s -> {path}")
                pending_start = None
                pending_audio = None
    except KeyboardInterrupt:
        print("Stopping live capture...")
    finally:
        service.stop()

    return segments


def annotate_segments(segments: list[LiveSegment]) -> list[LiveSegment]:
    annotated: list[LiveSegment] = []
    print("")
    print("Enter the sentence you actually said for each segment. Leave empty to skip WER/CER.")
    for segment in segments:
        print("")
        print(f"{segment.id}: {segment.path} ({segment.audio_duration_sec:.2f}s)")
        reference = input("Expected transcript: ").strip() or None
        cutoff_answer = input("Did the app/VAD cut you off? [y/N]: ").strip().lower()
        was_cut_off = True if cutoff_answer in {"y", "yes"} else False
        notes = input("Notes (optional): ").strip()
        annotated.append(
            LiveSegment(
                id=segment.id,
                path=segment.path,
                sample_rate=segment.sample_rate,
                bytes_len=segment.bytes_len,
                audio_duration_sec=segment.audio_duration_sec,
                capture_start_ms=segment.capture_start_ms,
                capture_end_ms=segment.capture_end_ms,
                gap_from_previous_end_ms=segment.gap_from_previous_end_ms,
                starts_before_previous_commit=segment.starts_before_previous_commit,
                vad_wall_ms=segment.vad_wall_ms,
                commit_delay_ms=segment.commit_delay_ms,
                commit_ready_ms=segment.commit_ready_ms,
                reference=reference,
                was_cut_off=was_cut_off,
                notes=notes,
            )
        )
    return annotated


def write_json_report(
    path: Path,
    segments: list[LiveSegment],
    results: list[BenchmarkResult],
    warmups: list[WarmupResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_epoch": int(time.time()),
        "summary": summarize_results(results),
        "warmups": [asdict(warmup) for warmup in warmups],
        "segments": [{**asdict(segment), "path": str(segment.path)} for segment in segments],
        "results": [{**asdict(result)} for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt_optional_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _fmt_optional_float(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.1f}{suffix}"


def write_markdown_report(
    path: Path,
    segments: list[LiveSegment],
    results: list[BenchmarkResult],
    warmups: list[WarmupResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Live Voice Benchmark Results",
        "",
        "## Segments",
        "",
        "| Segment | Audio s | VAD wall ms | Commit ready ms | Cut off | Reference |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for segment in segments:
        reference = (segment.reference or "").replace("|", "\\|")
        lines.append(
            f"| {segment.id} | {segment.audio_duration_sec:.2f} | "
            f"{_fmt_optional_float(segment.vad_wall_ms)} | "
            f"{_fmt_optional_float(segment.commit_ready_ms)} | "
            f"{segment.was_cut_off} | {reference} |"
        )

    lines.extend(
        [
            "",
            "## Segment Timing",
            "",
            "| Segment | Start ms | End ms | Gap from previous end ms | Starts before previous commit |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for segment in segments:
        lines.append(
            f"| {segment.id} | {_fmt_optional_float(segment.capture_start_ms)} | "
            f"{_fmt_optional_float(segment.capture_end_ms)} | "
            f"{_fmt_optional_float(segment.gap_from_previous_end_ms)} | "
            f"{segment.starts_before_previous_commit} |"
        )

    lines.extend(
        [
            "",
            "## Model Summary",
            "",
            "| Provider | Model | Segments | Mean WER | Mean WER no cutoffs | Mean CER | Mean CER no cutoffs | Median ASR ms | Median VAD-end -> transcript ms | Median RTFx | Cutoff reports |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summarize_results(results):
        rtf = item["median_realtime_factor"]
        lines.append(
            "| {provider} | {model} | {segments} | {wer} | {wer_uncut} | {cer} | {cer_uncut} | {ms:.1f} | {ready_ms:.1f} | {rtf} | {cutoffs} |".format(
                provider=item["provider"],
                model=item["model"],
                segments=item["segments"],
                wer=_fmt_optional_percent(item["mean_wer"]),
                wer_uncut=_fmt_optional_percent(item["mean_wer_without_cutoffs"]),
                cer=_fmt_optional_percent(item["mean_cer"]),
                cer_uncut=_fmt_optional_percent(item["mean_cer_without_cutoffs"]),
                ms=item["median_ms"],
                ready_ms=item["median_vad_end_to_transcript_ready_ms"],
                rtf=f"{rtf:.1f}x" if rtf is not None else "n/a",
                cutoffs=item["cutoff_reports"],
            )
        )

    if warmups:
        lines.extend(
            [
                "",
                "## Warmup Runs",
                "",
                "| Provider | Model | Segment | Run | ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for warmup in warmups:
            lines.append(
                f"| {warmup.provider} | {warmup.model} | {warmup.segment_id} | "
                f"{warmup.run_index} | {warmup.transcribe_ms:.1f} |"
            )

    lines.extend(
        [
            "",
            "## Per Segment",
            "",
            "| Segment | Model | Lang | WER | CER | ASR ms | VAD-end -> transcript ms | RTFx | Hypothesis |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        rtf = result.realtime_factor
        hypothesis = result.hypothesis.replace("|", "\\|")
        lines.append(
            f"| {result.segment_id} | {result.model} | {result.language or ''} | "
            f"{_fmt_optional_percent(result.wer)} | {_fmt_optional_percent(result.cer)} | "
            f"{result.transcribe_ms:.1f} | "
            f"{result.vad_end_to_transcript_ready_ms:.1f} | "
            f"{f'{rtf:.1f}x' if rtf is not None else 'n/a'} | {hypothesis} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audio_config(config: dict, args: argparse.Namespace) -> tuple[dict, int]:
    pipeline_config = config.get("pipeline", {})
    audio_source = config.get("audio", {})
    default_misses = pipeline_config.get("vad_required_misses", 30)
    commit_delay_ms = int(
        audio_source.get("speech_commit_delay_ms", 700)
        if args.speech_commit_delay_ms is None
        else args.speech_commit_delay_ms
    )

    def from_arg_or_config(arg_value, key: str, default):
        return audio_source.get(key, default) if arg_value is None else arg_value

    return (
        {
            "sample_rate": int(args.sample_rate),
            "channels": 1,
            "chunk_duration_ms": 32,
            "device": args.input_device,
            "start_muted": False,
            "vad_prob_threshold": float(from_arg_or_config(args.vad_prob_threshold, "vad_prob_threshold", 0.5)),
            "vad_db_threshold": float(from_arg_or_config(args.vad_db_threshold, "vad_db_threshold", -50)),
            "vad_required_hits": int(from_arg_or_config(args.vad_required_hits, "vad_required_hits", 3)),
            "vad_required_misses": int(
                from_arg_or_config(args.vad_required_misses, "vad_required_misses", default_misses)
            ),
            "vad_min_speech_ms": int(from_arg_or_config(args.vad_min_speech_ms, "vad_min_speech_ms", 450)),
            "vad_min_voiced_ms": int(from_arg_or_config(args.vad_min_voiced_ms, "vad_min_voiced_ms", 180)),
        },
        commit_delay_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark live microphone ASR and turn-taking.")
    parser.add_argument("--models", default="small,large-v3-turbo", help="Comma-separated Whisper models")
    parser.add_argument("--segments", default=5, type=int, help="Number of live utterances to capture")
    parser.add_argument("--max-duration-sec", default=120, type=int, help="Safety timeout for live capture")
    parser.add_argument("--language", default=None, help="Force one language for all ASR runs")
    parser.add_argument("--device", default="cuda", help="Whisper device: cuda, cpu, or auto")
    parser.add_argument("--compute-type", default="float16", help="Whisper compute type")
    parser.add_argument("--beam-size", default=1, type=int, help="Whisper beam size")
    parser.add_argument("--warmup-runs", default=1, type=int, help="Discarded ASR runs per model")
    parser.add_argument("--sample-rate", default=16000, type=int)
    parser.add_argument("--input-device", default=None, type=int)
    parser.add_argument("--vad-prob-threshold", default=None, type=float)
    parser.add_argument("--vad-db-threshold", default=None, type=float)
    parser.add_argument("--vad-required-hits", default=None, type=int)
    parser.add_argument("--vad-required-misses", default=None, type=int)
    parser.add_argument("--vad-min-speech-ms", default=None, type=int)
    parser.add_argument("--vad-min-voiced-ms", default=None, type=int)
    parser.add_argument("--speech-commit-delay-ms", default=None, type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/live_voice"))
    parser.add_argument("--json-output", type=Path, default=Path("data/benchmarks/live_voice.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("data/benchmarks/live_voice.md"))
    parser.add_argument("--no-annotate", action="store_true", help="Skip expected transcript prompts")
    return parser.parse_args()


def main() -> int:
    from src.utils.config_loader import load_yaml_config

    configure_stdio()
    args = parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise SystemExit("No models selected")

    config = load_yaml_config(PROJECT_ROOT / "config" / "config.yaml")
    audio_config, commit_delay_ms = build_audio_config(config, args)
    segments = capture_live_segments(
        output_dir=args.output_dir,
        target_segments=max(1, args.segments),
        max_duration_sec=max(1, args.max_duration_sec),
        commit_delay_ms=commit_delay_ms,
        audio_config=audio_config,
    )
    if not segments:
        raise SystemExit("No live speech segments captured")
    if not args.no_annotate:
        segments = annotate_segments(segments)

    results: list[BenchmarkResult] = []
    warmups: list[WarmupResult] = []
    for model in models:
        asr = create_whisper_provider(model, args.device, args.compute_type, args.beam_size)
        for warmup_index in range(1, max(0, args.warmup_runs) + 1):
            warmup = run_warmup(
                asr,
                model=model,
                segment=segments[0],
                forced_language=args.language,
                run_index=warmup_index,
            )
            warmups.append(warmup)
            print(
                f"{model} warmup {warmup_index}: "
                f"segment={warmup.segment_id} ms={warmup.transcribe_ms:.1f}"
            )
        for segment in segments:
            result = run_whisper_case(
                asr,
                model=model,
                segment=segment,
                forced_language=args.language,
            )
            results.append(result)
            print(
                f"{model} {segment.id}: "
                f"WER={_fmt_optional_percent(result.wer)} "
                f"CER={_fmt_optional_percent(result.cer)} "
                f"asr_ms={result.transcribe_ms:.1f} "
                f"ready_ms={result.vad_end_to_transcript_ready_ms:.1f} "
                f"text={result.hypothesis!r}"
            )

    write_json_report(args.json_output, segments, results, warmups)
    write_markdown_report(args.markdown_output, segments, results, warmups)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
