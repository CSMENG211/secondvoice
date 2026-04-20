import argparse
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio import AudioEnhancementConfig, enhance_wav  # noqa: E402
from audio.enhancement import load_wav_as_float32, write_float32_as_wav  # noqa: E402
from speech import create_transcriber  # noqa: E402
from speech.constants import (  # noqa: E402
    DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
    DEFAULT_FINAL_TRANSCRIPTION_MODEL,
)

DEFAULT_TEXT = (
    "I would solve two sum by iterating through the array once. "
    "For each number, I compute the complement and check whether it already "
    "exists in a hash map. If it does, I return the stored index and the "
    "current index. Otherwise I store the current number and index."
)
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    path: Path
    enhancement_seconds: float | None = None


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="secondvoice-noise-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    text = args.text
    clean_path = output_dir / "clean.wav"
    synthesize_speech(text, clean_path, args.voice)

    cases = [BenchmarkCase("clean", clean_path)]
    for noise_kind in args.noise_kind:
        for snr_db in args.snr_db:
            noisy_path = output_dir / f"{noise_kind}-snr-{format_snr(snr_db)}.wav"
            enhanced_path = output_dir / f"{noise_kind}-snr-{format_snr(snr_db)}-enhanced.wav"
            make_noisy_wav(clean_path, noisy_path, noise_kind, snr_db, args.seed)
            started = time.perf_counter()
            enhance_wav(noisy_path, enhanced_path, AudioEnhancementConfig())
            enhancement_seconds = time.perf_counter() - started
            cases.append(BenchmarkCase(f"{noise_kind} raw {snr_db:g} dB", noisy_path))
            cases.append(
                BenchmarkCase(
                    f"{noise_kind} enhanced {snr_db:g} dB",
                    enhanced_path,
                    enhancement_seconds,
                )
            )

    print(f"Benchmark WAVs: {output_dir}")
    print(f"Reference text:\n{text}\n")

    if args.no_transcribe:
        for case in cases:
            print(f"{case.label}: {case.path}")
        return

    transcriber = create_transcriber(args.backend, args.model)
    for case in cases:
        started = time.perf_counter()
        transcript = transcriber.transcribe(case.path)
        transcription_seconds = time.perf_counter() - started
        error_rate = word_error_rate(text, transcript)
        print_result(case, transcript, error_rate, transcription_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark final-transcription audio enhancement with macOS say and "
            "synthetic background noise."
        )
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Speech text passed to macOS say.",
    )
    parser.add_argument(
        "--voice",
        default="Samantha",
        help="macOS say voice. Default: Samantha.",
    )
    parser.add_argument(
        "--noise-kind",
        action="append",
        choices=("keyboard", "fan", "mixed"),
        default=[],
        help="Noise type to mix in. Can be passed more than once.",
    )
    parser.add_argument(
        "--snr-db",
        action="append",
        type=float,
        default=[],
        help="Signal-to-noise ratio in dB. Can be passed more than once.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for repeatable noise. Default: 7.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated WAVs. Defaults to a temp directory.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
        help=f"Transcription backend. Default: {DEFAULT_FINAL_TRANSCRIPTION_BACKEND}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_FINAL_TRANSCRIPTION_MODEL,
        help=f"Backend model name. Default: {DEFAULT_FINAL_TRANSCRIPTION_MODEL}.",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Only generate clean/noisy/enhanced WAV files.",
    )
    args = parser.parse_args()
    if not args.noise_kind:
        args.noise_kind = ["keyboard", "fan", "mixed"]
    if not args.snr_db:
        args.snr_db = [15.0, 5.0]
    return args


def synthesize_speech(text: str, output_path: Path, voice: str) -> None:
    """Use macOS say, then convert to the app's mono 16 kHz int16 WAV format."""
    with tempfile.TemporaryDirectory(prefix="secondvoice-say-") as temp_dir:
        aiff_path = Path(temp_dir) / "speech.aiff"
        subprocess.run(
            ["say", "-v", voice, "-o", str(aiff_path), text],
            check=True,
        )
        subprocess.run(
            [
                "afconvert",
                str(aiff_path),
                str(output_path),
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                "-c",
                "1",
            ],
            check=True,
        )


def make_noisy_wav(
    clean_path: Path,
    output_path: Path,
    noise_kind: str,
    snr_db: float,
    seed: int,
) -> None:
    clean, sample_rate = load_wav_as_float32(clean_path)
    rng = np.random.default_rng(seed + stable_noise_offset(noise_kind, snr_db))
    noise = generate_noise(noise_kind, len(clean), sample_rate, rng)
    noisy = clean + scale_noise_to_snr(clean, noise, snr_db)
    write_float32_as_wav(output_path, noisy)


def generate_noise(
    noise_kind: str,
    sample_count: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_kind == "keyboard":
        return keyboard_noise(sample_count, sample_rate, rng)
    if noise_kind == "fan":
        return fan_noise(sample_count, sample_rate, rng)
    if noise_kind == "mixed":
        return keyboard_noise(sample_count, sample_rate, rng) + fan_noise(
            sample_count,
            sample_rate,
            rng,
        )
    raise ValueError(f"Unsupported noise kind: {noise_kind!r}")


def keyboard_noise(
    sample_count: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate short click bursts with irregular timing."""
    noise = np.zeros(sample_count, dtype=np.float32)
    click_count = max(1, int(sample_count / sample_rate * 7))
    click_starts = rng.integers(0, sample_count, size=click_count)
    click_length = max(1, int(sample_rate * 0.012))
    envelope = np.exp(-np.linspace(0.0, 6.0, click_length)).astype(np.float32)
    for start in click_starts:
        end = min(sample_count, start + click_length)
        burst = rng.normal(0.0, 1.0, end - start).astype(np.float32)
        noise[start:end] += burst * envelope[: end - start]
    return noise


def fan_noise(
    sample_count: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate broadband room/fan noise with a low-frequency hum component."""
    white = rng.normal(0.0, 1.0, sample_count).astype(np.float32)
    smoothed = np.convolve(white, np.ones(64, dtype=np.float32) / 64, mode="same")
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    hum = 0.4 * np.sin(2 * np.pi * 120 * t)
    return smoothed + hum.astype(np.float32)


def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_rms = rms(clean)
    noise_rms = rms(noise)
    if speech_rms == 0.0 or noise_rms == 0.0:
        return noise
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    return noise * (target_noise_rms / noise_rms)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return edit_distance(reference_words, hypothesis_words) / len(reference_words)


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for index, left_word in enumerate(left, start=1):
        current = [index]
        for right_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def normalize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def stable_noise_offset(noise_kind: str, snr_db: float) -> int:
    return sum(ord(char) for char in noise_kind) + int(snr_db * 10)


def format_snr(snr_db: float) -> str:
    return f"{snr_db:g}".replace("-", "neg").replace(".", "p")


def print_result(
    case: BenchmarkCase,
    transcript: str,
    error_rate: float,
    transcription_seconds: float,
) -> None:
    enhancement = (
        f" enhancement_seconds={case.enhancement_seconds:.2f}"
        if case.enhancement_seconds is not None
        else ""
    )
    print(f"== {case.label} ==")
    print(
        f"path={case.path} wer={error_rate:.3f} "
        f"transcription_seconds={transcription_seconds:.2f}{enhancement}"
    )
    print(transcript.strip() or "(empty transcript)")
    print()


if __name__ == "__main__":
    main()
