import argparse
import queue
import sys
import time
from pathlib import Path

import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio import AudioEnhancementConfig, enhance_wav  # noqa: E402
from audio.constants import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE  # noqa: E402
from audio.levels import audio_blocksize  # noqa: E402
from audio.wav import write_wav_file  # noqa: E402
from speech import create_transcriber  # noqa: E402
from speech.constants import (  # noqa: E402
    DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
    DEFAULT_FINAL_TRANSCRIPTION_MODEL,
)

DEFAULT_READ_TEXT = (
    "I would solve two sum by iterating through the array once. "
    "For each number, I compute the complement and check whether it already "
    "exists in a hash map. If it does, I return the stored index and the "
    "current index. Otherwise I store the current number and index."
)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{args.label}-raw.wav"
    enhanced_path = output_dir / f"{args.label}-enhanced.wav"

    print(f"Output directory: {output_dir}")
    if args.read_prompt:
        print("\nRead this while the capture is running:")
        print(args.text)
    print(
        f"\nRecording {args.seconds:g}s from the current macOS input device. "
        "Start speaking/typing after the countdown."
    )

    chunks = record_chunks(args.seconds, args.countdown)
    write_wav_file(raw_path, chunks)
    print(f"Raw WAV: {raw_path}")

    started = time.perf_counter()
    transcription_path = enhance_wav(
        raw_path,
        enhanced_path,
        AudioEnhancementConfig(enabled=args.audio_enhancement),
    )
    enhancement_seconds = time.perf_counter() - started
    if transcription_path == enhanced_path:
        print(f"Enhanced WAV: {enhanced_path} ({enhancement_seconds:.2f}s)")
    else:
        print("Enhanced WAV: disabled or unavailable; using raw audio.")

    if args.no_transcribe:
        return

    transcriber = create_transcriber(args.backend, args.model)
    transcribe_and_print("raw", raw_path, transcriber)
    if transcription_path != raw_path:
        transcribe_and_print("enhanced", transcription_path, transcriber)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a fixed live microphone probe and compare raw/enhanced ASR."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label used in output WAV names, such as standard or voice-isolation.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=12.0,
        help="Capture duration. Default: 12.",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown before recording starts. Default: 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/secondvoice-live-probe"),
        help="Directory for output WAV files.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_READ_TEXT,
        help="Prompt text printed for repeatable read-aloud probes.",
    )
    parser.add_argument(
        "--read-prompt",
        action="store_true",
        help="Print a fixed read-aloud prompt before recording.",
    )
    parser.add_argument(
        "--no-audio-enhancement",
        action="store_false",
        dest="audio_enhancement",
        default=True,
        help="Do not create an enhanced WAV.",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Only record WAV files; skip Whisper transcription.",
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
    return parser.parse_args()


def record_chunks(seconds: float, countdown: int) -> list[bytes]:
    blocksize = audio_blocksize()
    blocks_needed = max(1, round(seconds / (blocksize / AUDIO_SAMPLE_RATE)))
    chunks: list[bytes] = []
    overflow_count = 0
    warnings: queue.SimpleQueue[str] = queue.SimpleQueue()

    for remaining in range(countdown, 0, -1):
        print(f"{remaining}...")
        time.sleep(1)
    print("Recording now.")

    with sd.RawInputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="int16",
        blocksize=blocksize,
    ) as stream:
        for _ in range(blocks_needed):
            data, overflowed = stream.read(blocksize)
            chunks.append(bytes(data))
            if overflowed:
                overflow_count += 1

    print("Recording done.")
    if overflow_count:
        warnings.put(f"Input overflowed {overflow_count} times.")
    while not warnings.empty():
        print(f"Warning: {warnings.get()}")
    return chunks


def transcribe_and_print(label: str, audio_path: Path, transcriber) -> None:
    started = time.perf_counter()
    transcript = transcriber.transcribe(audio_path)
    duration = time.perf_counter() - started
    print(f"\n== {label} transcript ({duration:.2f}s) ==")
    print(transcript.strip() or "(empty transcript)")


if __name__ == "__main__":
    main()
