import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constants import (  # noqa: E402
    DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
    DEFAULT_FINAL_TRANSCRIPTION_MODEL,
)
from transcription import create_transcriber  # noqa: E402


def main() -> None:
    """Benchmark a transcription backend/model on one or more WAV files."""
    args = parse_args()
    transcriber = create_transcriber(args.backend, args.model)

    for audio_path in args.audio_paths:
        started = time.perf_counter()
        transcript = transcriber.transcribe(audio_path)
        duration = time.perf_counter() - started
        print(f"\n== {audio_path} ==")
        print(f"backend={args.backend} model={args.model} seconds={duration:.2f}")
        print(transcript)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one local transcription backend and model."
    )
    parser.add_argument(
        "audio_paths",
        nargs="+",
        type=Path,
        help="WAV files to transcribe.",
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


if __name__ == "__main__":
    main()
