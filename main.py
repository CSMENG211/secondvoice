import argparse
import os
import sys
from pathlib import Path

# Keep third-party model/download progress bars out of terminal logs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("DISABLE_TQDM", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app import RuntimeOptions, run
from logging_config import configure_logging
from preflight import check_runtime_dependencies


def main() -> None:
    """Parse command-line options and start SecondVoice."""
    configure_logging()
    options = parse_args()
    check_runtime_dependencies(options)
    run(options)


def parse_args() -> RuntimeOptions:
    """Parse CLI flags into runtime options."""
    parser = argparse.ArgumentParser(
        description="Stream mock-interview audio segments to ChatGPT."
    )
    parser.add_argument(
        "--round",
        choices=("coding", "design", "behavior", "offer", "qa"),
        default="coding",
        help=(
            "Interview round mode: coding, design, behavior, offer, or qa. "
            "Default: coding."
        ),
    )
    parser.add_argument(
        "--no-ask",
        action="store_false",
        dest="ask_chatgpt",
        default=True,
        help="Transcribe without submitting the prompt to ChatGPT.",
    )
    parser.add_argument(
        "--photo-mode",
        choices=("none", "test", "live"),
        default="none",
        help=(
            "Control interview photo capture and upload: "
            "none disables all photo capture and upload; "
            "test captures to /Users/flora/interview/test.jpg; "
            "live captures to /Users/flora/interview/live.jpg. Default: none."
        ),
    )
    args = parser.parse_args()
    return RuntimeOptions(
        ask_chatgpt=args.ask_chatgpt,
        photo_mode=args.photo_mode,
        round_type=args.round,
    )


if __name__ == "__main__":
    main()
