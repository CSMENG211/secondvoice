import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app import RuntimeOptions, run
from logging_config import configure_logging


def main() -> None:
    """Parse command-line options and start SecondVoice."""
    configure_logging()
    run(parse_args())


def parse_args() -> RuntimeOptions:
    """Parse CLI flags into runtime options."""
    parser = argparse.ArgumentParser(
        description="Stream mock-interview audio segments to ChatGPT."
    )
    parser.add_argument(
        "--no-ask",
        action="store_false",
        dest="ask_chatgpt",
        default=True,
        help="Transcribe without submitting the prompt to ChatGPT.",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Record prompted interviewee voice samples and save the voice profile.",
    )
    parser.add_argument(
        "--photo-mode",
        choices=("static", "test", "live"),
        default="static",
        help=(
            "Upload a fixed interview photo with each ChatGPT prompt: "
            "static uses /Users/flora/interview/static.jpg without capture; "
            "test captures to /Users/flora/interview/test.jpg; "
            "live captures to /Users/flora/interview/live.jpg. Default: static."
        ),
    )
    args = parser.parse_args()
    return RuntimeOptions(
        ask_chatgpt=args.ask_chatgpt,
        enroll_me=args.enroll,
        photo_mode=args.photo_mode,
    )


if __name__ == "__main__":
    main()
