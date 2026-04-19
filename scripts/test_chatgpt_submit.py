import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import build_stream_prompt, interview_photo_path
from browser import submit_to_chatgpt
from logging_config import configure_logging


def main() -> None:
    """Read a transcript segment and submit it to ChatGPT through browser automation."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Submit a stdin transcript segment to ChatGPT.")
    parser.add_argument(
        "--photo-mode",
        choices=("static", "test", "live"),
        default="static",
        help=(
            "Upload a fixed interview photo with the prompt: "
            "static uses /Users/flora/interview/static.jpg; "
            "test uses /Users/flora/interview/test.jpg; "
            "live uses /Users/flora/interview/live.jpg. Default: static."
        ),
    )
    args = parser.parse_args()

    prompt = read_prompt()
    photo_path = interview_photo_path(args.photo_mode)
    submit_to_chatgpt(
        build_stream_prompt(
            prompt,
            include_mode_prompt=True,
            include_photo_context=photo_path is not None,
        ),
        photo_path=photo_path,
    )


def read_prompt() -> str:
    """Read a prompt from stdin or interactively from the terminal."""
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("Enter the transcript segment to send to ChatGPT:\n> ").strip()


if __name__ == "__main__":
    main()
