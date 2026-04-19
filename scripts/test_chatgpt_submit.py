import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import build_stream_prompt
from browser import submit_to_chatgpt
from logging_config import configure_logging


def main() -> None:
    """Read a transcript segment and submit it to ChatGPT through browser automation."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Submit a stdin transcript segment to ChatGPT.")
    parser.add_argument(
        "--browser",
        choices=("persistent", "cdp"),
        default="cdp",
        help="Browser automation mode. Default: cdp",
    )
    args = parser.parse_args()

    prompt = read_prompt()
    submit_to_chatgpt(
        build_stream_prompt(prompt, include_mode_prompt=True),
        browser_mode=args.browser_mode,
    )


def read_prompt() -> str:
    """Read a prompt from stdin or interactively from the terminal."""
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("Enter the transcript segment to send to ChatGPT:\n> ").strip()


if __name__ == "__main__":
    main()
