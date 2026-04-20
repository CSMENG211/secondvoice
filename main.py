import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app import RuntimeOptions, run
from audio import AUDIO_ENHANCEMENT_OFF, AUDIO_ENHANCEMENT_SPECTRAL_GATE
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
        choices=("none", "test", "live"),
        default="none",
        help=(
            "Control interview photo capture and upload: "
            "none disables all photo capture and upload; "
            "test captures to /Users/flora/interview/test.jpg; "
            "live captures to /Users/flora/interview/live.jpg. Default: none."
        ),
    )
    parser.add_argument(
        "--audio-enhancement",
        choices=(AUDIO_ENHANCEMENT_SPECTRAL_GATE, AUDIO_ENHANCEMENT_OFF),
        default=AUDIO_ENHANCEMENT_SPECTRAL_GATE,
        help=(
            "Preprocess audio before transcription. "
            "spectral-gate reduces steady and transient background noise; "
            "off transcribes raw captured audio. Default: spectral-gate."
        ),
    )
    args = parser.parse_args()
    return RuntimeOptions(
        ask_chatgpt=args.ask_chatgpt,
        enroll_me=args.enroll,
        photo_mode=args.photo_mode,
        audio_enhancement=args.audio_enhancement,
    )


if __name__ == "__main__":
    main()
