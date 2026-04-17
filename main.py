import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from offergpt.audio import list_microphones, record_until_enter
from offergpt.browser import submit_to_chatgpt
from offergpt.transcription import DEFAULT_MODEL, transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe it.")
    parser.add_argument("--list-mics", action="store_true", help="List available input devices.")
    parser.add_argument("--device", type=int, help="Input device index from --list-mics.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local faster-whisper model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--ask-chatgpt",
        action="store_true",
        help="Open ChatGPT in a browser, paste the transcript, and press Enter.",
    )
    args = parser.parse_args()

    if args.list_mics:
        list_microphones()
        return

    print("Press ENTER to start recording.")
    input()

    with tempfile.TemporaryDirectory(prefix="offergpt-") as temp_dir:
        audio_path = Path(temp_dir) / "recording.wav"
        record_until_enter(audio_path, args.device)

        print(f"Transcribing locally with faster-whisper ({args.model})...", flush=True)
        transcript = transcribe(audio_path, args.model)

    print("\nTranscript:")
    print(transcript.strip() or "(No speech detected.)")

    if args.ask_chatgpt:
        submit_to_chatgpt(transcript)


if __name__ == "__main__":
    main()
