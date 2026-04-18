import tempfile
from dataclasses import dataclass
from pathlib import Path

from audio import capture_utterance, record_until_enter
from browser import BrowserMode, submit_to_chatgpt
from constants import (
    ANSWER_MODE_PROMPTS,
    AnswerMode,
    DEFAULT_ANSWER_MODE,
    DEFAULT_MAX_RECORD_SECONDS,
    DEFAULT_QUESTION_START_PATTERN,
    DEFAULT_QUESTION_TRIGGER_MODE,
    DEFAULT_SILENCE_SECONDS,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_TRANSCRIPTION_MODEL,
)
from question_triggers import extract_question_prompt
from transcription import LocalTranscriber, transcribe


@dataclass(frozen=True)
class RuntimeOptions:
    """Options selected by the command-line interface."""

    ask_chatgpt: bool = True
    browser_mode: BrowserMode = "cdp"
    listen: bool = True
    answer_mode: AnswerMode = DEFAULT_ANSWER_MODE


def run(options: RuntimeOptions) -> None:
    """Run either continuous listen mode or a one-shot recording."""
    if options.listen:
        listen_loop(options)
        return

    transcript = record_and_transcribe_once()

    print("\nTranscript:")
    print(transcript.strip() or "(No speech detected.)")

    if options.ask_chatgpt:
        submit_prompt(transcript, options, include_mode_prompt=True)


def record_and_transcribe_once() -> str:
    """Record one manually-delimited utterance and return its transcript."""
    print("Press ENTER to start recording.")
    input()

    with tempfile.TemporaryDirectory(prefix="secondvoice-") as temp_dir:
        audio_path = Path(temp_dir) / "recording.wav"
        record_until_enter(audio_path)

        print(
            f"Transcribing locally with faster-whisper ({DEFAULT_TRANSCRIPTION_MODEL})...",
            flush=True,
        )
        return transcribe(audio_path, DEFAULT_TRANSCRIPTION_MODEL)


def listen_loop(options: RuntimeOptions) -> None:
    """Continuously capture utterances, extract prompts, and optionally submit them."""
    transcriber = LocalTranscriber(DEFAULT_TRANSCRIPTION_MODEL)
    is_first_submission = True

    print_listen_mode_banner()

    try:
        while True:
            transcript = capture_and_transcribe_utterance(transcriber)
            print_transcript(transcript)

            prompt = extract_question_prompt(
                transcript,
                DEFAULT_QUESTION_START_PATTERN,
                DEFAULT_QUESTION_TRIGGER_MODE,
            )
            if not should_handle_prompt(prompt):
                continue

            print("\nTriggered prompt:")
            print(prompt)

            if options.ask_chatgpt:
                submit_prompt(
                    prompt,
                    options,
                    include_mode_prompt=is_first_submission,
                )
                is_first_submission = False
            print()
    except KeyboardInterrupt:
        print("\nStopped listening.")


def capture_and_transcribe_utterance(transcriber: LocalTranscriber) -> str:
    """Capture one speech segment using silence detection and transcribe it."""
    with tempfile.TemporaryDirectory(prefix="secondvoice-") as temp_dir:
        audio_path = Path(temp_dir) / "utterance.wav"
        capture_utterance(
            audio_path,
            silence_seconds=DEFAULT_SILENCE_SECONDS,
            silence_threshold=DEFAULT_SILENCE_THRESHOLD,
            max_record_seconds=DEFAULT_MAX_RECORD_SECONDS,
        )
        return transcriber.transcribe(audio_path)


def submit_prompt(prompt: str, options: RuntimeOptions, include_mode_prompt: bool) -> None:
    """Submit a prompt to ChatGPT, prepending mode instructions once per run."""
    submit_to_chatgpt(
        build_chatgpt_prompt(prompt, options.answer_mode, include_mode_prompt),
        browser_mode=options.browser_mode,
    )


def build_chatgpt_prompt(
    prompt: str,
    answer_mode: AnswerMode,
    include_mode_prompt: bool,
) -> str:
    """Return the ChatGPT prompt with optional first-message mode instructions."""
    if not include_mode_prompt:
        return prompt

    return f"{ANSWER_MODE_PROMPTS[answer_mode]}\n\nQuestion:\n{prompt}"


def print_listen_mode_banner() -> None:
    """Print the active listen-mode trigger settings."""
    print("Listen mode is active.")
    print("Audio start trigger: speech begins")
    print(f"Audio stop trigger: {DEFAULT_SILENCE_SECONDS:g}s of silence")
    print(f"Question trigger mode: {DEFAULT_QUESTION_TRIGGER_MODE}")
    if DEFAULT_QUESTION_TRIGGER_MODE in ("phrase", "smart"):
        print(f"Question start pattern: {DEFAULT_QUESTION_START_PATTERN!r}")
    print("Press Ctrl+C to stop.")


def print_transcript(transcript: str) -> None:
    """Print the transcript from the most recent captured utterance."""
    print("\nHeard:")
    print(transcript.strip() or "(No speech detected.)")


def should_handle_prompt(prompt: str | None) -> bool:
    """Return whether a detected prompt is non-empty and ready for handling."""
    if prompt is None:
        print("No interview prompt detected. Listening again.\n")
        return False

    if not prompt:
        print("Prompt detector fired, but the prompt was empty. Listening again.\n")
        return False

    return True
