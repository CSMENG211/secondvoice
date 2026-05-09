import queue
import tempfile
import threading
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from loguru import logger

from audio import (
    CompletedStreamSegment,
    is_repetitive_transcript,
    stream_utterance_segments,
)
from audio.constants import (
    DEFAULT_SILENCE_THRESHOLD,
    STREAM_HARD_SILENCE_SECONDS,
    STREAM_TRANSCRIPT_AGREEMENT_COUNT,
)
from gpt import submit_to_chatgpt
from speech.constants import (
    DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
    DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL,
)
from gpt import ensure_context_templates, load_round_context
from gpt.prompts import BehaviorState, build_stream_prompt, parse_actual_story_id
from vision import (
    PhotoMode,
    PhotoUploadTracker,
    interview_photo_path,
    next_photo_upload,
    photo_capture_settings,
    start_photo_timer,
)
from speech import (
    OllamaSemanticEndpointDetector,
    Transcriber,
    create_transcriber,
    model_path_for_run,
)


InterviewRound = Literal["coding", "system-design", "behavior", "offer-negotiation"]


@dataclass(frozen=True)
class RuntimeOptions:
    """Options selected by the command-line interface."""

    ask_chatgpt: bool = True
    photo_mode: PhotoMode = "none"
    round_type: InterviewRound = "coding"


def run(options: RuntimeOptions) -> None:
    """Run continuous stream mode."""
    stream_loop(options)


def stream_loop(options: RuntimeOptions) -> None:
    """Continuously capture interview segments and submit each segment for feedback."""
    is_first_submission = True
    ensure_context_templates()
    round_context = load_round_context(options.round_type)
    behavior_state = BehaviorState()

    print_stream_mode_banner(options)

    with tempfile.TemporaryDirectory(prefix="secondvoice-eval-") as temp_dir:
        segment_queue: queue.Queue[CompletedStreamSegment | Exception] = queue.Queue()
        stop_event = threading.Event()
        semantic_endpoint_detector = OllamaSemanticEndpointDetector()
        stream_transcription_model = model_path_for_run(
            DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
            DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL,
            use_local_cache=True,
        )
        stream_transcriber = create_transcriber(
            DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
            stream_transcription_model,
        )
        recorder = start_stream_recorder(
            Path(temp_dir),
            segment_queue,
            stop_event,
            stream_transcriber,
            semantic_endpoint_detector,
        )
        photo_timer = (
            start_photo_timer(stop_event, options.photo_mode)
            if options.photo_mode != "none"
            else None
        )

        try:
            photo_tracker = PhotoUploadTracker()
            while True:
                segment = next_stream_segment(segment_queue)
                if segment is None:
                    continue

                submitted = process_stream_segment(
                    segment,
                    options,
                    include_mode_prompt=is_first_submission,
                    photo_tracker=photo_tracker,
                    round_context=round_context,
                    behavior_state=behavior_state,
                )
                if submitted:
                    is_first_submission = False
        except KeyboardInterrupt:
            logger.info("Stopping stream mode...")
        finally:
            stop_event.set()
            recorder.join()
            if photo_timer is not None:
                photo_timer.join()
            logger.info("Stopped stream mode.")


def start_stream_recorder(
    output_dir: Path,
    segment_queue: queue.Queue[CompletedStreamSegment | Exception],
    stop_event: threading.Event,
    transcriber: Transcriber,
    semantic_endpoint_detector: OllamaSemanticEndpointDetector,
) -> threading.Thread:
    """Start the background recorder that feeds completed segments into a queue."""
    recorder = threading.Thread(
        target=stream_utterance_segments,
        args=(
            output_dir,
            segment_queue,
            stop_event,
            transcriber,
            STREAM_HARD_SILENCE_SECONDS,
            DEFAULT_SILENCE_THRESHOLD,
            semantic_endpoint_detector.classify_transcript,
        ),
    )
    recorder.start()
    return recorder


def next_stream_segment(
    segment_queue: queue.Queue[CompletedStreamSegment | Exception],
) -> CompletedStreamSegment | None:
    """Return the next completed stream segment, or None while waiting."""
    try:
        item = segment_queue.get(timeout=0.2)
    except queue.Empty:
        return None

    if isinstance(item, Exception):
        raise item

    return item


def process_stream_segment(
    segment: CompletedStreamSegment,
    options: RuntimeOptions,
    include_mode_prompt: bool,
    photo_tracker: PhotoUploadTracker,
    round_context: str,
    behavior_state: BehaviorState,
) -> bool:
    """Transcribe one stream segment and optionally submit it for feedback."""
    audio_path = segment.path
    try:
        transcript = segment.transcript
    finally:
        audio_path.unlink(missing_ok=True)
    print_transcript(transcript)

    if not transcript:
        logger.info("No speech detected. Listening again.")
        return False

    if is_repetitive_transcript(transcript):
        logger.info("Skipping transcript because it appears repetitive or garbled.")
        return False

    if options.ask_chatgpt:
        logger.info(
            "Submitting to ChatGPT because segment ended by {}.",
            segment.completion_reason,
        )
        photo_path, photo_signature = next_photo_upload(options.photo_mode, photo_tracker)
        submitted_to_chatgpt, response_text = submit_to_chatgpt(
            build_stream_prompt(
                transcript,
                include_mode_prompt,
                include_photo_context=photo_path is not None,
                round_type=options.round_type,
                round_context=round_context,
                behavior_state=behavior_state,
            ),
            photo_path=photo_path,
        )
        if options.round_type == "behavior":
            actual_story_id = parse_actual_story_id(response_text or "")
            if actual_story_id is not None:
                behavior_state.used_story_ids.add(actual_story_id)
        if submitted_to_chatgpt and photo_signature is not None:
            photo_tracker.last_signature = photo_signature
    logger.info("")
    return options.ask_chatgpt


def print_stream_mode_banner(options: RuntimeOptions) -> None:
    """Print the active stream-mode trigger settings."""
    logger.info(
        "Audio segments end after semantic completion or {:g}s hard silence.",
        STREAM_HARD_SILENCE_SECONDS,
    )
    logger.debug(
        "Semantic endpoint check: stabilized transcript buffer via local Ollama qwen2.5:1.5b"
    )
    logger.info("Streaming transcription: {}", stream_transcription_label())
    logger.info(
        "n agreement = {}",
        STREAM_TRANSCRIPT_AGREEMENT_COUNT,
    )
    if options.photo_mode != "none":
        logger.info(
            "Photo upload: {} mode; using {}",
            options.photo_mode,
            interview_photo_path(options.photo_mode),
        )
        _, initial_seconds, interval_seconds = photo_capture_settings(options.photo_mode)
        logger.info(
            "Photo capture: first after {:g} min; then every {:g} min",
            initial_seconds / 60,
            interval_seconds / 60,
        )
    else:
        logger.info("Photo upload: disabled")
        logger.debug("Photo capture: disabled")
    if options.ask_chatgpt:
        logger.info("ChatGPT submission: enabled")
        logger.info("Round mode: {}", options.round_type)
    else:
        logger.info("ChatGPT submission: disabled")


def stream_transcription_label() -> str:
    """Return a compact transcription backend+model label for logs."""
    model = DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL.lower()
    if "small" in model:
        size = "small"
    elif "tiny" in model:
        size = "tiny"
    else:
        size = DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL
    return f"{DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND} {size}"


def print_transcript(transcript: str) -> None:
    """Print the transcript from the most recent captured utterance."""
    logger.info("Heard:\n{}", transcript.strip() or "(No speech detected.)")
