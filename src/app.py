import queue
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from audio import (
    AUDIO_ENHANCEMENT_OFF,
    AUDIO_ENHANCEMENT_SPECTRAL_GATE,
    AudioEnhancementConfig,
    CompletedStreamSegment,
    enhance_wav,
    is_repetitive_transcript,
    stream_utterance_segments,
    trim_repetitive_transcript_suffix,
)
from audio.constants import (
    DEFAULT_SILENCE_THRESHOLD,
    STREAM_FALLBACK_NEW_WORD_THRESHOLD,
    STREAM_HARD_SILENCE_SECONDS,
    STREAM_SEMANTIC_SILENCE_SECONDS,
)
from gpt import submit_to_chatgpt
from speech.constants import (
    DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
    DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL,
    DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
    DEFAULT_FINAL_TRANSCRIPTION_MODEL,
)
from gpt.prompts import build_stream_prompt
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
    SpeakerHint,
    SpeakerIdentifier,
    Transcriber,
    create_transcriber,
    enroll_interviewee_voice,
    model_path_for_run,
)


@dataclass(frozen=True)
class RuntimeOptions:
    """Options selected by the command-line interface."""

    ask_chatgpt: bool = True
    enroll_me: bool = False
    photo_mode: PhotoMode = "none"
    audio_enhancement: str = AUDIO_ENHANCEMENT_SPECTRAL_GATE


def run(options: RuntimeOptions) -> None:
    """Run continuous stream mode."""
    if options.enroll_me:
        enroll_interviewee_voice()
        return

    stream_loop(options)


def stream_loop(options: RuntimeOptions) -> None:
    """Continuously capture interview segments and submit each segment for feedback."""
    is_first_submission = True

    print_stream_mode_banner(options)

    with tempfile.TemporaryDirectory(prefix="secondvoice-eval-") as temp_dir:
        segment_queue: queue.Queue[CompletedStreamSegment | Exception] = queue.Queue()
        stop_event = threading.Event()
        semantic_endpoint_detector = OllamaSemanticEndpointDetector()
        recorder = start_stream_recorder(
            Path(temp_dir),
            segment_queue,
            stop_event,
            semantic_endpoint_detector,
        )
        photo_timer = (
            start_photo_timer(stop_event, options.photo_mode)
            if options.photo_mode != "none"
            else None
        )

        try:
            use_local_transcription_cache = options.ask_chatgpt
            endpoint_transcription_model = model_path_for_run(
                DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
                DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL,
                use_local_cache=use_local_transcription_cache,
            )
            final_transcription_model = model_path_for_run(
                DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
                DEFAULT_FINAL_TRANSCRIPTION_MODEL,
                use_local_cache=use_local_transcription_cache,
            )
            endpoint_transcriber = create_transcriber(
                DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
                endpoint_transcription_model,
            )
            final_transcriber = create_transcriber(
                DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
                final_transcription_model,
            )
            semantic_endpoint_detector.set_transcriber(endpoint_transcriber)
            speaker_identifier = SpeakerIdentifier()
            photo_tracker = PhotoUploadTracker()
            while True:
                segment = next_stream_segment(segment_queue)
                if segment is None:
                    continue

                submitted = process_stream_segment(
                    segment,
                    final_transcriber,
                    speaker_identifier,
                    options,
                    include_mode_prompt=is_first_submission,
                    photo_tracker=photo_tracker,
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
    semantic_endpoint_detector: OllamaSemanticEndpointDetector,
) -> threading.Thread:
    """Start the background recorder that feeds completed segments into a queue."""
    recorder = threading.Thread(
        target=stream_utterance_segments,
        args=(
            output_dir,
            segment_queue,
            stop_event,
            STREAM_HARD_SILENCE_SECONDS,
            DEFAULT_SILENCE_THRESHOLD,
            STREAM_SEMANTIC_SILENCE_SECONDS,
            semantic_endpoint_detector.detect,
            STREAM_FALLBACK_NEW_WORD_THRESHOLD,
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
    transcriber: Transcriber,
    speaker_identifier: SpeakerIdentifier,
    options: RuntimeOptions,
    include_mode_prompt: bool,
    photo_tracker: PhotoUploadTracker,
) -> bool:
    """Transcribe one stream segment and optionally submit it for feedback."""
    audio_path = segment.path
    enhanced_audio_path = audio_path.with_name(f"{audio_path.stem}-enhanced.wav")
    speaker_hint = build_speaker_hint(audio_path, speaker_identifier)
    transcription_path = enhance_wav(
        audio_path,
        enhanced_audio_path,
        AudioEnhancementConfig(mode=options.audio_enhancement),
    )
    try:
        transcript = transcriber.transcribe(transcription_path)
    finally:
        enhanced_audio_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
    print_transcript(transcript)
    print_speaker_hint(speaker_hint)

    if not transcript:
        logger.info("No speech detected. Listening again.")
        return False

    trimmed_transcript = trim_repetitive_transcript_suffix(transcript)
    if trimmed_transcript != transcript:
        logger.info("Trimmed repetitive transcript suffix before submission.")
        logger.debug(
            "Final transcript trimmed from {!r} to {!r}.",
            transcript,
            trimmed_transcript,
        )
        transcript = trimmed_transcript

    if not transcript:
        logger.info("Skipping transcript because it appears repetitive or garbled.")
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
        submitted_to_chatgpt = submit_to_chatgpt(
            build_stream_prompt(
                transcript,
                include_mode_prompt,
                speaker_hint,
                include_photo_context=photo_path is not None,
            ),
            photo_path=photo_path,
        )
        if submitted_to_chatgpt and photo_signature is not None:
            photo_tracker.last_signature = photo_signature
    logger.info("")
    return options.ask_chatgpt


def build_speaker_hint(audio_path: Path, speaker_identifier: SpeakerIdentifier) -> SpeakerHint:
    """Return a speaker hint, falling back gracefully when matching is unavailable."""
    try:
        return speaker_identifier.match(audio_path)
    except RuntimeError as error:
        logger.warning("Voice matching unavailable: {}", error)
        return SpeakerHint(
            role_hint="unknown",
            confidence=None,
            similarity=None,
            profile_available=False,
        )


def print_speaker_hint(speaker_hint: SpeakerHint) -> None:
    """Print the voice match confidence for the most recent segment."""
    logger.info("Interviewee voice hint: {}", speaker_hint.log_value())


def print_stream_mode_banner(options: RuntimeOptions) -> None:
    """Print the active stream-mode trigger settings."""
    logger.info("Stream mode is active.")
    logger.info(
        "Audio segments end after semantic completion or {:g}s silence fallback.",
        STREAM_HARD_SILENCE_SECONDS,
    )
    logger.debug(
        "Semantic endpoint check: {:g}s pause via local Ollama qwen2.5:1.5b",
        STREAM_SEMANTIC_SILENCE_SECONDS,
    )
    logger.info(
        "Endpoint transcription: {} {}",
        DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
        DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL,
    )
    logger.info(
        "Final transcription: {} {}",
        DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
        DEFAULT_FINAL_TRANSCRIPTION_MODEL,
    )
    if options.audio_enhancement != AUDIO_ENHANCEMENT_OFF:
        logger.info("Audio enhancement: {}", options.audio_enhancement)
    else:
        logger.info("Audio enhancement: disabled")
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
    logger.debug("Recording continues while segments are transcribed")
    if options.ask_chatgpt:
        logger.info("ChatGPT submission: enabled")
        logger.debug("Segment role detection: delegated to ChatGPT")
    else:
        logger.info("ChatGPT submission: disabled")
    logger.info("Press Ctrl+C to stop.")


def print_transcript(transcript: str) -> None:
    """Print the transcript from the most recent captured utterance."""
    logger.info("Heard:\n{}", transcript.strip() or "(No speech detected.)")
