import queue
import tempfile
import threading
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from loguru import logger

from audio import (
    CompletedStreamSegment,
    audio_blocksize,
    block_count_for_seconds,
    stream_utterance_segments,
    write_wav_slice,
)
from audio.constants import (
    DEFAULT_SILENCE_THRESHOLD,
    FINAL_TRANSCRIPTION_OVERLAP_SECONDS,
    STREAM_HARD_SILENCE_SECONDS,
    STREAM_TRANSCRIPT_AGREEMENT_COUNT,
)
from gpt import submit_to_chatgpt
from speech.constants import (
    DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND,
    DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL,
    DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
    DEFAULT_FINAL_TRANSCRIPTION_MODEL,
)
from gpt import ensure_context_templates, load_round_context
from gpt.prompts import (
    BehaviorState,
    DesignState,
    build_stream_prompt,
    parse_actual_story_id,
    parse_design_deep_dive_topic_ids,
)
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


InterviewRound = Literal["coding", "design", "behavior", "offer"]


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
    design_state = DesignState()

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
        final_transcription_model = model_path_for_run(
            DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
            DEFAULT_FINAL_TRANSCRIPTION_MODEL,
            use_local_cache=True,
        )
        final_transcriber = create_transcriber(
            DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
            final_transcription_model,
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
                    design_state=design_state,
                    final_transcriber=final_transcriber,
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
    design_state: DesignState,
    final_transcriber: Transcriber | None = None,
) -> bool:
    """Transcribe one stream segment and optionally submit it for feedback."""
    audio_path = segment.path
    try:
        transcript = segment.transcript
        if final_transcriber is not None:
            transcript = finalize_segment_transcript(
                audio_path,
                transcript,
                locked_transcript=segment.locked_transcript,
                locked_chunk_index=segment.locked_chunk_index,
                final_transcriber=final_transcriber,
            )
    finally:
        audio_path.unlink(missing_ok=True)
    print_transcript(transcript)

    if not transcript:
        logger.info("No speech detected. Listening again.")
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
                design_state=design_state,
            ),
            photo_path=photo_path,
        )
        if options.round_type == "behavior":
            actual_story_id = parse_actual_story_id(response_text or "")
            if actual_story_id is not None:
                behavior_state.used_story_ids.add(actual_story_id)
        if options.round_type == "design":
            design_state.used_deep_dive_topic_ids.update(
                parse_design_deep_dive_topic_ids(response_text or "")
            )
        if submitted_to_chatgpt and photo_signature is not None:
            photo_tracker.last_signature = photo_signature
    logger.info("")
    return options.ask_chatgpt


def finalize_segment_transcript(
    audio_path: Path,
    streamed_transcript: str,
    locked_transcript: str,
    locked_chunk_index: int,
    final_transcriber: Transcriber,
) -> str:
    """Upgrade one completed transcript by finalizing only the unresolved audio tail."""
    tail_audio_path = build_final_transcription_audio(
        audio_path,
        locked_transcript=locked_transcript,
        locked_chunk_index=locked_chunk_index,
    )
    try:
        final_transcript = final_transcriber.transcribe(
            tail_audio_path,
            log_progress=False,
        ).strip()
    except Exception as exc:
        logger.warning(
            "Final transcript pass failed for {}: {}. Falling back to live transcript.",
            tail_audio_path.name,
            exc,
        )
        return streamed_transcript
    finally:
        if tail_audio_path != audio_path:
            tail_audio_path.unlink(missing_ok=True)

    if not final_transcript:
        logger.warning(
            "Final transcript pass returned empty text for {}. Falling back to live transcript.",
            tail_audio_path.name,
        )
        return streamed_transcript

    finalized_transcript = combine_locked_and_tail_transcript(
        locked_transcript,
        final_transcript,
    )
    if finalized_transcript != streamed_transcript:
        logger.info("Using finalized transcript from saved segment audio tail.")
    return finalized_transcript


def build_final_transcription_audio(
    audio_path: Path,
    *,
    locked_transcript: str,
    locked_chunk_index: int,
) -> Path:
    """Return the best audio slice for one final transcription pass."""
    if not locked_transcript or locked_chunk_index <= 0:
        return audio_path

    overlap_blocks = block_count_for_seconds(FINAL_TRANSCRIPTION_OVERLAP_SECONDS)
    start_chunk_index = max(0, locked_chunk_index - overlap_blocks)
    if start_chunk_index <= 0:
        return audio_path

    start_frame = start_chunk_index * audio_blocksize()
    tail_audio_path = audio_path.with_name(
        f"{audio_path.stem}-final-tail-{start_chunk_index:04d}.wav"
    )
    write_wav_slice(tail_audio_path, audio_path, start_frame=start_frame)
    return tail_audio_path


def combine_locked_and_tail_transcript(locked_transcript: str, tail_transcript: str) -> str:
    """Merge a stabilized prefix with a tail transcript that may include overlap."""
    locked = locked_transcript.strip()
    tail = tail_transcript.strip()
    if not locked:
        return tail
    if not tail:
        return locked

    locked_words = locked.split()
    tail_words = tail.split()
    max_overlap = min(len(locked_words), len(tail_words), 12)
    for overlap in range(max_overlap, 0, -1):
        if [word.lower() for word in locked_words[-overlap:]] == [
            word.lower() for word in tail_words[:overlap]
        ]:
            tail_words = tail_words[overlap:]
            break

    if not tail_words:
        return locked
    return f"{locked} {' '.join(tail_words)}"


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
