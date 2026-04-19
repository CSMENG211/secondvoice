from array import array
from collections.abc import Iterable
from collections import deque
import math
import queue
import threading
import wave
from pathlib import Path
from typing import Callable

from loguru import logger
import sounddevice as sd

from constants import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SECONDS,
    AUDIO_PRE_ROLL_SECONDS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH_BYTES,
    DEFAULT_SILENCE_THRESHOLD,
    STREAM_HARD_SILENCE_SECONDS,
    STREAM_SEMANTIC_SILENCE_SECONDS,
)

SemanticEndpointDetector = Callable[[Path], bool]


def capture_enrollment_utterance(
    output_path: Path,
    silence_seconds: float,
    silence_threshold: int,
    max_record_seconds: float,
) -> None:
    """Capture one enrollment sentence with speech start and silence stop triggers."""
    blocksize = audio_blocksize()
    silence_blocks_needed = block_count_for_seconds(silence_seconds)
    max_blocks = block_count_for_seconds(max_record_seconds)
    pre_roll = create_pre_roll_buffer()
    recorded_blocks = 0
    silent_blocks = 0
    recording_started = False

    logger.info("Recording starts when you speak.")
    with open_wav_writer(output_path) as wav_file:
        with sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="int16",
            blocksize=blocksize,
        ) as stream:
            while True:
                data, overflowed = stream.read(blocksize)
                chunk = bytes(data)
                if overflowed:
                    logger.warning("Audio warning: input overflowed")

                is_speech = chunk_is_speech(chunk, silence_threshold)

                if not recording_started:
                    pre_roll.append(chunk)
                    if is_speech:
                        logger.info("Enrollment utterance started.")
                        recording_started = True
                        write_chunks(wav_file, pre_roll)
                        recorded_blocks += len(pre_roll)
                        silent_blocks = 0
                    continue

                wav_file.writeframes(chunk)
                recorded_blocks += 1

                if is_speech:
                    silent_blocks = 0
                else:
                    silent_blocks += 1

                if silent_blocks >= silence_blocks_needed:
                    logger.info("Enrollment utterance captured.")
                    break

                if recorded_blocks >= max_blocks:
                    logger.info("Enrollment max length reached ({:g}s).", max_record_seconds)
                    break


def stream_utterance_segments(
    output_dir: Path,
    segment_queue: queue.Queue[Path | Exception],
    stop_event: threading.Event,
    hard_silence_seconds: float = STREAM_HARD_SILENCE_SECONDS,
    silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
    semantic_silence_seconds: float = STREAM_SEMANTIC_SILENCE_SECONDS,
    semantic_endpoint_detector: SemanticEndpointDetector | None = None,
) -> None:
    """Continuously record voice-triggered utterance segments to WAV files."""
    blocksize = audio_blocksize()
    hard_silence_blocks_needed = block_count_for_seconds(hard_silence_seconds)
    semantic_silence_blocks_needed = block_count_for_seconds(semantic_silence_seconds)
    pre_roll = create_pre_roll_buffer()
    recorded_chunks: list[bytes] = []
    silent_blocks = 0
    semantic_checked_this_pause = False
    recording_started = False
    segment_index = 0
    wav_file: wave.Wave_write | None = None
    segment_path: Path | None = None

    def start_segment() -> wave.Wave_write:
        nonlocal segment_index, segment_path
        segment_index += 1
        segment_path = output_dir / f"stream-segment-{segment_index:04d}.wav"
        return open_wav_writer(segment_path)

    def finish_segment() -> None:
        nonlocal wav_file, segment_path, recorded_chunks
        if wav_file is None or segment_path is None:
            return
        wav_file.close()
        segment_queue.put(segment_path)
        wav_file = None
        segment_path = None
        recorded_chunks = []

    def write_segment_chunks(chunks: Iterable[bytes]) -> None:
        if wav_file is None:
            return
        for item in chunks:
            wav_file.writeframes(item)
            recorded_chunks.append(item)

    def current_segment_is_complete() -> bool:
        if semantic_endpoint_detector is None or segment_path is None or not recorded_chunks:
            return False

        draft_path = segment_path.with_name(f"{segment_path.stem}-semantic-check.wav")
        try:
            write_wav_file(draft_path, recorded_chunks)
            return semantic_endpoint_detector(draft_path)
        except Exception as error:
            logger.warning("Semantic endpoint check failed: {}", error)
            return False
        finally:
            draft_path.unlink(missing_ok=True)

    logger.info("Audio stream active. Waiting for speech...")
    try:
        try:
            with sd.RawInputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype="int16",
                blocksize=blocksize,
            ) as stream:
                while not stop_event.is_set():
                    data, overflowed = stream.read(blocksize)
                    chunk = bytes(data)
                    if overflowed:
                        logger.warning("Audio warning: input overflowed")

                    is_speech = chunk_is_speech(chunk, silence_threshold)

                    if not recording_started:
                        pre_roll.append(chunk)
                        if is_speech:
                            logger.info("Utterance started. Capturing audio...")
                            recording_started = True
                            wav_file = start_segment()
                            write_segment_chunks(pre_roll)
                            silent_blocks = 0
                            semantic_checked_this_pause = False
                        continue

                    if wav_file is not None:
                        write_segment_chunks([chunk])

                    if is_speech:
                        silent_blocks = 0
                        semantic_checked_this_pause = False
                    else:
                        silent_blocks += 1

                    if (
                        not semantic_checked_this_pause
                        and silent_blocks >= semantic_silence_blocks_needed
                    ):
                        semantic_checked_this_pause = True
                        if current_segment_is_complete():
                            logger.info(
                                "Audio segment trigger: semantic completion after {:g}s pause.",
                                semantic_silence_seconds,
                            )
                            finish_segment()
                            recording_started = False
                            silent_blocks = 0
                            pre_roll.clear()
                            continue
                        logger.info(
                            "Semantic endpoint check: incomplete; waiting for more speech."
                        )

                    if silent_blocks >= hard_silence_blocks_needed:
                        logger.info(
                            "Audio segment fallback trigger: {:g}s of silence.",
                            hard_silence_seconds,
                        )
                        finish_segment()
                        recording_started = False
                        silent_blocks = 0
                        semantic_checked_this_pause = False
                        pre_roll.clear()
        except Exception as error:
            segment_queue.put(error)
    finally:
        if recording_started:
            finish_segment()


def open_wav_writer(output_path: Path) -> wave.Wave_write:
    """Open a mono int16 WAV file using the app's audio settings."""
    wav_file = wave.open(str(output_path), "wb")
    wav_file.setnchannels(AUDIO_CHANNELS)
    wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH_BYTES)
    wav_file.setframerate(AUDIO_SAMPLE_RATE)
    return wav_file


def write_wav_file(output_path: Path, chunks: Iterable[bytes]) -> None:
    """Write a complete mono int16 WAV file using the app's audio settings."""
    with open_wav_writer(output_path) as wav_file:
        write_chunks(wav_file, chunks)


def audio_blocksize() -> int:
    """Return the number of samples in one audio chunk."""
    return int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS)


def block_count_for_seconds(seconds: float) -> int:
    """Return the number of audio chunks needed to cover a duration."""
    return max(1, math.ceil(seconds / AUDIO_CHUNK_SECONDS))


def create_pre_roll_buffer() -> deque[bytes]:
    """Return a buffer that keeps audio from just before speech starts."""
    return deque(maxlen=block_count_for_seconds(AUDIO_PRE_ROLL_SECONDS))


def write_chunks(wav_file: wave.Wave_write, chunks: Iterable[bytes]) -> None:
    """Write a sequence of raw int16 audio chunks to a WAV file."""
    for chunk in chunks:
        wav_file.writeframes(chunk)


def chunk_is_speech(chunk: bytes, threshold: int) -> bool:
    """Return whether an audio chunk crosses the configured speech threshold."""
    return rms_level(chunk) >= threshold


def rms_level(chunk: bytes) -> float:
    """Calculate the root-mean-square amplitude for an int16 audio chunk."""
    if not chunk:
        return 0.0

    samples = array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0

    square_sum = sum(sample * sample for sample in samples)
    return math.sqrt(square_sum / len(samples))
