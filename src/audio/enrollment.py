from pathlib import Path

from loguru import logger
import sounddevice as sd

from audio.levels import (
    audio_blocksize,
    block_count_for_seconds,
    chunk_is_speech,
    create_pre_roll_buffer,
)
from audio.wav import open_wav_writer, write_chunks
from constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
)


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
