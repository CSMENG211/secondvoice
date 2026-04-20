import math
from pathlib import Path
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audio.enhancement as enhancement
from audio import AudioEnhancementConfig
from audio.constants import AUDIO_SAMPLE_RATE


def write_test_wav(path: Path, samples: np.ndarray) -> None:
    """Write mono int16 samples to a test WAV."""
    int_samples = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(AUDIO_SAMPLE_RATE)
        wav_file.writeframes(int_samples.tobytes())


def wav_info(path: Path) -> tuple[int, int, int, int]:
    """Return channels, sample width, sample rate, and frame count."""
    with wave.open(str(path), "rb") as wav_file:
        return (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
            wav_file.getnframes(),
        )


def test_enhance_wav_preserves_pcm_shape() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.wav"
        output_path = Path(temp_dir) / "enhanced.wav"
        seconds = 0.5
        sample_count = int(AUDIO_SAMPLE_RATE * seconds)
        t = np.arange(sample_count) / AUDIO_SAMPLE_RATE
        speech = 8000 * np.sin(2 * math.pi * 220 * t)
        keyboard_noise = np.zeros(sample_count)
        keyboard_noise[::800] = 12000
        write_test_wav(input_path, speech + keyboard_noise)

        original_reduce_noise = enhancement.reduce_noise
        try:
            enhancement.reduce_noise = lambda samples, _sample_rate, _config: samples * 0.5
            result = enhancement.enhance_wav(
                input_path,
                output_path,
                AudioEnhancementConfig(),
            )
        finally:
            enhancement.reduce_noise = original_reduce_noise

        assert result == output_path
        assert output_path.exists()
        assert wav_info(output_path) == (1, 2, AUDIO_SAMPLE_RATE, sample_count)


def test_write_float32_as_wav_clips_to_int16() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "clipped.wav"
        enhancement.write_float32_as_wav(
            output_path,
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        )

        with wave.open(str(output_path), "rb") as wav_file:
            samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

        assert samples.min() >= -32767
        assert samples.max() <= 32767
        assert samples.tolist() == [-32767, -32767, 0, 32767, 32767]


def test_disabled_mode_returns_raw_without_reducing() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.wav"
        output_path = Path(temp_dir) / "enhanced.wav"
        write_test_wav(input_path, np.zeros(160, dtype=np.int16))

        original_reduce_noise = enhancement.reduce_noise
        try:
            enhancement.reduce_noise = fail_if_called
            result = enhancement.enhance_wav(
                input_path,
                output_path,
                AudioEnhancementConfig(enabled=False),
            )
        finally:
            enhancement.reduce_noise = original_reduce_noise

        assert result == input_path
        assert not output_path.exists()


def fail_if_called(*_args: object, **_kwargs: object) -> np.ndarray:
    raise AssertionError("reduce_noise should not be called")


def main() -> None:
    test_enhance_wav_preserves_pcm_shape()
    test_write_float32_as_wav_clips_to_int16()
    test_disabled_mode_returns_raw_without_reducing()
    print("audio enhancement tests passed")


if __name__ == "__main__":
    main()
