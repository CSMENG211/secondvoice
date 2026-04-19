import json
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from loguru import logger

from transcription import LocalTranscriber

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_ENDPOINT_MODEL = "qwen2.5:1.5b"
OLLAMA_KEEP_ALIVE = "60m"
OLLAMA_REQUEST_TIMEOUT_SECONDS = 10
ENDPOINT_LABEL_COMPLETE = "COMPLETE"
ENDPOINT_LABEL_INCOMPLETE = "INCOMPLETE"

ENDPOINT_SYSTEM_PROMPT = (
    "You are an endpoint detector for live coding interview speech. "
    "Return exactly one word: COMPLETE or INCOMPLETE. "
    "COMPLETE means the transcript is usable as a standalone segment. "
    "It may be short. A stated value, action, condition, complexity, "
    "edge case, or plan step can be COMPLETE. "
    "INCOMPLETE means the transcript ends mid-phrase, after a connector, "
    "or clearly expects more words immediately. "
    "Examples: Transcript: so it is -> INCOMPLETE. "
    "Transcript: so it is true -> COMPLETE. "
    "Transcript: return -> INCOMPLETE. "
    "Transcript: return false -> COMPLETE. "
    "Transcript: the time complexity is -> INCOMPLETE. "
    "Transcript: the time complexity is O of n -> COMPLETE. "
    "Transcript: I would use a hash map because -> INCOMPLETE. "
    "Transcript: I would use a hash map because lookup is constant time -> COMPLETE. "
    "Transcript: and then -> INCOMPLETE. "
    "Transcript: and then I move the left pointer -> COMPLETE."
)


class OllamaSemanticEndpointDetector:
    """Draft-audio endpoint detector backed by local transcription and Ollama."""

    def __init__(self, model: str = DEFAULT_ENDPOINT_MODEL) -> None:
        self.model = model
        self._transcriber: LocalTranscriber | None = None
        self._lock = threading.Lock()

    def set_transcriber(self, transcriber: LocalTranscriber) -> None:
        """Attach the shared transcriber once Whisper has loaded."""
        with self._lock:
            self._transcriber = transcriber

    def is_complete(self, audio_path: Path) -> bool:
        """Return True only when the draft audio is confidently complete."""
        transcriber = self._current_transcriber()
        if transcriber is None:
            logger.debug("Semantic endpoint check skipped; transcriber is not ready.")
            return False

        transcript = transcriber.transcribe(audio_path, log_progress=False)
        if not transcript:
            logger.debug("Semantic endpoint check skipped; draft transcript is empty.")
            return False

        try:
            label, duration_ms = classify_endpoint_transcript(transcript, self.model)
        except URLError as error:
            logger.warning("Semantic endpoint check could not reach Ollama: {}", error)
            return False
        except Exception as error:
            logger.warning("Semantic endpoint check failed: {}", error)
            return False

        logger.debug(
            "Semantic endpoint check: {} ({:.1f} ms) for draft: {}",
            label,
            duration_ms,
            transcript,
        )
        return label == ENDPOINT_LABEL_COMPLETE

    def _current_transcriber(self) -> LocalTranscriber | None:
        with self._lock:
            return self._transcriber


def classify_endpoint_transcript(
    transcript: str,
    model: str = DEFAULT_ENDPOINT_MODEL,
    url: str = OLLAMA_CHAT_URL,
    timeout_seconds: float = OLLAMA_REQUEST_TIMEOUT_SECONDS,
) -> tuple[str, float]:
    """Classify a transcript as COMPLETE or INCOMPLETE using Ollama."""
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 3,
        },
        "messages": [
            {"role": "system", "content": ENDPOINT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript: {transcript}"},
        ],
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    duration_ms = (time.perf_counter() - started) * 1000

    content = body["message"]["content"].strip().upper()
    if ENDPOINT_LABEL_INCOMPLETE in content:
        return ENDPOINT_LABEL_INCOMPLETE, duration_ms
    if ENDPOINT_LABEL_COMPLETE in content:
        return ENDPOINT_LABEL_COMPLETE, duration_ms
    return f"OTHER:{content}", duration_ms
