import json
from urllib.error import URLError
from urllib.request import urlopen
from typing import Protocol

from loguru import logger

from automation.constants import DEFAULT_CDP_URL
from audio import AUDIO_ENHANCEMENT_OFF
from speech import DEFAULT_ENDPOINT_MODEL, OLLAMA_CHAT_URL


class PreflightOptions(Protocol):
    """Runtime options required by dependency checks."""

    ask_chatgpt: bool
    enroll_me: bool
    audio_enhancement: str


def check_runtime_dependencies(options: PreflightOptions) -> None:
    """Abort early when required local services are not ready."""
    if options.enroll_me:
        return

    logger.info("Running startup dependency checks...")
    checks_passed = True

    if not ollama_model_is_ready():
        checks_passed = False

    if (
        options.audio_enhancement != AUDIO_ENHANCEMENT_OFF
        and not audio_enhancement_is_ready()
    ):
        checks_passed = False

    if options.ask_chatgpt and not cdp_browser_is_ready():
        checks_passed = False

    if not checks_passed:
        logger.error("Startup checks failed. Aborting before microphone capture.")
        raise SystemExit(1)

    logger.info("Startup dependency checks passed.")


def audio_enhancement_is_ready() -> bool:
    """Return whether the configured in-app noise reducer is importable."""
    try:
        import noisereduce  # noqa: F401
    except ImportError as error:
        logger.error("Audio enhancement dependency noisereduce is not installed: {}", error)
        logger.error(
            "Install dependencies with: .venv/bin/python -m pip install -r "
            "requirements.txt"
        )
        return False

    logger.info("Audio enhancement ready: noisereduce")
    return True


def ollama_model_is_ready() -> bool:
    """Return whether Ollama is reachable and has the endpoint model installed."""
    tags_url = OLLAMA_CHAT_URL.rsplit("/", 1)[0] + "/tags"
    try:
        with urlopen(tags_url, timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as error:
        logger.error("Could not reach Ollama at {}: {}", tags_url, error)
        logger.error("Start Ollama with: ollama serve")
        return False

    model_names = {model.get("name") for model in body.get("models", [])}
    if DEFAULT_ENDPOINT_MODEL not in model_names:
        logger.error("Ollama model {} is not installed.", DEFAULT_ENDPOINT_MODEL)
        logger.error("Install it with: ollama pull {}", DEFAULT_ENDPOINT_MODEL)
        return False

    logger.info("Ollama ready: {}", DEFAULT_ENDPOINT_MODEL)
    return True


def cdp_browser_is_ready() -> bool:
    """Return whether Chrome is reachable over the configured CDP port."""
    version_url = f"{DEFAULT_CDP_URL}/json/version"
    try:
        with urlopen(version_url, timeout=2) as response:
            response.read()
    except (OSError, URLError) as error:
        logger.error("Could not reach Chrome CDP at {}: {}", version_url, error)
        logger.error(
            "Start Chrome with:\n"
            "  open -na 'Google Chrome' --args \\\n"
            "    --remote-debugging-port=9222 \\\n"
            "    --user-data-dir=$HOME/.secondvoice/cdp-browser-profile"
        )
        return False

    logger.info("Chrome CDP ready: {}", DEFAULT_CDP_URL)
    return True
