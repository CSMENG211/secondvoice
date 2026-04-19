import platform
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path
from time import sleep


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import build_stream_prompt
from browser import submit_to_chatgpt
from constants import DEFAULT_CDP_URL, STATIC_INTERVIEW_PHOTO_PATH
from logging_config import configure_logging


CDP_BROWSER_PROFILE_DIR = Path.home() / ".secondvoice" / "cdp-browser-profile"

SMOKE_TEST_TRANSCRIPT = (
    "Given an array of integers and a target value, return whether there is a "
    "pair of integers in the array that sums to the target."
)


def main() -> None:
    """Submit a fixed transcript segment to ChatGPT through browser automation."""
    configure_logging()
    open_automation_chrome()
    submit_to_chatgpt(
        build_stream_prompt(
            SMOKE_TEST_TRANSCRIPT,
            include_mode_prompt=True,
            include_photo_context=True,
        ),
        photo_path=STATIC_INTERVIEW_PHOTO_PATH,
    )


def open_automation_chrome() -> None:
    """Open the Chrome profile that Playwright connects to over CDP."""
    if cdp_browser_is_running():
        return

    if platform.system() != "Darwin":
        return

    CDP_BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "open",
            "-na",
            "Google Chrome",
            "--args",
            "--remote-debugging-port=9222",
            f"--user-data-dir={CDP_BROWSER_PROFILE_DIR}",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sleep(2)


def cdp_browser_is_running() -> bool:
    """Return whether the Chrome CDP endpoint is already reachable."""
    try:
        with urlopen(f"{DEFAULT_CDP_URL}/json/version", timeout=0.5):
            return True
    except (OSError, URLError):
        return False


if __name__ == "__main__":
    main()
