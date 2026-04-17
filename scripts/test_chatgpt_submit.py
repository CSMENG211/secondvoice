import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offergpt.browser import submit_to_chatgpt


def main() -> None:
    submit_to_chatgpt("What is 1 + 1?")


if __name__ == "__main__":
    main()
