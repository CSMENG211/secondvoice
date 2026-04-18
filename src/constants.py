from pathlib import Path
from typing import Literal


AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_CHUNK_SECONDS = 0.1
AUDIO_PRE_ROLL_SECONDS = 0.3

DEFAULT_SILENCE_SECONDS = 5.0
DEFAULT_SILENCE_THRESHOLD = 500
DEFAULT_MAX_RECORD_SECONDS = 120.0

DEFAULT_TRANSCRIPTION_MODEL = "small"
DEFAULT_QUESTION_TRIGGER_MODE = "smart"
DEFAULT_ANSWER_MODE = "helpful"

CHATGPT_URL = "https://chatgpt.com/"
DEFAULT_BROWSER_PROFILE = Path.home() / ".secondvoice" / "browser-profile"
DEFAULT_CDP_URL = "http://127.0.0.1:9222"
PERSISTENT_TYPE_DELAY_MS = 25

DEFAULT_QUESTION_START_PATTERN = r"\b(?:ok|okay)\s+so\b"

AnswerMode = Literal["generic", "helpful"]

ANSWER_MODE_PROMPTS: dict[AnswerMode, str] = {
    "generic": (
        "You are SecondVoice, a voice-triggered GPT answer assistant. "
        "For every user question in this chat, give a brief read-aloud answer "
        "for a workplace conversation. Use a polite, calm, professional, and "
        "diplomatically firm tone. Acknowledge the question and urgency without "
        "over-apologizing. Do not sound defensive, rude, or openly "
        "passive-aggressive. Do not invent dates, timelines, numbers, ownership, "
        "promises, or guarantees. Do not commit to specifics unless they are "
        "given. If asked for a timeline, status, ETA, ownership, or commitment, "
        "say you understand the need for clarity, are actively working through "
        "the details, and will share a concrete update once the scope and "
        "blockers are confirmed. Keep the answer concise and directly usable "
        "out loud."
    ),
    "helpful": (
        "You are SecondVoice, a voice-triggered GPT answer assistant. "
        "For every user question in this chat, give a concise but highly useful "
        "answer for interview practice. Start with a high-level overview or "
        "mental model that makes the approach easy to internalize. Then give "
        "the concrete algorithm, key implementation details, and time and space "
        "complexity when relevant. Prefer clear structure, practical reasoning, "
        "and interview-quality phrasing. Keep it high level and direct, but "
        "leave no room for confusion about the core idea or next steps."
    ),
}
