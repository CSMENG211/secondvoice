import re
from dataclasses import dataclass, field

from gpt.constants import PHOTO_CONTEXT_PROMPT, ROUND_PROMPTS


@dataclass
class BehaviorState:
    """In-memory behavioral-story tracking for one app run."""

    used_story_ids: set[str] = field(default_factory=set)


@dataclass
class DesignState:
    """In-memory system-design deep-dive tracking for one app run."""

    used_deep_dive_topic_ids: set[str] = field(default_factory=set)


def build_stream_prompt(
    transcript: str,
    include_mode_prompt: bool,
    include_photo_context: bool = False,
    *,
    round_type: str = "coding",
    round_context: str = "",
    behavior_state: BehaviorState | None = None,
    design_state: DesignState | None = None,
) -> str:
    """Build the round-specific prompt sent to ChatGPT."""
    photo_context = f"\n{PHOTO_CONTEXT_PROMPT}\n" if include_photo_context else ""
    behavior_block = behavior_state_block(behavior_state) if round_type == "behavior" else ""
    design_block = design_state_block(design_state) if round_type == "design" else ""
    segment_prompt = (
        "Process this transcript segment.\n\n"
        f"{photo_context}"
        f"{behavior_block}"
        f"{design_block}"
        "Transcript:\n"
        f"{transcript}"
    )
    if include_mode_prompt:
        round_prompt = ROUND_PROMPTS.get(round_type, ROUND_PROMPTS["coding"])
        context_block = (
            f"\nRound Context:\n{round_context}\n" if round_context.strip() else "\nRound Context:\n(none)\n"
        )
        return f"{round_prompt}\n{context_block}\n{segment_prompt}"
    return segment_prompt


def behavior_state_block(behavior_state: BehaviorState | None) -> str:
    """Serialize used behavioral story IDs for exclusion in future picks."""
    if behavior_state is None or not behavior_state.used_story_ids:
        return "Used Story IDs:\n(none)\n\n"
    ids = ", ".join(sorted(behavior_state.used_story_ids))
    return f"Used Story IDs:\n{ids}\n\n"


def design_state_block(design_state: DesignState | None) -> str:
    """Serialize used design deep-dive topic IDs for exclusion in future picks."""
    if design_state is None or not design_state.used_deep_dive_topic_ids:
        return "Used Design Deep Dive Topic IDs:\n(none)\n\n"
    ids = ", ".join(sorted(design_state.used_deep_dive_topic_ids))
    return f"Used Design Deep Dive Topic IDs:\n{ids}\n\n"


def parse_actual_story_id(response_text: str) -> str | None:
    """Parse actual_story_id from a model metadata bullet."""
    if not response_text.strip():
        return None
    match = re.search(r"actual_story_id=([a-zA-Z0-9_.-]+)", response_text)
    if not match:
        return None
    value = match.group(1).strip().lower()
    if value == "unmapped":
        return None
    return value


def parse_design_deep_dive_topic_ids(response_text: str) -> set[str]:
    """Parse design_deep_dive_topic_ids from a model metadata bullet."""
    if not response_text.strip():
        return set()
    match = re.search(r"design_deep_dive_topic_ids=([a-zA-Z0-9_.\-, ]+)", response_text)
    if not match:
        return set()
    return {
        value.strip().lower()
        for value in match.group(1).split(",")
        if value.strip() and value.strip().lower() != "none"
    }
