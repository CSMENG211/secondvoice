import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt.prompts import (
    BehaviorState,
    DesignState,
    build_stream_prompt,
    parse_actual_story_id,
    parse_design_deep_dive_topic_ids,
)


def test_first_submission_includes_round_preamble_and_context() -> None:
    prompt = build_stream_prompt(
        "How would you approach this graph problem?",
        include_mode_prompt=True,
        round_type="coding",
        round_context="Use clear tradeoffs.",
    )
    assert "Coding response contract" in prompt
    assert "Round Context:\nUse clear tradeoffs." in prompt


def test_follow_up_submission_omits_round_preamble() -> None:
    prompt = build_stream_prompt(
        "I think BFS works.",
        include_mode_prompt=False,
        round_type="coding",
        round_context="ignored after first turn",
    )
    assert "Coding response contract" not in prompt
    assert "Round Context:" not in prompt
    assert "Transcript:\nI think BFS works." in prompt


def test_behavior_state_and_actual_story_parsing() -> None:
    state = BehaviorState(used_story_ids={"story_a"})
    prompt = build_stream_prompt(
        "Tell me about conflict.",
        include_mode_prompt=True,
        round_type="behavior",
        round_context="## story_id: story_a\n...\n## story_id: story_b\n...\n## story_id: story_c\n...",
        behavior_state=state,
    )
    assert "Pick the top 2 best untold stories" in prompt
    assert "Do not print raw story IDs in the visible answer text." in prompt
    assert "Do not add a 'strongest match' or restated-summary sentence under the title." in prompt
    assert "Each bullet should help the candidate remember the story quickly" in prompt
    assert "suggested_story_ids=<id1,id2>" in prompt
    assert "Used Story IDs:\nstory_a" in prompt
    parsed = parse_actual_story_id(
        "- META: suggested_story_ids=story_b,story_d; actual_story_id=story_c"
    )
    assert parsed == "story_c"
    assert parse_actual_story_id("- META: suggested_story_ids=story_b,story_d; actual_story_id=unmapped") is None


def test_design_state_and_deep_dive_topic_parsing() -> None:
    state = DesignState(used_deep_dive_topic_ids={"cache_invalidation"})
    prompt = build_stream_prompt(
        "Let's design a feed system.",
        include_mode_prompt=True,
        round_type="design",
        round_context="Prefer practical reliability tradeoffs.",
        design_state=state,
    )

    assert "System design response contract" in prompt
    assert "Silently infer the current phase of the system design round" in prompt
    assert "question phase, clarify functional requirement, clarify nonfunctional requirement" in prompt
    assert "If the current phase is question phase, do not propose deep-dive options" in prompt
    assert "one restating the problem in one sentence" in prompt
    assert "one confirming the main user journey plus primary object/action" in prompt
    assert "Only if the current phase is deep dive should you give exactly 2 possible deep-dive topics" in prompt
    assert "If no deep-dive topic is proposed, use: - META: design_deep_dive_topic_ids=none" in prompt
    assert "Use human-readable deep-dive titles in the visible answer" in prompt
    assert "Do not show raw snake_case topic IDs in visible deep-dive titles" in prompt
    assert "Use short stable snake_case IDs only in the metadata bullet" in prompt
    assert "Exclude topics listed in Used Design Deep Dive Topic IDs." in prompt
    assert "design_deep_dive_topic_ids=<id1,id2>" in prompt
    assert "Used Design Deep Dive Topic IDs:\ncache_invalidation" in prompt

    parsed = parse_design_deep_dive_topic_ids(
        "- META: design_deep_dive_topic_ids=shard_hotspots, queue_backpressure"
    )
    assert parsed == {"shard_hotspots", "queue_backpressure"}
    assert parse_design_deep_dive_topic_ids("- META: design_deep_dive_topic_ids=none") == set()


def main() -> None:
    test_first_submission_includes_round_preamble_and_context()
    test_follow_up_submission_omits_round_preamble()
    test_behavior_state_and_actual_story_parsing()
    test_design_state_and_deep_dive_topic_parsing()
    print("round prompt tests passed")


if __name__ == "__main__":
    main()
