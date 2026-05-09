from pathlib import Path

from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = REPO_ROOT / "context"
LEGACY_CONTEXT_DIR = Path.home() / ".secondvoice" / "context"

ROUND_CONTEXT_FILES = {
    "coding": "coding.md",
    "system-design": "system_design.md",
    "behavior": "behavior.md",
    "offer-negotiation": "offer_negotiation.md",
}


def ensure_context_templates() -> None:
    """Create round context files with editable placeholders when missing."""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    templates = {
        "coding.md": (
            "# Coding Round Context\n\n"
            "## Preferred Patterns\n"
            "- Add your preferred DS/algorithm patterns.\n\n"
            "## Strong Areas\n"
            "- Topics where you want concise confidence.\n\n"
            "## Weak Areas\n"
            "- Topics where you want fallback guidance.\n"
        ),
        "system_design.md": (
            "# System Design Round Context\n\n"
            "## Domain Background\n"
            "- Product/domain context you want answers grounded in.\n\n"
            "## Preferences\n"
            "- Scale assumptions, tech stack preferences, tradeoff style.\n\n"
            "## Deep Dive Priorities\n"
            "- Caching, consistency, queues, storage, reliability, etc.\n"
        ),
        "behavior.md": (
            "# Behavioral Round Context\n\n"
            "Use one story block per story.\n\n"
            "## story_id: conflict_cross_team\n"
            "Situation: <fill>\n"
            "Task: <fill>\n"
            "Action: <fill>\n"
            "Result: <fill>\n\n"
            "## story_id: ambiguity_new_scope\n"
            "Situation: <fill>\n"
            "Task: <fill>\n"
            "Action: <fill>\n"
            "Result: <fill>\n"
        ),
        "offer_negotiation.md": (
            "# Offer Negotiation Context\n\n"
            "## Current Role\n"
            "- Company / level / location\n"
            "- Current total compensation breakdown\n\n"
            "## Current Offers\n"
            "- Offer A: company, level, base/equity/bonus/sign-on/TC\n"
            "- Offer B: ...\n\n"
            "## Historical Same-Company/Same-Level Offers\n"
            "- Include past ranges and outcomes.\n\n"
            "## Constraints and Goals\n"
            "- Walk-away criteria, priorities, timeline, non-comp items.\n"
        ),
    }
    for filename, template in templates.items():
        path = CONTEXT_DIR / filename
        legacy_path = LEGACY_CONTEXT_DIR / filename
        if not path.exists() and legacy_path.exists():
            path.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
            continue
        if not path.exists():
            path.write_text(template, encoding="utf-8")


def load_round_context(round_type: str) -> str:
    """Load the configured context text for one interview round."""
    filename = ROUND_CONTEXT_FILES.get(round_type)
    if filename is None:
        return ""
    path = CONTEXT_DIR / filename
    if not path.exists():
        logger.warning("Context file missing for {}: {}", round_type, path)
        return ""
    return path.read_text(encoding="utf-8").strip()
