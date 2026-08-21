from __future__ import annotations

from functools import lru_cache
from pathlib import Path


SKILLS_DIR = Path(__file__).with_name("skills")


@lru_cache(maxsize=None)
def load_skill(name: str) -> str:
    """Load the instruction body of a bundled SKILL.md file."""
    path = SKILLS_DIR / name / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return text.strip()
    _, separator, body = text[4:].partition("\n---\n")
    if not separator:
        raise RuntimeError(f"Invalid skill frontmatter: {path}")
    return body.strip()


QUERY_REWRITING_SKILL = load_skill("query-rewriting")
CONTEXT_SUMMARIZATION_SKILL = load_skill("context-summarization")
