"""The README's table of contents is generated from its headings, and checked.

A hand-kept list of links goes stale the first time a heading is renamed, and
a dead anchor fails silently on GitHub. Regenerate the block with::

    python tests/test_readme_toc.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
BEGIN, END = "<!-- toc -->", "<!-- /toc -->"
TOC_HEADING = "Contents"


def headings(text: str) -> list[tuple[int, str]]:
    """Return ``(level, title)`` for every ``##`` to ``####`` heading outside code."""
    found: list[tuple[int, str]] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        match = None if fenced else re.match(r"^(#{2,4}) (.+?)\s*$", line)
        if match and match.group(2) != TOC_HEADING:
            found.append((len(match.group(1)), match.group(2)))
    return found


def slug(title: str, seen: dict[str, int]) -> str:
    """Build the anchor GitHub gives a heading: lower case, punctuation
    dropped, spaces to hyphens, and ``-1``, ``-2`` for a repeated title."""
    base = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def build_toc(text: str) -> str:
    seen: dict[str, int] = {}
    lines = [
        f"{'  ' * (level - 2)}- [{title}](#{slug(title, seen)})" for level, title in headings(text)
    ]
    return "\n".join([BEGIN, *lines, END])


def current_toc(text: str) -> str:
    return text[text.index(BEGIN) : text.index(END) + len(END)]


def test_the_table_of_contents_matches_the_headings() -> None:
    if not README.exists():  # pragma: no cover - a distribution without the README
        pytest.skip("README.md missing")
    text = README.read_text(encoding="utf-8")
    assert current_toc(text) == build_toc(text), (
        "README.md headings changed: run `python tests/test_readme_toc.py` to regenerate"
    )


def test_slugs_follow_the_github_rules() -> None:
    seen: dict[str, int] = {}
    assert slug("The `.cim` container", seen) == "the-cim-container"
    assert slug("As a library", seen) == "as-a-library"
    assert slug("Input and output", seen) == "input-and-output"
    assert slug("As a library", seen) == "as-a-library-1"


def test_headings_inside_code_fences_are_not_headings() -> None:
    text = "## Real\n```\n## Not real\n```\n### Also real\n## Contents\n##### Too deep\n"
    assert headings(text) == [(2, "Real"), (3, "Also real")]


if __name__ == "__main__":  # pragma: no cover
    source = README.read_text(encoding="utf-8")
    README.write_text(source.replace(current_toc(source), build_toc(source)), encoding="utf-8")
    print("README.md table of contents regenerated")
