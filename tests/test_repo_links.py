"""Regression test: relative links in markdown must match file names *exactly*, case included.

Windows and macOS file systems are case-insensitive, so `docs/` and `Docs/` both open
locally. GitHub and the Linux CI runner are case-sensitive, so a wrong-case link works on
a dev machine and 404s everywhere else.
"""

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [ROOT / "README.md", ROOT / "CLAUDE.md", *sorted((ROOT / "docs").rglob("*.md"))]
LINK = re.compile(r"\]\(([^)#\s]+)")  # target of [text](target), without any #anchor


def exists_exact_case(path: Path) -> bool:
    """True only if every component of `path` below ROOT matches the on-disk spelling."""
    current = ROOT
    for part in path.relative_to(ROOT).parts:
        if part not in {p.name for p in current.iterdir()}:
            return False
        current = current / part
    return True


def relative_links(md: Path) -> list[str]:
    targets = LINK.findall(md.read_text(encoding="utf-8"))
    return [t for t in targets if not re.match(r"[a-z]+:", t)]  # skip http:, mailto:, ...


@pytest.mark.parametrize("md", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve_with_exact_case(md: Path) -> None:
    for target in relative_links(md):
        # normpath, not resolve(): resolve() asks the OS, and on Windows the OS
        # silently corrects the case, which would hide exactly the bug we test for.
        resolved = Path(os.path.normpath(md.parent / target))
        assert exists_exact_case(resolved), f"{md.name}: broken or wrong-case link -> {target}"


def test_docs_folder_is_lowercase_on_disk() -> None:
    assert "docs" in {p.name for p in ROOT.iterdir()}
