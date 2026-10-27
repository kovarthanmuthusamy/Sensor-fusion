"""SessionStart hook: print where we left off. Stdout is added to Claude's context."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "(no git repo yet)"
    except (OSError, subprocess.SubprocessError):
        return "(git unavailable)"


def last_learning_entry() -> str:
    entries = sorted(p for p in (ROOT / "docs" / "learning").glob("[0-9]*.md"))
    if not entries:
        return "(none yet)"
    title = entries[-1].read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    return f"{entries[-1].name} — {title}"


def open_todos(limit: int = 5) -> list[str]:
    progress = ROOT / "docs" / "PROGRESS.md"
    if not progress.exists():
        return []
    todos = [
        ln.strip()
        for ln in progress.read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith("- [ ]")
    ]
    return todos[:limit]


def main() -> None:
    # Windows consoles default to cp1252; markdown contains unicode (→, —).
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== Sensor Fusion — session status ===")
    print(f"Branch:          {git_branch()}")
    print(f"Last log entry:  {last_learning_entry()}")
    print("Next open TODOs (docs/PROGRESS.md):")
    for todo in open_todos() or ["  (none)"]:
        print(f"  {todo}")


if __name__ == "__main__":
    main()
