"""PostToolUse hook: ruff-format + autofix any .py file an agent just wrote.

No-ops silently until the project venv has ruff installed,
so it never blocks work during bootstrap.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUFF = ROOT / ".venv" / "Scripts" / "ruff.exe"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith(".py") or not RUFF.exists():
        return 0

    for args in (["check", "--fix", "--quiet"], ["format", "--quiet"]):
        subprocess.run([str(RUFF), *args, file_path], cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
