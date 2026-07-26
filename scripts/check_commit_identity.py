"""Reject commit metadata that would add another GitHub contributor."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXPECTED_NAME = "kuotunyu"
EXPECTED_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
COAUTHOR_PATTERN = re.compile(r"^Co-Authored-By\s*:", re.IGNORECASE | re.MULTILINE)


def validate_commit(
    message: str,
    *,
    author_name: str,
    author_email: str,
) -> list[str]:
    errors: list[str] = []
    if COAUTHOR_PATTERN.search(message):
        errors.append("Co-Authored-By trailers are forbidden in this repository")
    if author_name != EXPECTED_NAME:
        errors.append(
            f"repo-local user.name must be {EXPECTED_NAME!r}, got {author_name!r}"
        )
    if author_email != EXPECTED_EMAIL:
        errors.append(
            f"repo-local user.email must be {EXPECTED_EMAIL!r}, got {author_email!r}"
        )
    return errors


def _local_config(key: str) -> str:
    result = subprocess.run(
        ["git", "config", "--local", "--get", key],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_commit_identity.py COMMIT_MSG_FILE")
    message = Path(sys.argv[1]).read_text(encoding="utf-8")
    errors = validate_commit(
        message,
        author_name=_local_config("user.name"),
        author_email=_local_config("user.email"),
    )
    if errors:
        for error in errors:
            print(f"commit blocked: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
