"""Audit contributor identity and Phase 1 blockers before publication."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT

EXPECTED_NAME = "kuotunyu"
EXPECTED_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
EXPECTED_H6_SHA256 = "0e385d857067aa293c5e3d0dd43ad84b4141ff9bac5c8d4aefed187ee9c45739"
COAUTHOR_PATTERN = re.compile(r"^Co-Authored-By\s*:", re.IGNORECASE | re.MULTILINE)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_git_log(text: str) -> list[dict[str, str]]:
    commits: list[dict[str, str]] = []
    for raw_record in text.split("\x1e"):
        record = raw_record.strip()
        if not record:
            continue
        fields = record.split("\x1f", 5)
        if len(fields) != 6:
            raise ValueError("Unexpected git-log record format")
        sha, author_name, author_email, committer_name, committer_email, body = fields
        commits.append(
            {
                "sha": sha,
                "author_name": author_name,
                "author_email": author_email,
                "committer_name": committer_name,
                "committer_email": committer_email,
                "body": body,
            }
        )
    return commits


def _identity_violations(commits: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            key: commit[key]
            for key in (
                "sha",
                "author_name",
                "author_email",
                "committer_name",
                "committer_email",
            )
        }
        for commit in commits
        if (
            commit["author_name"] != EXPECTED_NAME
            or commit["author_email"] != EXPECTED_EMAIL
            or commit["committer_name"] != EXPECTED_NAME
            or commit["committer_email"] != EXPECTED_EMAIL
        )
    ]


def _valid_h6_signoff(path: Path) -> bool:
    if not path.exists():
        return False
    signoff = _read_json(path)
    return (
        signoff.get("approved") is True
        and signoff.get("approved_by") == EXPECTED_NAME
        and signoff.get("grid_sha256") == EXPECTED_H6_SHA256
        and isinstance(signoff.get("real_helmet_count"), int)
    )


def audit(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    commits = _parse_git_log(
        _git(
            root,
            "log",
            "--all",
            "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e",
        )
    )
    identity_violations = _identity_violations(commits)
    coauthor_commits = [
        commit["sha"] for commit in commits if COAUTHOR_PATTERN.search(commit["body"])
    ]
    remotes = [line for line in _git(root, "remote", "-v").splitlines() if line.strip()]
    local_identity = {
        "name": _git(root, "config", "--local", "user.name").strip(),
        "email": _git(root, "config", "--local", "user.email").strip(),
    }
    hooks_path = _git(root, "config", "--local", "core.hooksPath").strip()
    hook_index_entry = _git(
        root,
        "ls-files",
        "--stage",
        ".githooks/commit-msg",
    ).strip()
    contributor_hook_valid = (
        hooks_path == ".githooks"
        and hook_index_entry.startswith("100755 ")
        and (root / ".githooks" / "commit-msg").exists()
    )

    h6_grid = root / "reports" / "figures" / "h6_hard_negative_candidates.png"
    h6_sha256 = _sha256(h6_grid) if h6_grid.exists() else None
    h6_grid_valid = h6_sha256 == EXPECTED_H6_SHA256
    h6_signoff = root / "reports" / "hard_negative_signoff.json"
    h6_approved = _valid_h6_signoff(h6_signoff)

    def _h4_is_consistent(report: dict) -> bool:
        return (
            isinstance(report.get("auc"), float)
            and isinstance(report.get("max_auc_for_scaleup"), float)
            and report.get("passed") is (report["auc"] <= report["max_auc_for_scaleup"])
        )

    # The M11 report is the pre-registered milestone artifact and stays the
    # consistency anchor. The M13 rerun on the delivered pool is what every
    # result table has to display, so quoting M11 alone would hand the reader a
    # superseded number under the words "every result table must display this".
    h4_path = root / "reports" / "h4_artifact_gate.json"
    h4 = _read_json(h4_path)
    h4_result_consistent = _h4_is_consistent(h4)

    h4_latest_path = root / "reports" / "h4_artifact_gate_m13.json"
    h4_latest = _read_json(h4_latest_path) if h4_latest_path.exists() else None
    if h4_latest is not None and not _h4_is_consistent(h4_latest):
        h4_result_consistent = False
    h4_current = h4_latest if h4_latest is not None else h4

    h4_passed = h4_result_consistent and bool(h4_current["passed"])

    ledger_path = root / "reports" / "filter_ledger.json"
    ledger = _read_json(ledger_path)
    ledger_consistent = (
        ledger.get("n_total") == ledger.get("n_pass", 0) + ledger.get("n_reject", 0)
        and all(bool(value) for value in ledger.get("checks", {}).values())
    )

    local_identity_valid = local_identity == {
        "name": EXPECTED_NAME,
        "email": EXPECTED_EMAIL,
    }
    contributor_identity_valid = not identity_violations and not coauthor_commits
    integrity_checks = {
        "all_authors_and_committers_are_kuotunyu": not identity_violations,
        "no_coauthored_by_trailers": not coauthor_commits,
        "repo_local_identity_is_kuotunyu_noreply": local_identity_valid,
        "single_contributor_commit_hook_is_active": contributor_hook_valid,
        "no_git_remote_before_user_request": not remotes,
        "h6_grid_exists_and_sha256_matches": h6_grid_valid,
        "h4_result_is_internally_consistent": h4_result_consistent,
        "m12_filter_ledger_is_internally_consistent": ledger_consistent,
    }
    blockers: list[str] = []
    known_limitations: list[str] = []
    if not h6_approved:
        blockers.append("M9/H6 requires kuotunyu's review and exact-grid signoff.")
    if not h4_passed:
        # ADR-011: the H4 verdict is UNCHANGED and is not claimed to pass. What
        # changed is the consequence. After nine synthesis routes and eighteen
        # labeler iterations failed to move it, the failure is carried forward as a
        # published limitation instead of blocking indefinitely, and the generation
        # cap becomes "1x, never 2x" rather than "unbounded once passed".
        provenance = (
            f"delivered pool, {h4_latest.get('n_examples', '?')} patches"
            if h4_latest is not None
            else "M11 pre-registration"
        )
        known_limitations.append(
            f"H4 AUC {h4_current['auc']:.4f} ({provenance}) exceeds the "
            f"{h4_current['max_auc_for_scaleup']:.2f} maximum: paste artifacts are "
            "detectable. Accepted as a reported limitation per ADR-011; generation is "
            "capped at 1x and 2x is forbidden. Every result table must display this AUC. "
            f"M11 pre-registration measured {h4['auc']:.4f} on 300 images."
        )

    # H6 gates whether hard-negative material may be used at all; H4 only caps scale.
    permitted_scale = "none" if not h6_approved else ("2x" if h4_passed else "1x")

    return {
        "head_commit": _git(root, "rev-parse", "HEAD").strip(),
        "expected_github_identity": {
            "name": EXPECTED_NAME,
            "email": EXPECTED_EMAIL,
        },
        "commit_count_all_refs": len(commits),
        "unique_author_identities": sorted(
            {
                f"{commit['author_name']} <{commit['author_email']}>"
                for commit in commits
            }
        ),
        "unique_committer_identities": sorted(
            {
                f"{commit['committer_name']} <{commit['committer_email']}>"
                for commit in commits
            }
        ),
        "identity_violations": identity_violations,
        "coauthor_trailer_commits": coauthor_commits,
        "local_identity": local_identity,
        "commit_hook": {
            "core_hooks_path": hooks_path,
            "tracked_index_entry": hook_index_entry,
        },
        "remotes": remotes,
        "h6": {
            "grid_sha256": h6_sha256,
            "expected_grid_sha256": EXPECTED_H6_SHA256,
            "approved": h6_approved,
            "signoff_path": str(h6_signoff.relative_to(root)),
        },
        "h4": {
            "auc": h4_current["auc"],
            "auc_ci95": h4_current.get("auc_ci95"),
            "n_examples": h4_current.get("n_examples"),
            "max_auc_for_scaleup": h4_current["max_auc_for_scaleup"],
            "passed": h4_passed,
            "disposition": "failed_and_accepted_per_adr_011" if not h4_passed else "passed",
            "source": (
                "h4_artifact_gate_m13.json"
                if h4_latest is not None
                else "h4_artifact_gate.json"
            ),
            "m11_preregistration_auc": h4["auc"],
        },
        "m12": {
            "n_total": ledger["n_total"],
            "n_pass": ledger["n_pass"],
            "n_reject": ledger["n_reject"],
        },
        "integrity_checks": integrity_checks,
        "integrity_passed": all(integrity_checks.values()),
        "github_identity_ready": contributor_identity_valid and local_identity_valid,
        "prepublication_state_safe": (
            contributor_identity_valid and local_identity_valid and not remotes
        ),
        "scale_up_allowed": h6_approved and h4_passed,
        "permitted_synthetic_scale": permitted_scale,
        "blockers": blockers,
        "known_limitations": known_limitations,
    }


def _render_markdown(result: dict[str, Any]) -> str:
    def mark(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    lines = [
        "# Phase 1 handoff preflight",
        "",
        f"- Audited HEAD: `{result['head_commit']}`",
        f"- Commits across all local refs: **{result['commit_count_all_refs']}**",
        (
            "- GitHub contributor identity ready: "
            f"**{mark(result['github_identity_ready'])}**"
        ),
        (
            "- Pre-publication state (identity + no remote): "
            f"**{mark(result['prepublication_state_safe'])}**"
        ),
        f"- Unrestricted scale-up (H4 passed): **{mark(result['scale_up_allowed'])}**",
        f"- Permitted synthetic scale: **{result['permitted_synthetic_scale']}**",
        "",
        "## Integrity checks",
        "",
    ]
    for name, passed in result["integrity_checks"].items():
        lines.append(f"- {mark(passed)} — `{name}`")
    lines.extend(
        [
            "",
            "## Contributor evidence",
            "",
            (
                "- Authors: "
                + ", ".join(f"`{item}`" for item in result["unique_author_identities"])
            ),
            (
                "- Committers: "
                + ", ".join(f"`{item}`" for item in result["unique_committer_identities"])
            ),
            f"- `Co-Authored-By:` commits: {len(result['coauthor_trailer_commits'])}",
            f"- Configured remotes: {len(result['remotes'])}",
            "",
            "## Phase 1 gates",
            "",
            (
                f"- H6 exact-grid approval: **{mark(result['h6']['approved'])}** "
                f"(SHA256 `{result['h6']['grid_sha256']}`)"
            ),
            (
                f"- H4 scale-up gate: **{mark(result['h4']['passed'])}** "
                f"(AUC {result['h4']['auc']:.4f}; maximum "
                f"{result['h4']['max_auc_for_scaleup']:.2f})"
            ),
            (
                f"- M12 ledger: {result['m12']['n_total']} = "
                f"{result['m12']['n_pass']} pass + "
                f"{result['m12']['n_reject']} reject"
            ),
            "",
            "## Blocking actions",
            "",
        ]
    )
    lines.extend(f"- {blocker}" for blocker in result["blockers"] or ["(none)"])
    lines.extend(["", "## Known limitations carried forward", ""])
    lines.extend(f"- {item}" for item in result["known_limitations"] or ["(none)"])
    lines.extend(
        [
            "",
            "A failed H6 line is a hard blocker: do not create a signoff on the user's",
            "behalf. A failed H4 line is NOT a blocker any more — per ADR-011 it is an",
            "accepted, published limitation that caps generation at 1x and forbids 2x.",
            "It is still a failure and must never be reported as a pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = audit()
    reports = PROJECT_ROOT / "reports"
    (reports / "phase1_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (reports / "phase1_preflight.md").write_text(
        _render_markdown(result),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
