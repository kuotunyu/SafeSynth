from scripts.audit_phase1_handoff import (
    _identity_violations,
    _parse_git_log,
)


def test_parse_git_log_preserves_multiline_body() -> None:
    log = (
        "abc\x1fkuotunyu\x1f61350295+kuotunyu@users.noreply.github.com"
        "\x1fkuotunyu\x1f61350295+kuotunyu@users.noreply.github.com"
        "\x1fsubject\n\nbody line\x1e\n"
    )

    commits = _parse_git_log(log)

    assert commits[0]["sha"] == "abc"
    assert commits[0]["body"] == "subject\n\nbody line"


def test_identity_violations_checks_author_and_committer() -> None:
    clean = {
        "sha": "clean",
        "author_name": "kuotunyu",
        "author_email": "61350295+kuotunyu@users.noreply.github.com",
        "committer_name": "kuotunyu",
        "committer_email": "61350295+kuotunyu@users.noreply.github.com",
        "body": "clean",
    }
    wrong_committer = {**clean, "sha": "wrong", "committer_name": "bot"}

    assert _identity_violations([clean, wrong_committer]) == [
        {
            "sha": "wrong",
            "author_name": "kuotunyu",
            "author_email": "61350295+kuotunyu@users.noreply.github.com",
            "committer_name": "bot",
            "committer_email": "61350295+kuotunyu@users.noreply.github.com",
        }
    ]
