from scripts.check_commit_identity import validate_commit

NAME = "kuotunyu"
EMAIL = "61350295+kuotunyu@users.noreply.github.com"


def test_clean_identity_and_message_pass() -> None:
    assert validate_commit(
        "feat: clean commit\n",
        author_name=NAME,
        author_email=EMAIL,
    ) == []


def test_coauthor_trailer_is_rejected_case_insensitively() -> None:
    errors = validate_commit(
        "feat: bad\n\nco-authored-by: Bot <bot@example.com>\n",
        author_name=NAME,
        author_email=EMAIL,
    )

    assert errors == ["Co-Authored-By trailers are forbidden in this repository"]


def test_wrong_author_or_email_is_rejected() -> None:
    errors = validate_commit(
        "feat: bad identity\n",
        author_name="bot",
        author_email="bot@example.com",
    )

    assert len(errors) == 2
    assert "user.name" in errors[0]
    assert "user.email" in errors[1]
