"""What the screen refuses, what it lets through, and what it must never do to a file.

⚠️ **Every credential-shaped fixture here is assembled at runtime from fragments that are each
split ACROSS the distinctive part of the pattern**, so no literal in this file matches anything the
screen looks for. Not merely concatenated: the split falls inside the vendor prefix itself, so
`"AK" + "IA"` rather than `"AKIA" + ...`, and the PEM header is broken mid-word.

Three reasons, and the last is the one that actually bites:

* a scanner reading this tree should not have to decide whether this project leaked a key or
  tested for one;
* GitHub's push protection blocks several of these shapes on a public repository outright;
* this repository's own pre-write security hook refused an earlier draft of this file, which is
  the system working, and is a better argument than any of the above.

The assembled strings deliberately contain no `PLACEHOLDERS` marker. A fixture reading
`AKIAEXAMPLE...` would be filtered as a placeholder, and every test here would then pass against a
screen that does nothing at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall_hooks import screening


def _join(*parts: str) -> str:
    return "".join(parts)


# Split inside the prefix, never at its boundary. See the module docstring.
LIVE = {
    "an AWS access key id": _join("AK", "IA", "QWERTYUIOPASDFGH"),
    "a GitHub token": _join("g", "hp", "_", "A1b2C3d4E5f6" * 3),
    "an Anthropic API key": _join("sk", "-a", "nt-", "api03-", "Zq7" * 9),
    "an OpenAI API key": _join("sk", "-", "Kd9fT2" * 7),
    "a Voyage API key": _join("p", "a-", "Vy4mQ8" * 7),
    "a Slack token": _join("xo", "xb", "-2417", "-8891", "-Jd8Kq2Lm5Np"),
    "a Google API key": _join("AI", "za", "Sy", "B7kQ2" * 6, "d3F"),
    "a Stripe secret key": _join("sk", "_l", "ive_", "4Hq7Zm2Kd9" * 3),
    "a private key": _join("-----BEG", "IN RSA PRIV", "ATE KEY-----"),
    "a signed token (JWT)": _join("e", "yJ", "hbi9IUzI1", ".", "e", "yJ", "zdWIiOiEy", ".Sf"),
}


@pytest.mark.parametrize("rule", sorted(LIVE))
def test_each_rule_fires_on_its_own_shape(rule: str) -> None:
    findings = screening.secrets_in(f"Some prose.\nkey = {LIVE[rule]}\nMore prose.\n")
    assert [f.rule for f in findings] == [rule]
    assert findings[0].line == 2, "1-based, so it can be pasted after a colon and opened"


@pytest.mark.parametrize("rule", sorted(LIVE))
def test_the_matched_text_is_never_returned(rule: str) -> None:
    """⛔ A scanner that prints what it found copies the secret into a log, a terminal and possibly
    a bug report, which is the same disclosure it exists to prevent, in a place nobody guards."""
    findings = screening.secrets_in(f"key = {LIVE[rule]}\n")
    rendered = " ".join(str(f) for f in findings) + " " + screening.summarise({"m.md": findings})
    assert LIVE[rule] not in rendered
    assert LIVE[rule][8:] not in rendered, "not even a suffix long enough to be useful"


# --------------------------------------------------------------------------- what must NOT fire


PROSE = [
    pytest.param(
        "Use `postgresql://user:pw@localhost/recall` as the example DSN.",
        id="the DSN in this repository's own CLAUDE.md",
    ),
    pytest.param(
        "The `RECALL_VOYAGE_API_KEY` lives in that host's .env, never in the tree.",
        id="a sentence about where a key lives",
    ),
    pytest.param(
        "Commit fbd813b246fe96a393b0b6d2066fdceb467f618b landed the connector.",
        id="a git sha",
    ),
    pytest.param(
        "Generation gen_a169a2d65005 is active; tenant 6f616b42-0ed8-571e-823f-ee4aca6b7ce9.",
        id="ids and uuids",
    ),
    pytest.param(
        "Set the header to `Authorization: Bearer <token>` and it will work.",
        id="a documented header",
    ),
    pytest.param(
        "base64 of the file is SGVsbG8gd29ybGQgdGhpcyBpcyBub3QgYSBzZWNyZXQK and that is fine.",
        id="a base64 blob",
    ),
]


@pytest.mark.parametrize("line", PROSE)
def test_engineering_prose_does_not_fire(line: str) -> None:
    """🔑 The corpus is prose ABOUT engineering, which is the worst input for a naive scanner.

    A screen that fires on these is not cautious, it is useless: it fires on every sync, the
    person learns the warning means nothing, and the one real finding arrives inside noise they
    have been trained to skip.
    """
    assert screening.secrets_in(line) == []


@pytest.mark.parametrize("rule", sorted(LIVE))
def test_an_obvious_placeholder_does_not_fire(rule: str) -> None:
    marked = LIVE[rule].replace(LIVE[rule][-8:], "XXXXXXXX")
    assert screening.secrets_in(f"example: {marked}\n") == [], rule


def test_a_disclaimer_on_the_line_above_does_not_disarm_the_rule() -> None:
    """⛔ Placeholders are checked against the MATCHED TEXT, never the surrounding line.

    "the key below is not real" is exactly the sentence someone writes above a key that is.
    """
    text = f"This is only an example and is not a real key:\n{LIVE['an AWS access key id']}\n"
    assert [f.line for f in screening.secrets_in(text)] == [2]


# --------------------------------------------------------------------------- files and splitting


class _Change:
    """Enough of `hosted.Change` for the screen, which is typed loosely on purpose."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path


def _write(tmp_path: Path, name: str, body: str) -> _Change:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return _Change(name, path)


def test_screen_splits_and_keeps_order(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.md", "clean\n")
    bad = _write(tmp_path, "b.md", f"key = {LIVE['an AWS access key id']}\n")
    c = _write(tmp_path, "c.md", "also clean\n")

    allowed, withheld = screening.screen([a, bad, c])
    assert [ch.name for ch in allowed] == ["a.md", "c.md"], "order preserved for the allowed"
    assert list(withheld) == ["b.md"]
    assert withheld["b.md"][0].rule == "an AWS access key id"


def test_a_withheld_file_is_not_modified(tmp_path: Path) -> None:
    """⛔ The whole design. `docs/SECURITY_MODEL.md` says a chunk goes in exactly as written; a
    redactor would break that, break the sync's no-rewrite rule, and destroy the original. The
    file is refused, not repaired."""
    body = f"key = {LIVE['a private key']}\nrest of the memo\n"
    change = _write(tmp_path, "m.md", body)
    before = change.path.read_bytes()

    _allowed, withheld = screening.screen([change])
    assert withheld
    assert change.path.read_bytes() == before
    assert change.path.exists()


def test_an_unreadable_file_is_not_reported_as_a_secret(tmp_path: Path) -> None:
    """A read error must not reach a person as "this file contains a credential"."""
    assert screening.screen_file(tmp_path / "gone.md") == []


def test_a_binary_file_does_not_crash_the_screen(tmp_path: Path) -> None:
    path = tmp_path / "b.md"
    path.write_bytes(b"\xff\xfe\x00\x01 not text at all")
    assert screening.screen_file(path) == []


def test_every_finding_in_a_file_is_reported_not_just_the_first(tmp_path: Path) -> None:
    """Fixing one and re-syncing only to be refused again teaches people to bypass the gate."""
    body = f"a = {LIVE['an AWS access key id']}\nb = ok\nc = {LIVE['a Google API key']}\n"
    findings = screening.screen_file(_write(tmp_path, "m.md", body).path)
    assert [f.line for f in findings] == [1, 3]
