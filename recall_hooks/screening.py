"""What must not leave the machine, decided before anything does.

⛔ **This does NOT rewrite anything, and that is the design rather than a limitation.**
`docs/SECURITY_MODEL.md` states RE-call's position plainly: *"RE-call does not redact, encrypt, or
classify any of it: a chunk goes in exactly as written and comes back exactly as written"*, and
its remedy for a corpus holding something sensitive is that *"that content should not be indexed
in the first place"*. A redactor would contradict that, would break `hosted.py`'s rule that nothing
under a memory root is ever modified, and would silently damage the memory it is meant to protect:
a memo whose credential has been replaced by `***` is a memo whose meaning is now wrong, and
nobody would ever know, because the original is gone.

So this REFUSES a file and names why. The file stays exactly where it is, the rest of the sync
proceeds, and the person decides. Withholding is recoverable; rewriting is not.

**Nothing here imports `recall`, or anything outside the standard library.** Same rule as the rest
of the package.

## Why the rule set is small on purpose

This screens **prose about engineering**, which is the worst possible input for a naive secret
scanner. A memory corpus legitimately contains example DSNs, host names, redacted tokens and
whole paragraphs discussing credentials. `CLAUDE.md` in this very repository contains
`postgresql://user:pw@localhost/recall`. A scanner that flags those is not cautious, it is useless:
it fires on every sync, the person learns the warning means nothing, and the one real finding
arrives inside noise they have already been trained to skip.

Every rule here therefore matches a **structurally distinctive live credential**: a vendor prefix
plus a length, or a PEM header. Those shapes do not occur by accident in prose. Things
deliberately NOT screened, because the false-positive cost exceeds the benefit:

* bare high-entropy strings (every git sha, uuid and base64 blob in the corpus)
* database URLs (see the `CLAUDE.md` example above)
* anything matched only by a nearby word like "password" or "token"

The gap is stated rather than hidden: **a secret with no distinctive shape gets through.** This
raises the floor; it is not a guarantee, and `docs/SECURITY_MODEL.md` remains the honest statement
of the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: `(rule name, pattern)`. The name is what a person is shown, so it says what was found rather
#: than which regex fired.
#:
#: Each pattern is anchored on a vendor's own prefix and its documented length. A prefix is what
#: makes a match evidence: `AKIA` followed by exactly 16 uppercase alphanumerics is an AWS access
#: key id and is not a word.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("a GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("an Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{24,}")),
    # ⚠️ The lookahead is not tidiness. Without it an Anthropic key matches this rule too, and
    # a person is told their memo holds an OpenAI key when it does not. The gate would still
    # refuse the file, so the mistake is invisible to a test that only asserts "withheld":
    # it was caught by asserting the exact rule LIST, which is why that test asserts equality
    # rather than membership.
    ("an OpenAI API key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_\-]{32,}")),
    ("a Voyage API key", re.compile(r"\bpa-[A-Za-z0-9_\-]{32,}\b")),
    ("a Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("a Stripe secret key", re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b")),
    ("a private key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("a signed token (JWT)", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.")),
)

#: Substrings that mark a match as an EXAMPLE rather than a credential.
#:
#: 🔑 Checked against the matched text itself, never against the surrounding line. A line reading
#: "the key below is not real" sitting above a live key would otherwise disarm the rule, and that
#: sentence is exactly what someone writes when pasting one.
PLACEHOLDERS: tuple[str, ...] = (
    "xxxx",
    "XXXX",
    "....",
    "example",
    "EXAMPLE",
    "redacted",
    "REDACTED",
    "your-",
    "YOUR-",
    "placeholder",
    "<",
)


@dataclass(frozen=True)
class Finding:
    """One reason a file may not leave the machine."""

    line: int
    """1-based, so it can be pasted after a colon and opened."""

    rule: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.rule}"


def _is_placeholder(matched: str) -> bool:
    return any(marker in matched for marker in PLACEHOLDERS)


def secrets_in(text: str) -> list[Finding]:
    """Every structurally distinctive credential in `text`, in line order.

    The matched text is never returned or logged. A scanner that prints what it found copies the
    secret into a log, a terminal and possibly a bug report, which is the same disclosure it exists
    to prevent, now in a place nobody is guarding.
    """
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in RULES:
            match = pattern.search(line)
            if match is not None and not _is_placeholder(match.group(0)):
                findings.append(Finding(line=number, rule=rule))
    return findings


def screen_file(path: Path) -> list[Finding]:
    """Findings for one file. An unreadable file yields none: it is `scan`'s job to notice that,
    and a read error must not be reported to a person as "this file contains a secret"."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return secrets_in(text)


def screen(changes: list) -> tuple[list, dict[str, list[Finding]]]:
    """Split planned uploads into those that may leave the machine and those that may not.

    Takes and returns `hosted.Change` objects, typed loosely so this module imports nothing from
    `hosted` and can be run over any corpus on its own.

    Order is preserved for the allowed ones, because `plan` has already batched them against the
    server's limits and reordering would invalidate that.
    """
    allowed: list = []
    withheld: dict[str, list[Finding]] = {}
    for change in changes:
        findings = screen_file(Path(change.path))
        if findings:
            withheld[change.name] = findings
        else:
            allowed.append(change)
    return allowed, withheld


def summarise(withheld: dict[str, list[Finding]]) -> str:
    """One line per withheld file, for a person. Never includes the matched text."""
    lines = []
    for name in sorted(withheld):
        detail = "; ".join(str(finding) for finding in withheld[name])
        lines.append(f"  {name}: {detail}")
    return "\n".join(lines)
