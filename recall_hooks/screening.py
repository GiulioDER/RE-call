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

## Screen the text the SERVER will store, not the text on disk

⚠️ `normalise()` exists because an audit found a real bypass: the server strips NUL bytes before
indexing (`recall/index.py`), so a key with a NUL byte inserted into the middle of its prefix
passed a screen that read the file verbatim, and was reconstituted into a live key on the far
side. A gate that inspects a different byte sequence from the one that is stored is not a gate.
Everything here screens the normalised form, and `hosted.scan` passes it the exact text it hashed.

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
#: makes a match evidence: a four-letter AWS prefix followed by exactly 16 uppercase alphanumerics
#: is an access key id and is not a word.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # The STS session prefix is included alongside the long-lived one: it has the identical 4+16
    # shape and is what a memo recording an `aws sts assume-role`, or a pasted credentials block,
    # actually contains. The identity-ARN prefixes are deliberately excluded, being identifiers
    # rather than credentials.
    ("an AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
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
    ("an npm access token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("a PyPI API token", re.compile(r"\bpypi-AgEI[A-Za-z0-9_\-]{50,}")),
    ("a private key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("a signed token (JWT)", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.")),
)

#: 🔑 A cheap literal test that must fire before any rule can. Every pattern above begins with one
#: of these, so a text containing none of them cannot match any rule, and the 13 regex scans per
#: line can be skipped entirely.
#:
#: Measured over the 987-file store on this machine: the literals appear in 71 of 987 files, and
#: the whole-corpus screen went from 1.77s and 2.82s (two samples) to 0.54s and 0.45s, with
#: verdicts identical on every file. Note the tight prefixes matter: a loose set ("gh", "sk-")
#: hits 665 of 987 and buys almost nothing.
#:
#: ⛔ A prefilter is a guard that can silently mask every rule behind it, so a test asserts that
#: each rule's own fixture still fires through it. Add a literal here whenever you add a rule.
LITERALS: tuple[str, ...] = (
    "AKIA", "ASIA", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_",
    "sk-", "pa-", "xox", "AIza", "sk_live_", "npm_", "pypi-AgEI", "-----BEGIN", "eyJ",
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


def normalise(text: str) -> str:
    """The text as the SERVER will store it.

    NUL bytes are stripped, because `recall/index.py` strips them before indexing. Screening the
    unstripped form let a NUL-interrupted key through a gate that the server then turned back
    into a live key.
    """
    return text.replace("\x00", "")


def _is_placeholder(matched: str) -> bool:
    return any(marker in matched for marker in PLACEHOLDERS)


def secrets_in(text: str) -> list[Finding]:
    """Every structurally distinctive credential in `text`, one Finding per match, in line order.

    The matched text is never returned or logged. A scanner that prints what it found copies the
    secret into a log, a terminal and possibly a bug report, which is the same disclosure it exists
    to prevent, now in a place nobody is guarding.

    ⛔ **Every match on a line is examined, not just the first.** An audit found that the earlier
    `pattern.search` version returned NOTHING for a line holding a well-known documentation key
    followed by a live one: the example, being first, disarmed the rule for everything after it.
    "Here is the example, here is mine" is ordinary memo prose, so this was a live bypass rather
    than a crafted one, and it defeated the exact invariant `PLACEHOLDERS` above claims to hold.

    Lines are split on `\\n` alone rather than with `splitlines()`, which also breaks on `\\x0b`,
    `\\x0c`, `\\x85`, `\\u2028` and `\\u2029`. An editor counts `\\n`, and a line number offered
    for pasting after a colon has to mean the editor's line.
    """
    text = normalise(text)
    if not any(literal in text for literal in LITERALS):
        return []
    findings: list[Finding] = []
    for number, line in enumerate(text.split("\n"), start=1):
        for rule, pattern in RULES:
            for match in pattern.finditer(line):
                if not _is_placeholder(match.group(0)):
                    findings.append(Finding(line=number, rule=rule))
    return findings


def screen_file(path: Path) -> list[Finding]:
    """Findings for one file, read here.

    ⚠️ Prefer `secrets_in` on text you have ALREADY read when you also intend to upload that
    file: reading it here and again at upload time leaves a window in which the content can
    change, and `hosted.scan` therefore screens the buffer it hashed rather than calling this.
    This entry point is for standalone use over a corpus nobody is uploading.

    An unreadable file yields no findings, and the CALLER must not read that as "clean": see
    `hosted.scan`, which records an unreadable file rather than dropping it.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return []
    return secrets_in(text)


def screen(changes: list) -> tuple[list, dict[str, list[Finding]]]:
    """Split planned uploads into those that may leave the machine and those that may not.

    Takes and returns `hosted.Change` objects, typed loosely so this module imports nothing from
    `hosted` and can be run over any corpus on its own. A change carrying already-read `text`
    is screened from that buffer; otherwise the file is read here.

    Order is preserved for the allowed ones, because this must be a pure filter over its input:
    the caller rebuilds a mapping from the result and only then hands it to `plan`, which is what
    batches against the server's limits.
    """
    allowed: list = []
    withheld: dict[str, list[Finding]] = {}
    for change in changes:
        text = getattr(change, "text", None)
        findings = secrets_in(text) if text is not None else screen_file(Path(change.path))
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
