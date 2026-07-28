"""A corpus file name printed to a terminal must not be able to drive the terminal.

The CLI renders corpus-controlled strings — `provenance.file`, `superseded_by`, lint issue paths
— straight into `print()`. A file name may contain almost any byte except `/` and NUL, ANSI escape
sequences included, and a terminal executes those: `\\x1b[2K\\r` erases the line just written,
`\\x1b[1A` moves up over it. A corpus can therefore make `recall lint` print a clean report while
hiding the very issues it found, or overwrite a `0 errors` summary onto a run that had errors.

The same class as the `advice` injection — untrusted text reaching a channel that interprets it —
with a terminal as the interpreter instead of a model. The chunk PREVIEW is the same problem: it
is corpus content by definition.

This is a lower-severity path than the MCP one (a local operator, usually their own corpus), but
the fix is one call and the failure is silent, which is the combination worth closing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from recall.cli import _print_result
from recall.trust import terminal_safe
from recall.types import (
    Chunk,
    Provenance,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)

#: Erase-line + cursor-up: enough to scroll away the line that was just printed.
ANSI_ERASE = "\x1b[2K\r\x1b[1A"


def _result_with(file: str, *, superseded_by: str | None = None, text: str = "body") -> TrustedResult:
    hit = TrustedHit(
        chunk=Chunk(id="1", source=f"/corpus/{file}", text=text, metadata={"file": file}),
        cosine=0.9,
        confidence=0.9,
        verdict="ok",
        provenance=Provenance(source=f"/corpus/{file}", file=file, ord=0,
                              indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=superseded_by),
    )
    return TrustedResult(
        query="q", hits=[hit], abstained=False, reason="", calibrated=True, gap_warning=False,
        staleness=StalenessReport(stale=False, newest_indexed_at=None, age=None,
                                  max_age=timedelta(days=1)),
    )


def test_an_escape_sequence_in_a_file_name_is_not_printed(capsys):
    _print_result(_result_with(f"notes{ANSI_ERASE}.md"))

    out = capsys.readouterr().out
    assert "\x1b" not in out, "an ANSI escape from the corpus reached the terminal"


def test_an_escape_sequence_in_a_successor_name_is_not_printed(capsys):
    _print_result(_result_with("stale.md", superseded_by=f"v2{ANSI_ERASE}.md"))

    out = capsys.readouterr().out
    assert "\x1b" not in out


def test_an_escape_sequence_in_the_chunk_preview_is_not_printed(capsys):
    """The preview is corpus CONTENT, so it is the most obviously attacker-controlled of the three."""
    _print_result(_result_with("notes.md", text=f"hello{ANSI_ERASE}world"))

    out = capsys.readouterr().out
    assert "\x1b" not in out


def test_an_ordinary_result_still_prints_readably(capsys):
    _print_result(_result_with("rate_limits_v2.md"))

    out = capsys.readouterr().out
    assert "rate_limits_v2.md" in out
    assert "ok" in out


class TestTerminalSafe:
    def test_strips_ansi_and_control_bytes(self):
        assert terminal_safe(f"a{ANSI_ERASE}b") == "ab"
        assert "\x07" not in terminal_safe("bell\x07here")

    def test_keeps_ordinary_text_identical(self):
        """Unlike `safe_ref` this adds no quotes — it is a filter, not a renderer."""
        assert terminal_safe("rate_limits_v2.md") == "rate_limits_v2.md"

    def test_handles_none(self):
        assert terminal_safe(None) == ""
