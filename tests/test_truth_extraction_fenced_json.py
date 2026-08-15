"""A fenced JSON answer must reach the ladder, and nothing looser may.

Chat models wrap JSON in markdown fences by habit, and the system prompt's "Return JSON and
nothing else" does not stop them: the shipped default engine, `anthropic/claude-sonnet-4.5`,
does it on a request as small as `Return exactly {"claims": []}`.

Unstripped, `json.loads` fails at character 0 and the whole file is refused at the `json` rung.
Measured before this existed: **30 of 30 documents returned zero claims of every kind** on a real
corpus, each carrying a `batch_rejection` nothing surfaced. It read as a careful model declining
and it was a parser rejecting a correct answer, which is the more expensive of the two mistakes
because a refusal looks like evidence.

The tolerance is deliberately narrow: exactly one fence wrapping the WHOLE payload. Prose around
it, a second fence, or a trailing sentence still fails the rung, because being lenient about a
near-universal wrapper is not the same as accepting arbitrary text.
"""
from __future__ import annotations

import pytest

from recall.truth_extraction import ExtractionBatchRejected, normalize_extraction
from recall.truth_extraction._normalize import _unfence

FENCE = "`" * 3
CLAIMS = '{"claims": []}'


@pytest.mark.parametrize(
    "raw,expected,label",
    [
        (f"{FENCE}json\n{CLAIMS}\n{FENCE}", CLAIMS, "json info string"),
        (f"{FENCE}JSON\n{CLAIMS}\n{FENCE}", CLAIMS, "uppercase info string"),
        (f"{FENCE}\n{CLAIMS}\n{FENCE}", CLAIMS, "bare fence"),
        (f"  {FENCE}json\n{CLAIMS}\n{FENCE}  ", CLAIMS, "surrounding whitespace"),
        (CLAIMS, CLAIMS, "plain JSON is untouched"),
        # Shapes the first version refused, each of which reproduced the zero-claims failure in a
        # narrower form. The info string is not what makes this safe: the body still has to parse
        # as JSON and still climbs every rung, so restricting the tag to two literal spellings
        # bought nothing and cost a whole run.
        (f"{FENCE}Json\n{CLAIMS}\n{FENCE}", CLAIMS, "mixed-case info string"),
        (f"{FENCE}json\n{CLAIMS}{FENCE}", CLAIMS, "closing fence on the payload's line"),
        (f"{FENCE}json\r\n{CLAIMS}\r\n{FENCE}", CLAIMS, "CRLF line endings"),
    ],
)
def test_a_whole_payload_fence_is_unwrapped(raw: str, expected: str, label: str) -> None:
    assert _unfence(raw) == expected, label


@pytest.mark.parametrize(
    "raw,label",
    [
        (f"here you go:\n{FENCE}json\n{CLAIMS}\n{FENCE}", "prose before the fence"),
        (f"{FENCE}json\n{CLAIMS}\n{FENCE}\nhope that helps", "prose after the fence"),
        (f"{FENCE}json\n{CLAIMS}\n{FENCE}\n{FENCE}json\n{CLAIMS}\n{FENCE}", "two fences"),
        ("not json at all", "no fence, no JSON"),
        ("", "empty output"),
    ],
)
def test_anything_looser_is_left_alone_and_still_refused(raw: str, label: str) -> None:
    """The narrowness half. A lenient unwrapper would turn this rung into a suggestion."""
    with pytest.raises(ExtractionBatchRejected) as caught:
        normalize_extraction(raw, file="a.md", human_body="body", corpus_names=("a",))
    assert caught.value.rung in {"json", "top_level_shape"}, label


@pytest.mark.parametrize(
    "filler,label",
    [
        ("\n", "newlines: the first pattern's degenerate character"),
        (" ", "spaces: the SECOND pattern's, introduced by the fix for the first"),
        ("\t", "tabs"),
        ("`", "backticks: many closing-fence candidates"),
        ("x", "one long line"),
    ],
)
@pytest.mark.parametrize("closed", [False, True])
def test_the_unwrapper_is_bounded_on_degenerate_input(filler: str, label: str, closed: bool
                                                      ) -> None:
    """The rung must not become a denial of service on a degenerate reply.

    Two successive regex versions each turned one character into a hang on the INGEST path,
    where `extract_file_claims`'s guard does not reach and the HTTP timeout is long since
    satisfied:

      - `\\s*` on both sides of the newline after the info string, ambiguous with the newline
        run following it. `` ```json `` plus 20k newlines consumed 3,389 seconds of CPU
        without returning.
      - the fix for that left the lazy body ambiguous with the closing fence's indent, cleanly
        quadratic on spaces and tabs: 11.1s at 64k spaces, roughly 45 minutes at 1MB.

    So this is parametrised over the FILLER, and over closed and unterminated fences, rather
    than over the one character that happened to be reported. The first version of this test
    used newlines only, and certified a pattern that was quadratic on spaces.

    64k characters, and the bound is generous, because a timing assertion on a shared machine
    has to fail for the defect and not for a busy CPU. The old patterns blew this by 5x and
    the current implementation returns in under a millisecond.
    """
    import time

    body = f"{FENCE}json\n" + filler * 64_000
    raw = body + (f"\n{FENCE}" if closed else "x")
    started = time.perf_counter()
    _unfence(raw)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"{label}: unwrapping took {elapsed:.1f}s on 64k characters"


def test_an_unterminated_fence_is_left_alone() -> None:
    """Bounded time is not enough on its own: it must also still be refused."""
    raw = f"{FENCE}json" + "\n" * 100 + "x"
    assert _unfence(raw) == raw, "an unterminated fence is not a fence"


@pytest.mark.parametrize("reply", [None, 5, ["claims"], {"claims": []}, object()])
def test_a_reply_that_is_not_a_string_is_refused_not_raised(reply) -> None:
    """An engine that violates the port's `-> str` must cost ONE file, not the run.

    Every operation in `_unfence` is a string method, so unwrapping before a type check turned a
    `None` reply into an `AttributeError` raised OUT of the ladder: `extract_file_claims` guards
    `engine.run` alone, and the `except` clauses downstream catch only `ExtractionBatchRejected`,
    so one such reply aborted a corpus run and discarded every extraction already built. That
    was a regression introduced by the fence work; before it, `json.loads` refused these at the
    `json` rung, which is where a malformed answer belongs.
    """
    with pytest.raises(ExtractionBatchRejected) as caught:
        normalize_extraction(reply, file="a.md", human_body="body", corpus_names=("a",))
    assert caught.value.rung == "json"


def test_a_bytes_reply_still_parses() -> None:
    """The one non-str the JSON decoder accepts, and the port's likeliest violation.

    Kept working rather than refused: it worked before the fence work, `json.loads` handles it,
    and narrowing that silently would be a behaviour change smuggled in with a type guard.
    """
    claims, rejections = normalize_extraction(
        CLAIMS.encode("utf-8"), file="a.md", human_body="body", corpus_names=("a",)
    )
    assert not claims and not rejections


def test_a_fenced_answer_now_survives_the_ladder() -> None:
    """End to end, because `_unfence` passing in isolation was never the failure.

    The bug was that a correct answer never reached the rungs beyond `json`. This asserts the
    fenced form produces the same outcome as the bare form, which is the property the 30-document
    run needed and did not have.
    """
    body = "This memo supersedes old.md."
    payload = (
        '{"claims": [{"kind": "supersession", "superseded": "old", '
        '"quote": "This memo supersedes old.md."}]}'
    )
    bare_claims, bare_rejections = normalize_extraction(
        payload, file="new.md", human_body=body, corpus_names=("old",)
    )
    fenced_claims, fenced_rejections = normalize_extraction(
        f"{FENCE}json\n{payload}\n{FENCE}", file="new.md", human_body=body, corpus_names=("old",)
    )
    assert bare_claims, "the bare form must produce a claim, or this test proves nothing"
    assert [c.superseded for c in fenced_claims] == [c.superseded for c in bare_claims]
    assert len(fenced_rejections) == len(bare_rejections)
