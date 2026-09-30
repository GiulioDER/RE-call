"""W4: T-1's content gate and its v2 (relative, anchored) week render.

Round two, 2026-09-28. 90.6% of the first Textual Full's Searches took the code route, where the
route gate switches T-1 off, so T-1 barely reached the traffic it was built for. The content gate
resolves on every route and protects code by what the text looks like instead; the v2 render keeps
week expressions relative, which AML's LoCoMo and LongMemEval judge requires. Both are off by
default: the served C9 renders exactly as before.

Each test names the mutation of the production code it was watched to fail on (the red proof).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import re
from string import ascii_lowercase
import time

import pytest

from recall_aml.temporal_render import _key, looks_like_code, resolve_relative_times, resolve_text
from recall_aml.variants import variant
from tests.test_aml_relative_dates_and_adjacency import (
    CODE_QUERY,
    _add,
    _item,
    _search,
)
from tests.test_aml_specialist_fusion import _service

ANCHOR = date(2023, 5, 8)  # a Monday


@pytest.mark.parametrize(
    ("phrase", "resolution"),
    [
        ("last week", "[= the week before 2023-05-08]"),
        ("this week", "[= the week of 2023-05-08]"),
        ("next week", "[= the week after 2023-05-08]"),
        ("last weekend", "[= the weekend before 2023-05-08]"),
        ("next weekend", "[= the weekend after 2023-05-08]"),
        ("two weeks ago", "[= 2 weeks before 2023-05-08]"),
        ("a week ago", "[= 1 week before 2023-05-08]"),
        # Not week-based: exactly as v1.
        ("yesterday", "[= 2023-05-07]"),
        ("3 days ago", "[≈ 2023-05-05]"),
        ("last month", "[= 2023-04]"),
        ("last Friday", "[= 2023-05-05]"),
    ],
)
def test_v2_keeps_week_expressions_relative_and_everything_else_as_v1(phrase: str, resolution: str) -> None:
    """Invariant: under v2 a week expression is rendered relative to the item's own day, and no
    other expression changes.

    Red proof: `_resolve_v2_week` returning None unconditionally sends every week expression back
    to the v1 form (``[week of 2023-05-01]``) and fails the equality for the week rows.
    """
    assert resolve_text(f"We met {phrase}.", ANCHOR, render="v2") == f"We met {phrase} {resolution}."


def test_code_punctuation_protects_a_phrase_but_prose_punctuation_does_not() -> None:
    """Invariant: under the content gate a phrase glued to code (``date.today()``,
    ``today.strftime``, ``$yesterday``, ``yesterday=True``) is left as written, while a phrase
    ending a sentence or clause is still resolved.

    Red proof: counting ``.`` after a phrase as code whatever follows it (dropping the
    identifier condition in `_glued_to_code`) leaves "We met yesterday." unresolved and fails the
    first assertion.
    """
    assert resolve_text("We met yesterday.", ANCHOR, skip_code=True) == "We met yesterday [= 2023-05-07]."
    assert resolve_text("Plans for next week: go", ANCHOR, skip_code=True).startswith(
        "Plans for next week [week of 2023-05-15]:"
    )
    for code in ("x = date.today()", 'today.strftime("%d")', "$yesterday ok", "since yesterday=True"):
        assert resolve_text(code, ANCHOR, skip_code=True) == code, code


def test_the_content_gate_leaves_an_item_that_looks_like_code_untouched() -> None:
    """Invariant: an item with two or more code signals is kept byte for byte by the content gate,
    and the same item is resolved by the route gate's plain render.

    Red proof: removing the `looks_like_code` skip in `resolve_relative_times` resolves the
    ``yesterday`` in the comment and fails the first assertion.
    """
    code = "def load(): return cache == None  # rebuilt yesterday"
    assert looks_like_code(code)
    assert not looks_like_code("We met yesterday; it was fun (really).")
    items = [_item("c", code, 2)]
    assert resolve_relative_times(items, skip_code=True)[0].content == code
    assert resolve_relative_times(items)[0].content != code


def test_the_content_gate_resolves_prose_on_the_code_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: with ``RECALL_AML_T1_GATE=content`` a prose item is resolved even when the query
    took the code route, which the route gate never does.

    Red proof: dropping ``content_gate or`` from the gate condition in `HostedService.search` puts
    the route gate back and fails the ``endswith`` assertion.
    """
    monkeypatch.setenv("RECALL_AML_RESOLVE_RELATIVE_TIMES", "1")
    monkeypatch.setenv("RECALL_AML_T1_GATE", "content")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add(service, "w4-code", "w4-c", "I adopted a puppy yesterday.", 2)
    response = _search(service, "w4-code", CODE_QUERY)
    assert response.data, "precondition: the item is returned"
    assert any(str(item.content).endswith("yesterday [= 2023-05-02].") for item in response.data)
    assert "+t1-content-gate-v1" in service.search_content_profile


def test_the_served_c9_keeps_the_route_gate_and_the_v1_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: W4 changes nothing served until the owner switches it on: C9 keeps the route
    gate and v1, and its search-content profile carries no W4 suffix.

    Red proof: setting ``relative_times_gate="content"`` on the C9 variant fails the first
    assertion; 🔁 audit 2026-09-30: the profile half now builds C9 itself (it built C7 with T-1
    forced on), and adding the content-gate suffix whatever the gate (dropping its condition in
    `HostedService.search_content_profile`) fails the last assertion.
    """
    c9 = variant("C9_routed_specialists_grounded_graph_atomic")
    assert (c9.relative_times_gate, c9.relative_times_render) == ("route", "v1")
    for name in ("RECALL_AML_T1_GATE", "RECALL_AML_T1_RENDER", "RECALL_AML_RESOLVE_RELATIVE_TIMES"):
        monkeypatch.delenv(name, raising=False)
    service, _, _, _, _ = _service(c9.name)
    assert "+relative-times-resolved-v1" in service.search_content_profile
    assert "+t1-" not in service.search_content_profile


@pytest.mark.parametrize(("name", "value"), [("RECALL_AML_T1_GATE", "everywhere"), ("RECALL_AML_T1_RENDER", "v3")])
def test_a_bad_w4_value_stops_service_startup(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    """Invariant: an unknown gate or render is refused when the service starts, not on a Search.

    Red proof: removing ``self.relative_times_gate`` from the startup read in
    `HostedService.__init__` lets the service start and fails the gate row.
    """
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _service("C7_routed_specialists")


# Audit 2026-09-30 (CCA DEEP, PR 808). Red proofs ran on the testbench host against the pre-fix
# module (PR head 1681e66f) unless a mutation is named.

#: Code the first content gate annotated because it held fewer than two substring signals, or
#: because only the right-hand side of a phrase was checked.
CODE_SHAPES = [
    "today = datetime.now()",
    "yesterday = today - timedelta(days=1)",
    "let yesterday = new Date(today)",
    "report(day=yesterday)",
    'run(since="yesterday")',
    "WHERE day = 'yesterday'",
    "SELECT '2 weeks ago'::interval",
    'git log --since="2 weeks ago" --oneline',
    "git log --since=yesterday --oneline",
    "date -d yesterday +%F",
    'date -d "next monday"',
    "find . -newermt yesterday",
    '["today", "tomorrow"]',
    '{"when": "yesterday"}',
    '{"today": 1}',
    "expire = tomorrow",
    "$this->today",
    "Set `since=yesterday` in the config.",
    "```\nfind . -newermt yesterday\n```",
]


@pytest.mark.parametrize("code", CODE_SHAPES)
def test_code_shapes_are_left_as_written_under_the_content_gate(code: str) -> None:
    """Invariant: under the content gate a phrase that is part of code (an operand, a call
    argument, a quoted literal, a shell flag's value, member access, a fence or a backtick span)
    is left as written.

    Red proof: every row fails its first equality on the pre-fix module, which annotated each of
    them (``today = datetime.now()`` became ``today [= 2023-05-08] = datetime.now()``).
    """
    assert resolve_text(code, ANCHOR, skip_code=True) == code
    assert resolve_relative_times([_item("c", code, 7)], skip_code=True)[0].content == code


@pytest.mark.parametrize(
    ("prose", "resolved"),
    [
        ("Can we meet today/tomorrow?", "Can we meet today [= 2023-05-08]/tomorrow [= 2023-05-09]?"),
        ("Well...yesterday I went.", "Well...yesterday [= 2023-05-07] I went."),
        ("Meeting tomorrow:10am", "Meeting tomorrow [= 2023-05-09]:10am"),
        ("Note:yesterday was great", "Note:yesterday [= 2023-05-07] was great"),
        (
            "Plans for today/this week/this month.",
            "Plans for today [= 2023-05-08]/this week [week of 2023-05-08]/this month [= 2023-05].",
        ),
        ("I ran `today = date.today()` yesterday.", "I ran `today = date.today()` yesterday [= 2023-05-07]."),
        ('He said "tomorrow" and left.', 'He said "tomorrow [= 2023-05-09]" and left.'),
        ('She said: "tomorrow", then left.', 'She said: "tomorrow [= 2023-05-09]", then left.'),
        ("We met (yesterday) at noon.", "We met (yesterday [= 2023-05-07]) at noon."),
        ("It was - yesterday, I think.", "It was - yesterday [= 2023-05-07], I think."),
    ],
)
def test_prose_punctuation_is_not_read_as_code(prose: str, resolved: str) -> None:
    """Invariant: the content gate resolves prose whose punctuation merely touches a phrase: an
    alternative slash, an ellipsis, a time or a label after a colon, a quotation, a parenthesis, a
    dash, and prose after a backtick span.

    Red proof: the first six rows fail on the pre-fix module, which read ``/``, ``.`` and ``:``
    as code whatever surrounded them and did not know backtick spans; the backtick row also fails
    on a span pattern that runs to the end of the text. The last four were already resolved
    there; they fail on mutations of `_glued_to_code`: an opening quote counted as code (both
    quote rows), a quoted value after any colon counted as code (the second quote row), a
    parenthesis without the identifier condition (the parenthesis row), and any word starting
    with ``-`` read as a flag (the dash row).
    """
    assert resolve_text(prose, ANCHOR, skip_code=True) == resolved


@pytest.mark.parametrize(
    ("prose", "resolved"),
    [
        ("Title\n=====\nToday we met.", "Title\n=====\nToday [= 2023-05-08] we met."),
        ("Thanks,\n-John\nyesterday was fun", "Thanks,\n-John\nyesterday [= 2023-05-07] was fun"),
        ("I didn`t go yesterday, but I`ll go.", "I didn`t go yesterday [= 2023-05-07], but I`ll go."),
        ("昨天(yesterday) I went out.", "昨天(yesterday [= 2023-05-07]) I went out."),
        ("I felt -meh- today.", "I felt -meh- today [= 2023-05-08]."),
    ],
)
def test_the_code_rules_stay_on_their_own_line_and_shape(prose: str, resolved: str) -> None:
    """Invariant: an ``=`` or a flag on another line (a Markdown underline, a signature), a
    backtick used as an apostrophe, a parenthesis after a non-ASCII word, and a word merely
    wrapped in dashes do not make a phrase code.

    Red proof: the rules as this audit first drafted them (``=`` and flags looked for across
    lines, spans opened at any backtick, any alphanumeric before ``(`` as a call name, a flag
    allowed to end in ``-``) leave every row unresolved; the first draft is kept as the baseline
    in the pull request's audit notes.
    """
    assert resolve_text(prose, ANCHOR, skip_code=True) == resolved


def test_a_long_token_at_the_window_edge_is_read_conservatively() -> None:
    """Invariant: the bounded look does not change a decision next to a long token: a path whose
    segment is longer than the window (a sha256 digest) stays code, and a long hyphenated word
    the window cuts into is not read as a flag.

    Red proof: the version of this fix before the edge handling annotates the path and fails the
    first equality; removing ``not cut`` from the flag rule in `_glued_to_code` reads the cut word
    as a flag and fails the second.
    """
    path = "see blobs/sha256/" + "0123456789abcdef" * 4 + "/today now"
    assert resolve_text(path, ANCHOR, skip_code=True) == path
    word = "the-quick-brown-fox-jumps-over-the-lazy-dog-and-keeps-running-far-away-verbose"
    assert resolve_text(f"{word} today", ANCHOR, skip_code=True) == f"{word} today [= 2023-05-08]"


@pytest.mark.parametrize("unit", ["today/", "today ", "`x` today "], ids=["slashes", "spaces", "spans"])
def test_the_content_gate_costs_linear_time_on_a_long_item(unit: str) -> None:
    """Invariant: deciding whether a phrase is code looks at a bounded neighbourhood, so the cost
    of an item grows linearly with its length and one long retrieved item cannot stall a Search.
    The test compares the cost at two lengths, four times apart, so it holds on any machine:
    linear is about 4 times, quadratic about 16.

    Red proof: the first draft of this audit's rewrite, which sliced and split the whole text
    around every phrase and scanned every span per phrase, fails every row (a 120 KB item of the
    first row's shape took 310 s there, against 1.1 s now).
    """

    def cost(repeats: int, runs: int) -> float:
        text = unit * repeats
        best = float("inf")
        for _ in range(runs):
            started = time.perf_counter()
            resolve_text(text, ANCHOR, skip_code=True, render="v2")
            best = min(best, time.perf_counter() - started)
        return best

    small, large = cost(1000, 3), cost(4000, 2)
    assert large < 8 * small, f"4x the length cost {large / small:.1f}x the time"


@pytest.mark.parametrize("path", ["see logs/2023/today", "open /tmp/today now", "open ~/today now"])
def test_a_path_is_code_under_the_content_gate(path: str) -> None:
    """Invariant: a phrase inside a path is code, although an alternative ("today/tomorrow") is
    prose.

    Red proof: deleting the path rule from `_glued_to_code` annotates every row. The pre-fix
    module protected these rows too, by treating every ``/`` as code.
    """
    assert resolve_text(path, ANCHOR, skip_code=True) == path


@pytest.mark.parametrize(
    "prose",
    [
        "I will return the book myself. We met yesterday.",
        "My template is {name} and we met yesterday.",
        "Paris -> London was last week, fun => great.",
    ],
)
def test_english_words_and_chat_arrows_do_not_make_an_item_code(prose: str) -> None:
    """Invariant: "myself.", "return", one brace pair and chat arrows are prose, so the content
    gate still resolves the item.

    Red proof: every row fails ``not looks_like_code`` on the pre-fix module, whose substring
    signals counted ``self.`` with ``return ``, ``{`` with ``}``, and ``->`` with ``=>``.
    """
    assert not looks_like_code(prose)
    assert resolve_relative_times([_item("p", prose, 7)], skip_code=True)[0].content != prose


def test_case_folding_characters_resolve_like_their_ascii_letters() -> None:
    """Invariant: a phrase the pattern matches through case folding (the long s, the dotless i)
    resolves exactly as its ASCII spelling, instead of failing the Search or naming the wrong day.

    Red proof: on the pre-fix module "laſt week" raises KeyError in `_resolve`, failing the test
    at its first call; with that row removed, "yeſterday" resolves to 2023-05-08 and fails its
    equality.
    """
    assert resolve_text("We met laſt week.", ANCHOR) == "We met laſt week [week of 2023-05-01]."
    assert resolve_text("yeſterday", ANCHOR) == "yeſterday [= 2023-05-07]"
    assert resolve_text("next Tueſday", ANCHOR) == "next Tueſday [= 2023-05-09]"
    assert resolve_text("thıs week", ANCHOR, render="v2") == "thıs week [= the week of 2023-05-08]"


def test_the_fold_table_covers_every_character_the_pattern_folds() -> None:
    """Invariant: every non-ASCII character that ``re.IGNORECASE`` matches to an ASCII letter
    is folded to that letter before a lookup.

    Red proof: removing U+017F from ``_FOLD`` leaves the long s unfolded and fails the equality.
    """
    letter = re.compile("[a-z]", re.IGNORECASE)
    folded = [chr(point) for point in range(0x80, 0x110000) if letter.fullmatch(chr(point))]
    assert folded, "precondition: case folding maps some non-ASCII characters to ASCII letters"
    # The fold must reach an ASCII letter: a character left as itself still matches itself.
    unfolded = [
        c
        for c in folded
        if not (len(_key(c)) == 1 and _key(c) in ascii_lowercase and re.fullmatch(_key(c), c, re.IGNORECASE))
    ]
    assert unfolded == []


def test_an_anchor_at_the_edge_of_the_calendar_keeps_the_item() -> None:
    """Invariant: an item dated 0001-01-01, which has no day before it, is returned as written
    rather than failing the whole Search.

    Red proof: the pre-fix module raises OverflowError from `resolve_relative_times`, failing the
    test at the call.
    """
    edge = _item("e", "We met yesterday.", 7).model_copy(
        update={"created_at": datetime(1, 1, 1, tzinfo=timezone.utc)}
    )
    assert resolve_relative_times([edge])[0].content == "We met yesterday."


@pytest.mark.parametrize(
    "phrase",
    [
        "the day before yesterday", "3 days ago", "a week ago", "two months ago", "a year ago",
        "today", "tonight", "this morning", "this afternoon", "this evening", "yesterday",
        "last night", "tomorrow", "last weekend", "this week", "next month", "last year",
        "next friday",
    ],
)
def test_the_keyword_prefilter_never_skips_a_phrase(phrase: str) -> None:
    """Invariant: the ASCII keyword prefilter lets every form the pattern resolves through, in
    any case.

    Red proof: dropping ``"ago"`` from ``_KEYWORDS`` leaves the four ``ago`` rows unresolved;
    dropping ``"next"`` fails the last two.
    """
    for text in (f"We spoke {phrase}.", f"WE SPOKE {phrase.upper()}."):
        assert resolve_text(text, ANCHOR) != text, text


def test_a_second_pass_leaves_v2_output_alone() -> None:
    """Invariant: applying T-1 again, in either render, changes nothing already resolved by v2.

    Red proof: removing ``=|`` from the already-resolved lookahead in ``_PATTERN`` annotates
    ``last week [= the week before ...]`` a second time and fails the first equality.
    """
    once = resolve_text("last week, two weeks ago, next weekend and yesterday", ANCHOR, render="v2")
    assert resolve_text(once, ANCHOR, render="v2") == once
    assert resolve_text(once, ANCHOR, render="v1") == once


def test_a_w4_option_set_while_t1_is_off_is_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Invariant: a W4 option that cannot take effect, because T-1 itself is off, is logged at
    startup instead of being ignored silently.

    Red proof: the pre-fix service logs nothing and fails the assertion.
    """
    monkeypatch.setenv("RECALL_AML_RESOLVE_RELATIVE_TIMES", "0")
    monkeypatch.setenv("RECALL_AML_T1_GATE", "content")
    monkeypatch.delenv("RECALL_AML_T1_RENDER", raising=False)
    with caplog.at_level(logging.WARNING, logger="recall_aml"):
        _service("C7_routed_specialists")
    assert "RECALL_AML_T1_GATE set while T-1 is off" in caplog.text
