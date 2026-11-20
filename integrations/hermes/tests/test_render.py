"""The prefetch block: citable, retired and abstained answers must read differently."""

from re_call_hermes.render import ABSTAIN_LINE, RETIRED_LEAD, render_search, render_unavailable


def hit(chunk, verdict="ok", memo="a.md", text="fact", **extra):
    return {"memo_id": memo, "chunk_id": chunk, "verdict": verdict, "text": text, **extra}


def test_only_citable_hits_are_presented_as_current():
    """Invariant: a hit is shown as current only when RE-call lists its chunk in evidence.citable.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    RENDER-CITABLE`): in re_call_hermes/render.py, `citable_ids is None or str(hit.get("chunk_id")) in
    citabl...` became `True`; the test failed at `AssertionError: assert 'Likes tea' not in '## RE-call
    ...d: Likes tea'`. Restored, green.
    """
    answer = {
        "abstained": False,
        "evidence": {"decision": "answer", "citable": ["c1"]},
        "hits": [hit("c1", text="Works at Acme"), hit("c2", memo="b.md", text="Likes tea")],
    }
    rendered = render_search(answer)
    assert "[ok] memo a.md: Works at Acme" in rendered.text
    assert "Likes tea" not in rendered.text
    assert rendered.citable == 1


def test_superseded_hits_are_flagged_with_their_replacement_and_never_counted():
    """Invariant: a superseded hit is listed under the retired lead, naming what replaced it.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    RENDER-RETIRED`): in re_call_hermes/render.py, `if verdict in RETIRED_VERDICTS:` became `if verdict in
    ():`; the test failed at `assert 'Retired memories (superseded or expired): do not state these as
    curre...`. Restored, green.
    """
    answer = {
        "abstained": False,
        "evidence": {"citable": ["c1", "c0"]},
        "hits": [
            hit("c1", memo="new.md", text="Works at Acme", valid_from="2026-11-01"),
            hit("c0", verdict="superseded", memo="old.md", text="Works at Initech", superseded_by="new.md"),
        ],
    }
    text = render_search(answer).text
    assert RETIRED_LEAD in text
    current, retired = text.split(RETIRED_LEAD)
    assert "Initech" not in current
    assert "[superseded, replaced by new.md] memo old.md: Works at Initech" in retired
    assert "(valid from 2026-11-01)" in current
    assert render_search(answer).citable == 1


def test_abstention_says_so_and_cites_nothing():
    """Invariant: when RE-call abstains, no hit is presented as current, whatever its verdict.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    RENDER-ABSTAIN`): in re_call_hermes/render.py, `elif not abstained and verdict == "ok"` became `elif
    verdict == "ok"`; the test failed at `assert 'RE-call abstained: nothing in memory passed its calibrated
    trust chec...`. Restored, green.
    """
    answer = {"abstained": True, "evidence": {"citable": ["c1"]}, "hits": [hit("c1", text="Works at Acme")]}
    rendered = render_search(answer)
    assert ABSTAIN_LINE in rendered.text
    assert "Acme" not in rendered.text
    assert rendered.citable == 0


def test_unavailable_names_the_reason_on_one_line():
    """Nonbehavioral control: formatting only; covered for shape, no red proof claimed."""
    text = render_unavailable("store is not calibrated").text
    assert text.splitlines()[1].endswith("store is not calibrated")
