#!/usr/bin/env python3
"""Tests for the discoverability probe's apparatus, which decides a scientific verdict.

Every test here is about ONE question: can this instrument publish a number it has not earned?
A retrieval probe fails in a direction that looks exactly like a real null, so the tests below
are all shaped around that asymmetry: an empty arm, a mis-keyed DSN, a no-op exclusion and a
gate applied to the wrong denominator all print `rescue 0/14` and `retention 26/26`, which is
character for character what the genuine 2026-08-27 result printed.

Nothing here needs a database, a corpus, an OpenRouter key or a network. The session rows are
fixtures, the retrieval answers are dicts, and the two scripts are imported for their pure
predicates only.

Written 2026-08-27 after a DEEP audit of the run that produced
`docs/preregistrations/2026-08-27-memo-discoverability-authoring.md`, which found 14 confirmed
defects in the apparatus, four of them able to fabricate that exact null. Every test was watched
RED against the pre-fix scripts before the fix landed; the reds are named per test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import agent_ab_prepare_discoverability_corpus as prep  # noqa: E402
import agent_ab_probe_discoverability as probe  # noqa: E402

from recall.document import parse_document  # noqa: E402


def indexed_body(text: str) -> str:
    """Exactly what the corpus builder chunks, via the builder's own parser.

    `recall/generations.py` sets `body = parse_document(text).human_body` and chunks THAT, so a
    test that asserts against this function is asserting against the real index rather than
    against a local reimplementation of it. This is the whole reason the description defect was
    invisible: the treated bytes were plainly there in the file.
    """

    return parse_document(text).human_body

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")


def session(base: str, memo: str, hit: bool, queries: list[str] | None = None) -> dict:
    return {
        "task_id": f"{base}#r1",
        "base": base,
        "memo": memo,
        "queries": queries if queries is not None else [f"q-{base}"],
        "hit_in_run": hit,
    }


def registered_population() -> list[dict]:
    """14 misses and 26 hits after excluding ts-raise-on-missing, as the record fixes them."""

    rows = []
    for i in range(6):
        rows.append(session("ts-lf-rewrite", f"m-lf-{i}", False))
    for i in range(5):
        rows.append(session("ts-worktree-import", f"m-wt-{i}", False))
    for i in range(3):
        rows.append(session("ts-sample-covers-tail", f"m-st-{i}", False))
    for i in range(26):
        rows.append(session("ts-hitfamily", f"m-hit-{i}", True))
    # The registered exclusion: 1 miss and 5 hits that must reach no endpoint.
    rows.append(session("ts-raise-on-missing", "m-excl-0", False))
    for i in range(5):
        rows.append(session("ts-raise-on-missing", f"m-excl-h{i}", True))
    return rows


# ---------------------------------------------------------------- gate integrity


def test_gate_refuses_an_unregistered_denominator() -> None:
    """RED pre-fix: no function existed; the gate compared bare numerators at any n.

    The audit's counterfactual: drop --exclude-base and the population becomes 15 misses and 31
    hits, where 13 and 24 still clear. A gate whose stringency moves with the data is not the
    gate that was registered.
    """

    ok, _ = probe.population_matches_registration(14, 26)
    check("gate accepts the registered 14/26", ok)
    for misses, hits in ((15, 31), (14, 31), (15, 26), (8, 26)):
        bad, reason = probe.population_matches_registration(misses, hits)
        check(
            f"gate refuses unregistered population {misses}/{hits}",
            not bad and bool(reason),
            f"got ok={bad} reason={reason!r}",
        )


def test_exclusion_that_matches_nothing_is_refused() -> None:
    """RED pre-fix: `excluded = set(args.exclude_base)` was never compared to any row's base.

    A typo in the family name silently scored the excluded sessions, which is how the wrong
    denominator arrives in the first place.
    """

    rows = registered_population()
    unmatched = probe.unmatched_exclusions(rows, {"ts-raise-on-missing"})
    check("the real exclusion matches rows", unmatched == [], f"got {unmatched}")
    unmatched = probe.unmatched_exclusions(rows, {"ts-raise-on-mising"})
    check(
        "a typo'd exclusion is reported",
        unmatched == ["ts-raise-on-mising"],
        f"got {unmatched}",
    )


def test_void_is_not_a_zero_exit() -> None:
    """RED pre-fix: main() returned 0 unconditionally, so void and verdict were indistinguishable."""

    check("a void run exits non-zero", probe.exit_code(void=True) != 0)
    check("a valid run exits zero", probe.exit_code(void=False) == 0)


# ---------------------------------------------------------------- arm identity


def test_an_arm_identical_to_control_is_refused() -> None:
    """RED pre-fix: no such function; a mis-keyed --arm DSN produced the published null exactly.

    Control reproduced 14/14 misses, so an arm pointed at control CANNOT rescue any miss and
    MUST retain every hit: rescue 0/14 and retention 26/26, byte-identical to the real result.
    """

    control = {f"q{i}": [f"a{i}.md", f"b{i}.md"] for i in range(20)}
    twin = {k: list(v) for k, v in control.items()}
    # The real arms diverged from control on 72 to 77 of 78 queries.
    live = {k: ([f"z{k}.md"] if i % 2 == 0 else list(v)) for i, (k, v) in enumerate(control.items())}

    ok, share = probe.arm_is_distinct(control, twin)
    check("an arm identical to control is refused", not ok, f"share={share}")
    ok, share = probe.arm_is_distinct(control, live)
    check("a genuinely different arm is accepted", ok, f"share={share}")

    nudged = {k: list(v) for k, v in control.items()}
    nudged["q0"] = ["z.md"]
    ok, share = probe.arm_is_distinct(control, nudged)
    check(
        "an arm differing on 1 of 20 queries is refused as too close to control",
        not ok,
        f"share={share}",
    )
    check("an empty arm is refused", not probe.arm_is_distinct({}, {})[0])


def test_reserved_and_duplicate_arm_names_are_refused() -> None:
    """RED pre-fix: `{"control": dsn, **arms}` let --arm control=... replace the control corpus."""

    for bad in ("control", "memo", "excluded", "task_id", "hit_in_run"):
        err = probe.validate_arm_names([bad])
        check(f"arm name {bad!r} is refused", err is not None, "accepted a reserved name")
    err = probe.validate_arm_names(["retitle", "retitle"])
    check("a duplicate arm name is refused", err is not None, "accepted a duplicate")
    err = probe.validate_arm_names(["retitle", "restructured", "pointer"])
    check("the registered arm names are accepted", err is None, f"got {err}")


def test_retention_has_a_direct_twin() -> None:
    """RED pre-fix: only `rescued_direct` existed, so pointer retention used the loose criterion.

    In the pointer arm a `<stem>--tasks.md` document satisfies the substring test, so a hit whose
    memo fell out of top-5 could still score as retained. It did not happen in the 2026-08-27 run,
    and nothing in the instrument made that a property rather than luck.
    """

    row = {"memo": "python-write-text-crlf-churn", "queries": ["q"]}
    pointer_only = {"q": ["python-write-text-crlf-churn--tasks.md", "other.md"]}
    direct = {"q": ["python-write-text-crlf-churn.md", "other.md"]}

    check(
        "the loose criterion credits a pointer document",
        probe.outcome(row, pointer_only),
        "substring criterion failed to match the pointer doc",
    )
    check(
        "the direct criterion does NOT credit a pointer document",
        not probe.outcome_direct(row, pointer_only),
        "direct criterion matched a pointer doc",
    )
    check(
        "the direct criterion credits the memo itself",
        probe.outcome_direct(row, direct),
        "direct criterion missed the memo",
    )


# ---------------------------------------------------------------- treatment fidelity


def test_the_generated_description_reaches_the_indexed_body() -> None:
    """RED pre-fix: the description was written ONLY into frontmatter, which the builder strips.

    Verified against the live corpora during the audit: the generated description appeared in
    indexed chunk text for 0 of 40 memos while the title appeared in 40 of 40, because
    `recall/generations.py` chunks `parse_document(text).human_body`. A treatment that reaches no
    index is not a treatment, and the retitle arm was therefore measured as title-only.
    """

    rewrite = {
        "title": "Editing a version file from a Python script on Windows",
        "description": "An engineer scripts a file edit and git reports it modified forever.",
        "tasks": [f"task {i}" for i in range(5)],
    }
    source = '---\nname: x\ndescription: "old"\n---\n\nBody text here.\n'
    for name, fn in (
        ("retitle", prep.retitle_text),
        ("restructured", prep.restructured_text),
    ):
        text = fn(source, rewrite, "x")
        body = indexed_body(text)
        check(
            f"{name}: the generated description is in the INDEXED body",
            rewrite["description"] in body,
            "description is frontmatter-only, so it never reaches the index",
        )
        check(
            f"{name}: the generated title is in the indexed body",
            rewrite["title"] in body,
            "title missing from body",
        )


def test_a_source_without_frontmatter_still_gets_the_full_treatment() -> None:
    """RED pre-fix: `head = with_description(...) if frontmatter else ""` dropped it silently.

    27 of 190 memos in the shipped run took this branch, 21 of them because the chunk
    reconstruction had already destroyed their frontmatter.
    """

    rewrite = {
        "title": "A title",
        "description": "A searcher-oriented description.",
        "tasks": [f"task {i}" for i in range(5)],
    }
    bare = "Just a body with no frontmatter at all.\n"
    for name, fn in (
        ("retitle", prep.retitle_text),
        ("restructured", prep.restructured_text),
    ):
        text = fn(bare, rewrite, "bare-memo")
        body = indexed_body(text)
        check(
            f"{name}: a frontmatter-less source keeps its description",
            rewrite["description"] in body,
            "description dropped for a source without frontmatter",
        )


def test_a_zero_fidelity_joiner_is_refused() -> None:
    """RED pre-fix: no gate; the run reconstructed 25 sources with a joiner scoring 0/167.

    `learn_joiner` returns the FIRST candidate on an all-zero tie, which is the empty string, the
    one that glues chunk boundaries together with no separator.
    """

    ok, reason = prep.reconstruction_is_trustworthy(reproduced=167, verified=167)
    check("a perfect joiner is accepted", ok, f"got {reason}")
    ok, reason = prep.reconstruction_is_trustworthy(reproduced=0, verified=167)
    check("a 0/167 joiner is refused", not ok and bool(reason), f"got ok={ok} reason={reason!r}")
    ok, reason = prep.reconstruction_is_trustworthy(reproduced=160, verified=167)
    check("a partial joiner is refused", not ok, "accepted a joiner that reproduces only some")
    ok, reason = prep.reconstruction_is_trustworthy(reproduced=0, verified=0)
    check(
        "no verified sample at all is refused",
        not ok and bool(reason),
        "accepted an unmeasurable joiner",
    )


# ---------------------------------------------------------------- cache provenance


def test_the_uri_decoder_yields_an_absolute_path() -> None:
    """RED pre-fix: the path was re-derived with `urlparse` + `unquote` + `lstrip("/")`.

    On POSIX that yields a RELATIVE path, so every `is_file()` is False, every sha check fails,
    and the whole corpus is silently reconstructed instead of verified. `recall/manifest.py`
    documents `local_path_for` as the single implementation of this decision and names the
    re-derivation as the wrong decoder.
    """

    from urllib.parse import unquote, urlparse

    for uri in ("file:///tmp/a%20b.md", "file:///home/g/memory/x.md"):
        old = Path(unquote(urlparse(uri).path.lstrip("/")))
        new = prep.local_path_for(uri)
        check(
            f"{uri}: the decoder in use yields an absolute path",
            new.is_absolute(),
            f"got {new}",
        )
        check(
            f"{uri}: the old derivation was NOT absolute (this is the defect)",
            not old.is_absolute() or old == new,
            f"old={old}",
        )


def test_the_resume_cache_is_bound_to_its_generation_terms() -> None:
    """RED pre-fix: the cache was keyed on filename alone.

    The generation terms changed twice in two days on this file (a parser repair, then reasoning
    disabled), and `rewrites.json` attests the CURRENT prompt over whatever the cache held.
    """

    a = prep.terms_fingerprint()
    check("the fingerprint is stable within one build", a == prep.terms_fingerprint())
    b = prep.terms_fingerprint(prompt="a different prompt")
    check("changing the prompt changes the fingerprint", a != b)
    c = prep.terms_fingerprint(model="anthropic/claude-haiku-4.5")
    check("changing the model changes the fingerprint", a != c)
    d = prep.terms_fingerprint(reasoning=True)
    check("changing the reasoning flag changes the fingerprint", a != d)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rewrites.partial.json"
        prep.write_cache(path, {"m.md": {"title": "t", "description": "d", "tasks": ["x"]}})
        loaded = prep.load_cache(path)
        check("a cache written under current terms is reused", "m.md" in loaded, f"got {loaded}")

        stale = json.loads(path.read_text(encoding="utf-8"))
        stale["terms"] = "0" * 64
        path.write_text(json.dumps(stale), encoding="utf-8", newline="\n")
        loaded = prep.load_cache(path)
        check("a cache from other terms is discarded", loaded == {}, f"got {loaded}")

        path.write_text('{"rewrites": {"m.md":', encoding="utf-8", newline="\n")
        loaded = prep.load_cache(path)
        check("a torn cache is treated as empty, not fatal", loaded == {}, f"got {loaded}")


def test_the_cache_write_is_atomic() -> None:
    """RED pre-fix: `write_text` truncated in place 190 times, and the reader had no guard.

    An interrupt during the write is the exact event the cache exists to survive.
    """

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rewrites.partial.json"
        prep.write_cache(path, {"a.md": {"title": "t", "description": "d", "tasks": ["x"]}})
        leftovers = [p.name for p in Path(tmp).iterdir() if p.name != path.name]
        check("no temp file is left behind", leftovers == [], f"found {leftovers}")
        prep.write_cache(path, {"a.md": {"title": "t2", "description": "d", "tasks": ["x"]}})
        check(
            "a rewrite replaces the content",
            prep.load_cache(path)["a.md"]["title"] == "t2",
        )


def test_generation_failure_names_the_memo() -> None:
    """RED pre-fix: generate_rewrite raised SystemExit, which `except Exception` cannot catch.

    issubclass(SystemExit, Exception) is False, so the wrapper that attaches the memo name was
    dead on exactly the two failures it was written for.
    """

    check(
        "the generator's failure type is catchable by except Exception",
        issubclass(prep.GenerationError, Exception)
        and not issubclass(prep.GenerationError, SystemExit),
        "GenerationError must derive from Exception, not SystemExit",
    )


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"discoverability apparatus: {len(tests)} test groups\n")
    for test in tests:
        print(test.__name__)
        test()
        print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
