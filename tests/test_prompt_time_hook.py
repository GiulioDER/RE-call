"""The `UserPromptSubmit` memory-recall hook: the properties that stop it breaking a session.

This hook runs before EVERY user turn, and the thing it sits on is the user's own message. So its
failure modes are not "it ranked badly", they are "the prompt never reached the model" and "the
session died on a hook". Those are what is asserted here, alongside the store resolution, which is
the part that silently returns nothing when it is wrong.

Nothing here needs a database or a network: the whole point of this hook is that it reads files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from recall_hooks import prompt_time


@pytest.fixture
def hook_env(tmp_path, monkeypatch):
    """An isolated CLAUDE_CONFIG_DIR, so no test can read the developer's own memory store."""

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    return tmp_path


def make_store(root: Path, memos: dict[str, str]) -> Path:
    store = root / "store"
    store.mkdir(parents=True, exist_ok=True)
    for stem, description in memos.items():
        (store / f"{stem}.md").write_text(
            f"---\nname: {stem}\ndescription: {description}\n---\n\n{description}\n",
            encoding="utf-8",
            newline="\n",
        )
    return store


def write_config(root: Path, **prompt_time_block) -> None:
    (root / "recall-hook.json").write_text(
        json.dumps({"prompt_time": prompt_time_block}) + "\n", encoding="utf-8", newline="\n"
    )


CORPUS = {
    "embedding-runs-on-vps2": "Embedding runs on VPS2 only, never on this workstation, and never two at once.",
    "pyarrow-blocked-by-application-control": "Smart App Control blocked the pyarrow parquet DLL for three hours and then cleared itself.",
    "suite-runs-in-parallel": "The pytest suite takes fifty minutes serial and fourteen minutes under four xdist workers.",
    "MEMORY": "hub index pointing at embedding pyarrow parquet suite xdist workstation vps2",
}


# --------------------------------------------------------------------------------------
# The two properties that matter more than retrieval quality
# --------------------------------------------------------------------------------------


def test_it_can_never_block_the_prompt(hook_env, capsys):
    """Exit 2 on this event DISCARDS the user's message. Nothing here may return anything but 0."""

    store = make_store(hook_env, CORPUS)
    write_config(hook_env, store=str(store))

    assert prompt_time.user_prompt_submit({"prompt": "should we run embedding on this workstation"}) == 0

    document = json.loads(capsys.readouterr().out)
    # Asserting the SHAPE rather than grepping the text: a memo whose summary contains the word
    # "blocked" made the substring form of this test fail for the wrong reason, which is the
    # test being wrong and not the hook.
    assert set(document) == {"hookSpecificOutput"}
    assert document["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert set(document["hookSpecificOutput"]) == {"hookEventName", "additionalContext"}
    assert "decision" not in document
    assert "continue" not in document


@pytest.mark.parametrize(
    "breakage",
    ["load_memos", "rank", "render", "find_store", "settings"],
)
def test_any_internal_failure_is_silence_not_an_exception(hook_env, monkeypatch, capsys, breakage):
    """A hook that raises is charged to the client, and this one raises into the user's message."""

    store = make_store(hook_env, CORPUS)
    write_config(hook_env, store=str(store))

    def explode(*args, **kwargs):
        raise RuntimeError("corpus on fire")

    monkeypatch.setattr(prompt_time, breakage, explode)

    assert prompt_time.user_prompt_submit({"prompt": "should we run embedding on this workstation"}) == 0
    assert capsys.readouterr().out == ""


def test_a_hand_broken_config_does_not_discard_the_prompt(hook_env, capsys):
    """The regression this exists for: `settings()` sat OUTSIDE the guard, so `int("abc")` on a
    hand-edited `k` raised straight out of the hook and took the user's message with it."""

    make_store(hook_env, CORPUS)
    (hook_env / "recall-hook.json").write_text(
        json.dumps({"prompt_time": {"k": "three"}}) + "\n", encoding="utf-8", newline="\n"
    )

    assert prompt_time.user_prompt_submit({"prompt": "should we run embedding on this workstation"}) == 0
    assert capsys.readouterr().out == ""


def test_the_injected_json_survives_a_cp1252_stdout(hook_env):
    """Memos are full of emoji. `ensure_ascii` is what keeps this hook off a Windows stdout crash.

    Asserting the ENCODABILITY of the payload rather than the flag, because the flag is a default
    that a future edit could pass explicitly as False without any test noticing.
    """

    store = make_store(hook_env, {"blocked": "⛔ never rebuild the corpus 🔑 embedding workstation vps2"})
    write_config(hook_env, store=str(store))

    payload = json.dumps(
        {
            "hookSpecificOutput": {
                "additionalContext": prompt_time.render(
                    [("blocked", "⛔ never rebuild 🔑", 4.2)], store
                )
            }
        }
    )
    payload.encode("cp1252")  # raises UnicodeEncodeError if a raw emoji got through


# --------------------------------------------------------------------------------------
# Store resolution: the failure that looks exactly like "no memos matched"
# --------------------------------------------------------------------------------------


def test_a_worktree_resolves_to_its_repositorys_store(hook_env):
    """⚠️ The regression this exists for: a session's cwd is usually a worktree, whose own slug
    has no store, and walking up is what finds the repository's."""

    repository = hook_env / "Documents" / "recall"
    worktree = repository / ".claude" / "worktrees" / "some-branch"
    worktree.mkdir(parents=True)
    store = hook_env / "projects" / prompt_time.project_slug(repository.resolve()) / "memory"
    store.mkdir(parents=True)

    assert prompt_time.find_store(str(worktree)) == store


def test_it_never_walks_down_into_another_projects_store(hook_env):
    """Only an ENCLOSING project's memories are reachable, never a sibling's."""

    mine = hook_env / "Documents" / "mine"
    theirs = hook_env / "Documents" / "mine" / "nested"
    theirs.mkdir(parents=True)
    store = hook_env / "projects" / prompt_time.project_slug(theirs.resolve()) / "memory"
    store.mkdir(parents=True)

    assert prompt_time.find_store(str(mine)) is None


def test_an_unconfigured_project_is_silent(hook_env, capsys):
    """No store means behave exactly as if this hook were not installed."""

    project = hook_env / "Documents" / "unrelated"
    project.mkdir(parents=True)

    assert prompt_time.user_prompt_submit({"prompt": "a long enough prompt about embedding", "cwd": str(project)}) == 0
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------------------
# When it declines to speak
# --------------------------------------------------------------------------------------


def test_a_short_prompt_reads_no_files(hook_env, monkeypatch, capsys):
    """"ok" must not cost a scan of the whole store."""

    write_config(hook_env, store=str(make_store(hook_env, CORPUS)))
    monkeypatch.setattr(prompt_time, "load_memos", lambda store: pytest.fail("read the store"))

    assert prompt_time.user_prompt_submit({"prompt": "ok"}) == 0
    assert capsys.readouterr().out == ""


def test_disabled_means_disabled(hook_env, monkeypatch, capsys):
    write_config(hook_env, enabled=False, store=str(make_store(hook_env, CORPUS)))
    monkeypatch.setattr(prompt_time, "load_memos", lambda store: pytest.fail("read the store"))

    assert prompt_time.user_prompt_submit({"prompt": "should we run embedding on this workstation"}) == 0
    assert capsys.readouterr().out == ""


def test_an_off_topic_prompt_injects_nothing(hook_env):
    """A single coincidental word is not a hit; MIN_MATCHED_TOKENS is what says so."""

    store = make_store(hook_env, CORPUS)
    assert prompt_time.rank(prompt_time.load_memos(store), "book a restaurant near the station") == []


# --------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------


def test_it_finds_the_memo_that_governs_the_prompt(hook_env):
    store = make_store(hook_env, CORPUS)
    hits = prompt_time.rank(prompt_time.load_memos(store), "should we run the embedding on this workstation")

    assert hits, "the governing memo did not come back at all"
    assert hits[0][0] == "embedding-runs-on-vps2"


def test_the_hub_and_sub_indexes_are_not_rankable(hook_env):
    """They repeat every memo's hook line, so they out-score the memos they point at."""

    store = make_store(hook_env, CORPUS)
    stems = {stem for stem, _, _ in prompt_time.load_memos(store)}

    assert "MEMORY" not in stems
    assert stems == set(CORPUS) - prompt_time.INDEX_STEMS


def test_it_names_at_most_k_memos(hook_env):
    store = make_store(hook_env, CORPUS)
    hits = prompt_time.rank(prompt_time.load_memos(store), "embedding pyarrow suite workstation parquet", k=2)

    assert len(hits) <= 2


def test_the_rendered_block_carries_a_readable_path(hook_env):
    store = make_store(hook_env, CORPUS)
    text = prompt_time.render([("embedding-runs-on-vps2", "runs on VPS2", 9.1)], store)

    assert "embedding-runs-on-vps2" in text
    assert str(store / "embedding-runs-on-vps2.md") in text
    assert "Do not re-derive" in text


DEPLOYED = Path.home() / ".claude" / "hooks" / "recall_hooks"
SOURCE = Path(__file__).resolve().parent.parent / "recall_hooks"


def _source_modules() -> list[str]:
    """Every module in the package, asked for at collection time rather than listed here.

    ⛔ **The list used to be four literal names, and that is a guard that cannot fail for anything
    it does not already know about.** Two modules were added to this package and deployed to no
    machine, and the guard stayed green the whole time, because it only ever compared the four it
    named. The parametrisation now comes from the source directory, so a module that exists cannot
    be outside the comparison.

    The failure it was blind to is worse than drift. `__init__.py` imports the new modules lazily
    and returns 0 on `ImportError`, which is the correct degrade for a hook, so an undeployed
    module does not raise: the feature is simply absent while every test passes and every session
    starts cleanly.
    """
    return sorted(path.name for path in SOURCE.glob("*.py"))


@pytest.mark.skipif(not DEPLOYED.is_dir(), reason="no deployed copy on this machine")
@pytest.mark.parametrize("module", _source_modules())
def test_the_deployed_copy_matches_the_source(module):
    """⚠️ CLAUDE.md records `session_start_hook.py` and its deployed twin drifting by 57 lines
    with nothing reporting it. This hook runs from a deployed copy on purpose, because `-m`
    otherwise resolves the package out of whatever branch the cwd happens to be, so a forgotten
    redeploy is the way this feature silently reverts. A failure here means: copy it again.

    Skipped where there is no deployed copy, which is every machine but the author's and CI.
    """

    deployed = DEPLOYED / module
    assert deployed.exists(), (
        f"{module} exists in the source and not in ~/.claude/hooks/recall_hooks; deploy it. "
        "An absent module does not raise: the lazy imports degrade to exit 0, so the feature is "
        "missing while everything looks healthy."
    )
    assert deployed.read_bytes() == (SOURCE / module).read_bytes(), (
        f"{module} in ~/.claude/hooks/recall_hooks differs from the source; redeploy it"
    )


@pytest.mark.skipif(not DEPLOYED.is_dir(), reason="no deployed copy on this machine")
def test_the_deployed_copy_holds_nothing_the_source_lost():
    """The mirror of the above: a module deleted or renamed in the source stays deployed forever.

    That one is importable and therefore live, so a rename leaves the OLD module answering on a
    machine where the new one is also present, and which wins depends on the import. Copying files
    over never removes anything, so nothing else would catch it.
    """

    extra = sorted(p.name for p in DEPLOYED.glob("*.py"))
    assert set(extra) <= set(_source_modules()), (
        f"~/.claude/hooks/recall_hooks holds {sorted(set(extra) - set(_source_modules()))}, "
        "which the source no longer has; remove it rather than leaving it importable"
    )


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("decision", "decisions"),
        ("test", "tests"),
        ("run", "runs"),
        ("chunk", "chunks"),
        ("file", "files"),
        ("embedding", "embeddings"),
        ("generation", "generations"),
        ("calibration", "calibrations"),
    ],
)
def test_the_stemmer_folds_a_word_onto_its_plural(singular, plural):
    """The property under test is AGREEMENT, not the root. A crude stemmer only has to agree
    with itself, so asserting the exact string would pin an implementation detail and would have
    to be rewritten for every rule added."""

    assert prompt_time.stem(singular) == prompt_time.stem(plural)


@pytest.mark.parametrize("word", ["bus", "gas", "has", "css", "dns"])
def test_the_stemmer_does_not_shred_a_short_word(word):
    """STEM_MIN_LENGTH is the whole guard, and `bus` losing its `s` is what it is guarding."""

    assert prompt_time.stem(word) == word


def test_stemming_is_what_makes_a_paraphrased_prompt_hit(hook_env):
    """The regression this exists for: the prompt says "decisions", the memo says "decision"."""

    store = make_store(
        hook_env, {"prior-decision-record": "A committed decision about rebuilding the corpus."}
    )
    hits = prompt_time.rank(prompt_time.load_memos(store), "what prior decisions cover rebuilding")

    assert [stem for stem, _, _ in hits] == ["prior-decision-record"]


def test_an_unreadable_memo_does_not_cost_the_others(hook_env, monkeypatch):
    store = make_store(hook_env, CORPUS)
    real = Path.read_text

    def refuse(self, *args, **kwargs):
        if self.stem == "embedding-runs-on-vps2":
            raise OSError("locked by another process")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)
    stems = {stem for stem, _, _ in prompt_time.load_memos(store)}

    assert "embedding-runs-on-vps2" not in stems
    assert "pyarrow-blocked-by-application-control" in stems


# --------------------------------------------------------------------------------------
# Provenance: every injected record carries the dates it states about itself, none inferred
# --------------------------------------------------------------------------------------


def write_memo(store: Path, stem: str, frontmatter: str, body: str) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{stem}.md"
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8", newline="\n")
    return path


def test_provenance_labels_each_date_by_the_field_it_came_from(hook_env):
    """Event time and record time stay apart, and the latest dated correction is reported.

    Red proof: replacing `max(corrections)` with `corrections[0]` in `prompt_time.provenance`
    reports `last 2026-08-20` and fails the equality below.
    """

    path = write_memo(
        hook_env / "store",
        "2026-08-18-a-dated-memo",
        "name: x\ndescription: y\nmetadata:\n  valid_from: 2026-08-10\n"
        "  modified: 2026-08-19T10:00:00Z\n  originSessionId: deadbeef-0001",
        "Body.\n\n\U0001f501 Corrected 2026-08-20.\n\n\U0001f501 **Corrected 2026-09-02**: later.",
    )

    assert prompt_time.provenance(path) == (
        "[written 2026-08-18; valid from 2026-08-10; modified 2026-08-19; "
        "corrected 2x, last 2026-09-02; session deadbeef]"
    )


def test_a_date_in_the_body_is_never_taken_as_the_memos_own(hook_env):
    """A memo whose frontmatter states no date is `[undated]`, whatever dates its prose quotes.

    The body is read for one thing only, the date on a `🔁` correction line, which the
    `corrected` part of the stamp names as such.

    Red proof: searching the whole file instead of the frontmatter (`VALID_FROM_RE.search(raw)`
    in `prompt_time.provenance`) returns `[valid from 2026-01-01]` and fails the equality below.
    """

    path = write_memo(
        hook_env / "store",
        "an-undated-memo",
        "name: an-undated-memo\ndescription: quotes a date",
        "valid_from: 2026-01-01 is what the old run was measured against.",
    )

    assert prompt_time.provenance(path) == "[undated]"


def test_a_horizontal_rule_is_not_read_as_frontmatter(hook_env):
    """A memo opening with a rule and a heading has no frontmatter, as `frontmatter_span` says.

    Red proof: deleting the `_is_frontmatter` check in `prompt_time.provenance` pairs the two
    rules, reads the prose line as the memo's own field and returns `[valid from 2026-01-01]`,
    failing the equality below.
    """

    path = write_memo(
        hook_env / "store",
        "a-memo-opening-with-a-rule",
        "# Notes\nvalid_from: 2026-01-01 was what the old run used.",
        "More prose.",
    )

    assert prompt_time.provenance(path) == "[undated]"


def test_a_line_of_correction_markers_is_read_in_linear_time(hook_env):
    """The stamp runs on every prompt, so no memo may make it quadratic.

    Red proof: restoring `CORRECTION_RE`'s gap to `[^\\n]*?` in `prompt_time` scans the line
    once per marker, and 20,000 markers take several seconds and fail the bound below.
    """

    path = write_memo(
        hook_env / "store", "markers", "name: m\ndescription: m", "\U0001f501" * 20_000
    )

    started = time.monotonic()
    stamp = prompt_time.provenance(path)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"provenance took {elapsed:.2f}s on one line of markers"
    assert stamp == "[undated]"


def test_a_head_of_blank_lines_is_searched_in_linear_time(hook_env):
    """A field regex must not let its indent run across newlines.

    Red proof: restoring the leading `^\\s*` of `VALID_FROM_RE`, `MODIFIED_RE` and `SESSION_RE`
    in `prompt_time` makes each search quadratic in the blank lines, and 40,000 take several
    seconds and fail the bound below.
    """

    path = write_memo(
        hook_env / "store", "blank-head", "\n" * 40_000 + "name: b", "Body."
    )

    started = time.monotonic()
    stamp = prompt_time.provenance(path)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"provenance took {elapsed:.2f}s on a head of blank lines"
    assert stamp == "[undated]"


def test_the_rendered_block_leads_each_hit_with_its_stamp_and_states_today(hook_env):
    """A date is only useful beside the date it is compared with, so `today` is in the block.

    Red proof: rendering each hit as `- {stem} (score ...)` without `provenance(path)` in
    `prompt_time.render` fails the first assertion.
    """

    from datetime import date

    store = hook_env / "store"
    write_memo(store, "2026-09-24-dated", "name: d\ndescription: d", "Body.")
    text = prompt_time.render(
        [("2026-09-24-dated", "summary", 3.0)], store, today=date(2026, 10, 14)
    )

    assert "- [written 2026-09-24] 2026-09-24-dated (score 3.0)" in text
    assert "Today is 2026-10-14." in text
