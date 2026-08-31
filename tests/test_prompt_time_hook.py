"""The `UserPromptSubmit` memory-recall hook: the properties that stop it breaking a session.

This hook runs before EVERY user turn, and the thing it sits on is the user's own message. So its
failure modes are not "it ranked badly", they are "the prompt never reached the model" and "the
session died on a hook". Those are what is asserted here, alongside the store resolution, which is
the part that silently returns nothing when it is wrong.

Nothing here needs a database or a network: the whole point of this hook is that it reads files.
"""

from __future__ import annotations

import json
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


@pytest.mark.skipif(not DEPLOYED.is_dir(), reason="no deployed copy on this machine")
@pytest.mark.parametrize("module", ["__init__.py", "prompt_time.py", "write_time.py", "__main__.py"])
def test_the_deployed_copy_matches_the_source(module):
    """⚠️ CLAUDE.md records `session_start_hook.py` and its deployed twin drifting by 57 lines
    with nothing reporting it. This hook runs from a deployed copy on purpose, because `-m`
    otherwise resolves the package out of whatever branch the cwd happens to be, so a forgotten
    redeploy is the way this feature silently reverts. A failure here means: copy it again.

    Skipped where there is no deployed copy, which is every machine but the author's and CI.
    """

    source = Path(__file__).resolve().parent.parent / "recall_hooks" / module
    assert (DEPLOYED / module).read_bytes() == source.read_bytes(), (
        f"{module} in ~/.claude/hooks/recall_hooks differs from the source; redeploy it"
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
