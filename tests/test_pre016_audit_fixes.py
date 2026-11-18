"""Fixes from the diff-scoped CCA audit of v0.15.0..master before 0.16.0 (run 20261118T201647Z).

- T1 BUG-001: a Lite store whose calibration does not certify, with the rescue active, does not
  ask the MCP server for a refresh every interval for an artifact that cannot be built.
- T2 ENV-001: a wizard re-run beside the Docker stack does not take the previous run's
  RECALL_ATOMIC_RESCUE_WHEN_MISSING, loaded from the project `.env`, for a shell choice.
- T3 CODE-001: the PostgreSQL rescue builder refuses a relative artifact root, as the search gate
  and the builders do.
- T4 DOC-001: beside the assistant surface, `recall_withdraw` says to name a memo by its memo_id.
- T5 STAKES-001: `recall_forget_memo` passes on a forget's warning when the embedding cache or the
  atomic rescue was not fully erased.
- T6 SEC-001: a Lite forget run without the variable still erases under the root a build used.
- T7 STAKES-002: a Lite write that replaces a memo's text erases the old text's vectors from the
  embedding cache and the rescue's view store, which a later forget can no longer find.

Red proof, run on the test host 2026-11-18 against master `b87b09d58` (the pre-fix production
code) with this file added, each failing in its own assertion:
T1 `test_an_uncertified_store_asks_for_no_rescue_refresh` at "an uncertified store asked again
for a refresh that changes nothing" (`_rescue_stale`; its control half is red at "a memory changed
since the last refresh asked for none" under the over-broad first fix, which refused every
uncertified store, as the differential review found); T2 `test_a_wizard_rerun_does_not_read_its_own_env_file_as_a_shell_choice`
at "'refuse' == 'skip'" (`_atomic_rescue_env`); T3 `test_the_rescue_builder_refuses_a_relative_root`
at "a relative root would build" (`RescueBuilder.from_env`); T4
`test_withdraw_beside_the_assistant_surface_names_memos_by_memo_id` at "'memo_id' in" (the
`recall_withdraw` declaration); T5 `test_forget_memo_passes_on_an_incomplete_rescue_erasure` at
"the incomplete erasure was not reported" (`AssistantMemos.forget`); T6
`test_a_lite_forget_without_the_variable_erases_under_the_root_a_build_used` at "assert (0 > 0)"
(`erase_forgotten`); T7 `test_a_lite_edit_erases_the_replaced_text_from_the_cache_and_the_view_store`
at "the edited memo's old text kept its cached vector" (`LiteStore.replace_sources`). The T2
control passed there, as it should. `test_the_record_of_rescue_roots_keeps_each_root_once` uses new
symbols (ImportError on that commit, not proof), so it was proved by mutation instead:
`unreached_rescue_roots` returning `[]` fails it at "assert [] == [...gone]".
`test_a_relative_root_is_never_recorded_nor_read_as_a_rescue_root` was added after the architect
gate found the first SEC-001 fix recording a root before `manifest_path` refused it. Against that
first fix (record before `manifest_path`, `LiteStore.rescue_roots` unfiltered) it fails at "a build
recorded a root it refused"; with only the `rescue_roots` filter removed, at "a relative entry was
read back".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_assistant_surface import _published, envs  # noqa: F401  # envs is a fixture


def test_an_uncertified_store_asks_for_no_rescue_refresh(tmp_path: Path, monkeypatch) -> None:
    """T1. Once a refresh has calibrated the memory as it is, an uncertified store asks for no
    other; a memory changed since (here a forget's delete) asks for one again, since a refit of it
    may certify (a control: a stale or missing fit after a forget must still heal)."""
    from recall.calibration_v2 import CalibrationStatus
    from recall.lite import LiteStore
    from recall_mcp.lite_refresh import LiteAutoRefresher
    from tests.test_lite_atomic_rescue import _blank_rescue_environment, _env
    from tests.test_lite_calibration import DIM, ContentWordEmbedder

    _blank_rescue_environment(monkeypatch)
    memos = tmp_path / "m"
    memos.mkdir()
    for k in range(3):  # far too little memory to certify
        (memos / f"memo{k}.md").write_text(" ".join(f"word{k}{j:02d}" for j in range(30)), encoding="utf-8")
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    refresher = LiteAutoRefresher(
        store, ContentWordEmbedder(), [memos], interval_s=30, wait_s=1, clock=lambda: 1000.0,
        env=_env(tmp_path / "atomic"),
    )
    try:
        refresher._run()  # the server's catch-up refresh, run here in the foreground
        refresher._run()  # it wrote the memos, so it is checked once more; this one writes nothing
        assert store.count() > 0
        assert store.resolve_calibration().status is not CalibrationStatus.CERTIFIED
        assert not refresher._rescue_stale(), "an uncertified store asked again for a refresh that changes nothing"
        assert store.delete_sources([str(memos / "memo0.md")]) > 0
        assert refresher._rescue_stale(), "a memory changed since the last refresh asked for none"
    finally:
        refresher.stop()
        store.close()


def _desktop_config(tmp_path: Path):
    from recall.wizard.headless import HeadlessConfig

    return HeadlessConfig(  # type: ignore[call-arg]
        embedder="hashing", corpus_version="2026-01-01", docs_root=tmp_path / "docs", code_root=tmp_path / "code",
        memory_root=tmp_path / "memory", dsn="postgresql://recall@localhost/recall",
        project_root=tmp_path / "project", data_root=tmp_path / "data",
    )


def test_a_wizard_rerun_does_not_read_its_own_env_file_as_a_shell_choice(tmp_path: Path, monkeypatch) -> None:
    """T2."""
    from recall.setup import _update_env_block
    from recall.wizard.headless import _atomic_rescue_env

    policy = "RECALL_ATOMIC_RESCUE_WHEN_MISSING"
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    _update_env_block(project / ".env", {policy: "refuse"})  # what a run without the stack wrote
    monkeypatch.setenv(policy, "refuse")  # and what the CLI loaded from it

    env, _note = _atomic_rescue_env(_desktop_config(tmp_path))
    assert env.get(policy) == "skip", f"the previous run's .env value was taken for a shell choice: {env}"


def test_a_shell_refuse_beside_the_stack_is_still_kept(tmp_path: Path, monkeypatch) -> None:
    """T2, a control: a value the project `.env` does not hold is the shell's and is kept."""
    from recall.wizard.headless import _atomic_rescue_env

    policy = "RECALL_ATOMIC_RESCUE_WHEN_MISSING"
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", raising=False)
    monkeypatch.setenv(policy, "refuse")

    env, _note = _atomic_rescue_env(_desktop_config(tmp_path))
    assert env.get(policy) == "refuse", env


def test_the_rescue_builder_refuses_a_relative_root(monkeypatch) -> None:
    """T3."""
    from recall.generation_store import GenerationStore
    from recall_mcp.rescue_builder import RescueBuilder

    store = object.__new__(GenerationStore)
    store._tenant = "t"  # type: ignore[attr-defined]
    base = {"RECALL_ATOMIC_RESCUE_MODE": "active", "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip"}
    absolute = str(Path.cwd().resolve() / "atomic")

    assert RescueBuilder.from_env(store, object(), {**base, "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": absolute}) is not None
    assert RescueBuilder.from_env(store, object(), {**base, "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": "relative/atomic"}) is None, (  # type: ignore[arg-type]
        "a relative root would build under the server's working directory"
    )


def test_withdraw_beside_the_assistant_surface_names_memos_by_memo_id(envs) -> None:  # noqa: F811
    """T4."""
    assistant, plain = envs
    assert "memo_id" in _published(assistant)["recall_withdraw"]["description"], (
        "recall_withdraw tells an assistant to pass a source field its search does not show"
    )
    assert "(the `source` field)" in _published(plain)["recall_withdraw"]["description"], "the plain text changed"


def test_forget_memo_passes_on_an_incomplete_rescue_erasure(tmp_path: Path, monkeypatch) -> None:
    """T5."""
    from recall.atomic_rescue import AtomicRescueArtifactError
    from recall.lite import atomic
    from recall.lite.calibration import auto_calibrate
    from recall_mcp.assistant import AssistantMemos
    from tests.test_lite_calibration import ContentWordEmbedder, _store

    store = _store(tmp_path, 48)
    try:
        assert auto_calibrate(store, ContentWordEmbedder()).artifact.certified
        memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)
        written = memos.remember(store, ContentWordEmbedder(), "The user drinks tea every morning.")

        def held(*_args: object, **_kwargs: object) -> tuple[int, int]:
            raise AtomicRescueArtifactError("an artifact directory is held open by another process")

        monkeypatch.setattr(atomic, "erase_forgotten", held)
        answer = memos.forget(store, ContentWordEmbedder(), written["memo_id"])
    finally:
        store.close()
    assert answer["deleted"] == 1, answer
    assert "atomic rescue" in answer.get("warning", ""), f"the incomplete erasure was not reported: {answer}"


def test_a_lite_forget_without_the_variable_erases_under_the_root_a_build_used(tmp_path: Path, monkeypatch) -> None:
    """T6."""
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest
    from recall.lite.atomic import VIEW_STORE_NAME, ensure_rescue_artifact, erase_forgotten
    from tests.test_lite_atomic_rescue import _blank_rescue_environment, _certified, _env

    _blank_rescue_environment(monkeypatch)
    custom = tmp_path / "elsewhere" / "atomic"  # set in the server's environment only
    store, embedder = _certified(tmp_path)
    try:
        built = ensure_rescue_artifact(store, embedder, env=_env(custom))
        assert built is not None and built.manifest is not None, built
        source = sorted({chunk.source for chunk in store.iter_chunks()})[0]
        texts = [text for text, _ in store.texts_for_sources([source])]
        store.delete_sources([source])

        erased, _removed = erase_forgotten(store, texts, [source], env={})  # a CLI shell without it
    finally:
        store.close()
    with AtomicViewStore(custom / VIEW_STORE_NAME) as views:
        left = [text for text in texts if views.erase_digests({chunk_view_digest(text)})]
    assert erased > 0 and not left, f"a forget left {len(left)} view set(s) under the build's root"


def test_a_lite_edit_erases_the_replaced_text_from_the_cache_and_the_view_store(tmp_path: Path, monkeypatch) -> None:
    """T7."""
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest
    from recall.cache import EmbeddingCache
    from recall.lite import LiteStore
    from recall.lite.atomic import VIEW_STORE_NAME, store_default_root
    from recall.types import Chunk

    cache_path = tmp_path / "cache.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(cache_path))
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", raising=False)
    old, new, same = "the user lives in Paris", "the user lives in Rome", "the user drinks tea"
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        store.replace_sources(
            ["a.md"],
            [Chunk(id="a1", source="a.md", text=old, metadata={}), Chunk(id="a2", source="a.md", text=same, metadata={})],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        root = store_default_root(store.path)
        root.mkdir()
        with AtomicViewStore(root / VIEW_STORE_NAME) as views:
            for text in (old, same):
                views.put("profile", chunk_view_digest(text), [[1.0, 0.0, 0.0, 0.0]])
            views.commit()
        with EmbeddingCache(cache_path) as cache:
            cache.put_many([("k-old", [1.0, 0.0, 0.0, 0.0]), ("k-same", [0.0, 1.0, 0.0, 0.0])], texts=[old, same])

        store.replace_sources(
            ["a.md"],
            [Chunk(id="a1", source="a.md", text=new, metadata={}), Chunk(id="a2", source="a.md", text=same, metadata={})],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )

    with EmbeddingCache(cache_path) as cache:
        cached = cache.get_many(["k-old", "k-same"])
    with AtomicViewStore(root / VIEW_STORE_NAME) as views:
        old_views = views.get("profile", chunk_view_digest(old), view_count=1, dimension=4)
        same_views = views.get("profile", chunk_view_digest(same), view_count=1, dimension=4)
    assert "k-old" not in cached, "the edited memo's old text kept its cached vector"
    assert old_views is None, "the edited memo's old text kept its micro-views"
    assert "k-same" in cached and same_views is not None, "a chunk the edit kept lost its vectors"


def test_the_record_of_rescue_roots_keeps_each_root_once(tmp_path: Path) -> None:
    """T6, the store half: a root is recorded once, and a root since removed is reported."""
    from recall.lite import LiteStore
    from recall.lite.atomic import unreached_rescue_roots

    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        here, gone = tmp_path / "here", tmp_path / "gone"
        here.mkdir()
        for root in (here, gone, here):
            store.record_rescue_root(root)
        assert store.rescue_roots() == sorted([str(here), str(gone)])
        assert unreached_rescue_roots(store) == [str(gone)]


def test_a_relative_root_is_never_recorded_nor_read_as_a_rescue_root(tmp_path: Path, monkeypatch) -> None:
    """T6, the validation half: a build under a relative root raises before it records one, and a
    relative entry already in the store's meta is not read back, since it would resolve against
    whatever directory a later forget runs in."""
    from recall.atomic_rescue import AtomicRescueArtifactError
    from recall.lite.atomic import ensure_rescue_artifact, unreached_rescue_roots
    from tests.test_lite_atomic_rescue import _blank_rescue_environment, _certified, _env

    _blank_rescue_environment(monkeypatch)
    store, embedder = _certified(tmp_path)
    try:
        with pytest.raises(AtomicRescueArtifactError, match="absolute"):
            ensure_rescue_artifact(store, embedder, env=_env(Path("relative/atomic")))
        assert store.rescue_roots() == [], "a build recorded a root it refused"
        store.record_rescue_root(Path("relative/atomic"))
        assert store.rescue_roots() == [] and unreached_rescue_roots(store) == [], "a relative entry was read back"
    finally:
        store.close()
