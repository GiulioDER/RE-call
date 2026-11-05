"""Regression tests for the second DEEP CCA hunt of the first-run code: the wizard's corpus
pipeline, its query-set generators, its embedder identity, its database preflight and the terminal
interview.

Each test names the finding it pins. Red proof for every test is taken on the remote test host
against the pre-fix snapshot of the production files, and the verdicts are recorded in the pull
request. A test that needs a symbol the fix added imports it after its first assertion, so the
pre-fix run fails on behaviour rather than on the import.

No database, no network, no Docker: every collaborator is a fake, a stub or a monkeypatch.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.wizard.corpora import code_corpus, docs_corpus
from recall.wizard.queryset import MIN_PER_CLASS, QuerySetError


def _markdown_corpus(root: Path, files: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for n in range(files):
        (root / f"doc{n}.md").write_text(f"# Note {n}\n\nsome text\n", encoding="utf-8")
    return root


# ----------------------------------------------------------------------------------------------
# DAT-010: a file that vanishes during the inventory must not be built over
# ----------------------------------------------------------------------------------------------


class _BuildReached(Exception):
    """Raised by the stubbed build, so a run that gets that far is visible to the assertion."""


def test_a_file_that_vanishes_during_the_inventory_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DAT-010. Invariant: `run_corpus` builds only over a manifest that lists every file the
    walk found; a file that disappears between the walk and the read refuses the run with a
    `PipelineRefusal` before `build_generation`. Failure caught: `build_inventory` dropped the
    `vanished` count, so a generation was built, calibrated and promoted over fewer files than the
    corpus holds, with nothing saying so. Target: `recall.wizard.pipeline.run_corpus` (its
    manifest step, which now reads `build_inventory_report`)."""
    import recall.wizard.inventory as inventory
    import recall.wizard.pipeline as pipeline

    spec = docs_corpus(_markdown_corpus(tmp_path / "docs", files=3))
    entries = [{"query": f"answerable {n}", "answerable": True} for n in range(MIN_PER_CLASS)] + [
        {"query": f"gap {n}", "answerable": False} for n in range(MIN_PER_CLASS)
    ]
    monkeypatch.setattr(pipeline, "_labelled_set", lambda *a, **k: (entries, MIN_PER_CLASS))

    real_entry = inventory._entry

    def _entry(path: Path) -> dict[str, Any]:
        if path.name == "doc1.md":
            raise FileNotFoundError(path)
        return real_entry(path)

    monkeypatch.setattr(inventory, "_entry", _entry)

    def _build(*_args: Any, **_kwargs: Any) -> Any:
        raise _BuildReached("the build ran over a manifest missing a vanished file")

    monkeypatch.setattr(pipeline, "build_generation", _build)
    manager = SimpleNamespace(environment="development", tenant_id=spec.tenant)

    with pytest.raises((pipeline.PipelineRefusal, _BuildReached)) as caught:
        pipeline.run_corpus(
            spec,
            manager=manager,  # type: ignore[arg-type]
            calibrations=object(),  # type: ignore[arg-type]
            embedder=object(),  # type: ignore[arg-type]
            corpus_version="2026-11-05",
        )
    assert caught.type is pipeline.PipelineRefusal, (
        "a corpus that changed under the build must be refused, not built short"
    )
    assert "1 file(s) disappeared" in str(caught.value)


# ----------------------------------------------------------------------------------------------
# PERF-005: the retry at the certification floor must not read the corpus again
# ----------------------------------------------------------------------------------------------


def test_the_corpus_is_read_once_when_the_query_set_falls_back_to_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-005. Invariant: `_labelled_set` walks, reads and chunks the corpus once, however many
    per-class sizes it tries. Failure caught: `chunks_from_directory` sat inside the retry loop,
    so the fallback to `MIN_PER_CLASS` (the path a large code corpus takes) re-read the whole
    corpus. Target: `recall.wizard.pipeline._labelled_set`."""
    import recall.wizard.pipeline as pipeline

    reads: list[str] = []

    def _chunks(root: Any, glob: str, chunker: Any) -> list[str]:
        reads.append(glob)
        return ["chunk"]

    def _generate(chunks: Any, *, per_class: int, seed: int) -> list[dict[str, Any]]:
        if per_class != MIN_PER_CLASS:
            raise QuerySetError(f"cannot produce {per_class} per class")
        return [{"query": "q", "answerable": True}]

    monkeypatch.setattr(pipeline, "chunks_from_directory", _chunks)
    monkeypatch.setattr(pipeline, "generate_offline", _generate)
    monkeypatch.setattr(pipeline, "canonicalize", lambda entries: entries)
    spec = SimpleNamespace(root=tmp_path, glob="*.md", tenant="default-docs")

    _entries, used = pipeline._labelled_set(
        spec, lambda text: [text], per_class=MIN_PER_CLASS * 2, seed=0  # type: ignore[arg-type]
    )

    assert used == MIN_PER_CLASS, "precondition: the fallback attempt is the one that succeeded"
    assert len(reads) == 1, f"the corpus was read {len(reads)} times for one query set"

    # The read is still converted: a missing root is a refusal, not a raw QuerySetError.
    def _missing(root: Any, glob: str, chunker: Any) -> list[str]:
        raise QuerySetError("'/nowhere' does not exist")

    monkeypatch.setattr(pipeline, "chunks_from_directory", _missing)
    with pytest.raises(pipeline.PipelineRefusal, match="does not exist"):
        pipeline._labelled_set(spec, lambda text: [text], per_class=MIN_PER_CLASS, seed=0)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------
# PERF-006: one embedder is hashed once, however many corpora it builds
# ----------------------------------------------------------------------------------------------


class _Onnx:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = str(model_dir)


class _Inner:
    def __init__(self, model_dir: Path) -> None:
        self.model = _Onnx(model_dir)


class FastEmbedEmbedder:  # noqa: N801 - the identity guard matches on this exact class name
    """Shaped and named like the real one, so `artifact_identity_for` reaches the digest."""

    name = "BAAI/bge-small-en-v1.5"
    dim = 384

    def __init__(self, model_dir: Path) -> None:
        self._model = _Inner(model_dir)


def test_one_embedder_is_hashed_once_for_two_corpora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-006. Invariant: the model directory of one embedder instance is digested once per
    process, so a docs and a code corpus built with the same embedder share one hash. Failure
    caught: `_identified_request` re-streamed every file of the model (over a gigabyte for
    bge-large) for each calibrated corpus. Target: `recall.wizard.identity.artifact_identity_for`,
    reached through its consumer `recall.wizard.pipeline._identified_request`."""
    import recall.wizard.identity as identity
    import recall.wizard.pipeline as pipeline

    model = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q" / "snapshots" / ("5" * 40)
    model.mkdir(parents=True)
    (model / "model_optimized.onnx").write_bytes(b"weights" * 100)

    hashed: list[Path] = []
    real_digest = identity._artifact_digest

    def _digest(directory: Path) -> str:
        hashed.append(directory)
        return real_digest(directory)

    monkeypatch.setattr(identity, "_artifact_digest", _digest)
    embedder = FastEmbedEmbedder(model)
    docs = docs_corpus(_markdown_corpus(tmp_path / "docs", files=1))
    code = code_corpus(_markdown_corpus(tmp_path / "code", files=1))

    first = pipeline._identified_request(docs, embedder, None)  # type: ignore[arg-type]
    second = pipeline._identified_request(code, embedder, None)  # type: ignore[arg-type]

    assert first.artifact_digest is not None and first.artifact_digest == second.artifact_digest
    assert len(hashed) == 1, f"the same model directory was hashed {len(hashed)} times"


# ----------------------------------------------------------------------------------------------
# PERF-011: generate_offline tokenizes the corpus once, not once per helper
# ----------------------------------------------------------------------------------------------


def test_the_offline_generator_never_tokenizes_the_whole_corpus_in_one_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PERF-011. Invariant: `generate_offline` takes the corpus vocabulary from its document
    frequency, so `word_tokens` only ever sees one text at a time and no list of every word
    occurrence in the corpus is built. Failure caught: `offtopic_subjects_absent_from` called
    `word_tokens(chunks)` over the whole corpus before `_document_frequency` tokenized every chunk
    again. Target: `recall.wizard.queryset.generate_offline`."""
    import recall.wizard.queryset as queryset

    widths: list[int] = []
    real_tokens = queryset.word_tokens

    def _tokens(texts: Any) -> list[str]:
        texts = list(texts)
        widths.append(len(texts))
        return real_tokens(texts)

    monkeypatch.setattr(queryset, "word_tokens", _tokens)
    chunks = [
        f"# {word.capitalize()} handbook\n\n" + f"{word} " * 10 + "the system stores records"
        for word in ("kaloium", "mineium", "puraium", "setiium")
    ]

    entries = queryset.generate_offline(chunks, per_class=2, seed=0)

    assert sum(1 for e in entries if e["answerable"]) == 2, "precondition: the set was produced"
    assert max(widths) == 1, (
        f"word_tokens was handed {max(widths)} texts at once, so the corpus was tokenized whole"
    )


# ----------------------------------------------------------------------------------------------
# ENV-012: a remote database named in keyword form is remote
# ----------------------------------------------------------------------------------------------


def test_a_keyword_dsn_naming_a_remote_host_is_not_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-012. Invariant: `is_local_host` reads the DSN as libpq does, with the credential
    guard's own predicate, so a keyword DSN or a `?host=` URI naming a remote host is remote.
    Failure caught: `urlsplit` saw no hostname in either form, and an empty hostname was in the
    local list, so `host=db.example.com dbname=recall` was reported as this machine. Target:
    `recall.wizard.database.is_local_host`."""
    from recall.wizard.database import is_local_host

    for name in ("PGHOST", "PGHOSTADDR", "PGSERVICE"):
        monkeypatch.delenv(name, raising=False)

    assert is_local_host("host=db.example.com dbname=recall") is False
    assert is_local_host("postgresql:///recall?host=db.example.com") is False
    # The two predicates now agree on the rest of the loopback range as well.
    assert is_local_host("postgresql://u@127.0.0.2:5432/recall") is True


# ----------------------------------------------------------------------------------------------
# CODE-008: the terminal interview shows the connection example once
# ----------------------------------------------------------------------------------------------


def test_the_terminal_interview_shows_the_connection_example_once(tmp_path: Path) -> None:
    """CODE-008. Invariant: the DSN prompt is the declared `Question.prompt`, so the example in
    the question's help is printed once. Failure caught: `_ask_working_dsn` hardcoded its own
    prompt that restated the example the loop had already printed as help, so it appeared twice
    and an edit to the declared question reached the GUI only. Target:
    `recall.wizard.interactive._ask_working_dsn`."""
    from recall.wizard.database import DatabaseReport, Finding
    from recall.wizard.interactive import Prompter, ask_config

    transcript: list[str] = []
    answers = ["p", "hashing", "existing", "postgresql://me:pw@127.0.0.1:5432/mine", "", "", "", "", ""]

    def _read(prompt: str) -> str:
        transcript.append(prompt)
        return answers.pop(0)

    def _usable(dsn: str, *, expected_dimension: int | None = None) -> DatabaseReport:
        return DatabaseReport(
            dsn=dsn, findings=(Finding(name="reachable", ok=True, detail="connected"),)
        )

    ask_config(
        Prompter(read=_read, write=transcript.append), default_root=tmp_path, probe=_usable
    )

    shown = "\n".join(transcript)
    example = "postgresql://user:password@127.0.0.1:5432/recall"
    assert shown.count(example) == 1, f"the example was shown {shown.count(example)} times"


# ----------------------------------------------------------------------------------------------
# DOC-007: the printed reclaim advice must actually reclaim
# ----------------------------------------------------------------------------------------------


def test_the_reclaim_advice_names_a_gc_that_collects_a_generation_built_today(
    tmp_path: Path,
) -> None:
    """DOC-007. Invariant: the gc command printed for an uncertified generation passes
    `--retention-days 0`. Failure caught: `abandon` sets no retirement time, so gc ages the
    generation from its creation, and the default seven day retention kept the generation the
    wizard had just built, so the two printed commands reclaimed nothing. Target:
    `recall.wizard.pipeline._degraded_reason`."""
    from recall.wizard.pipeline import _degraded_reason

    spec = docs_corpus(_markdown_corpus(tmp_path / "docs", files=1))
    reason = _degraded_reason(spec, "gen_1", "separability below the bar", None)
    commands = [part for part in reason.split("`")[1::2] if part.startswith("recall ")]
    gc = [shlex.split(command) for command in commands if " gc" in command]

    assert gc == [
        ["recall", "--tenant", spec.tenant, "generation", "gc", "--retention-days", "0"]
    ], gc
