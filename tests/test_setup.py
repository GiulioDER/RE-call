from __future__ import annotations

import io
import pytest

from recall.setup import HardwareProbe, embedder_choices, run_setup_wizard


def test_embedder_choices_hide_cloud_when_security_is_required():
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    choices = embedder_choices(probe, security_required=True, cloud_keys={})
    labels = [c.label for c in choices]
    assert "voyage cloud" not in labels
    assert "openai compatible cloud" not in labels
    assert "hashing" in labels
    assert "fastembed" in labels


def test_setup_wizard_writes_env_and_accepts_api_keys(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("calibration should be skipped")),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "n",  # scaffold CLAUDE.md / memory/? declined
        "n",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "RECALL_DSN=postgresql://example/recall" in text
    assert "RECALL_SECURITY_REQUIRED=0" in text
    assert "VOYAGE_API_KEY=voyage-key" in text
    assert "OPENAI_API_KEY=openai-key" in text
    assert "OPENROUTER_API_KEY=openrouter-key" in text
    assert "RECALL_EMBEDDER=voyage:voyage-3" in text
    assert "RECALL_RERANK=0" in text
    assert "RECALL_SPARSE=fts" in text
    assert "RECALL_ENTAILMENT=0" in text
    assert "Calibration skipped" in output.getvalue()


def test_setup_wizard_skips_blank_calibration_inputs(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("calibration should be skipped")),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "n",  # scaffold CLAUDE.md / memory/? declined
        "y",
        "",
        "",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "RECALL_DSN=postgresql://example/recall" in text
    assert "RECALL_EMBEDDER=voyage:voyage-3" in text
    assert "RECALL_SPARSE=fts" in text
    assert "RECALL_ENTAILMENT=0" in text
    assert "Calibration skipped" in output.getvalue()


def test_setup_wizard_treats_calibration_directory_as_output_folder(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    (tmp_path / "nested").mkdir()
    seen = {}

    def fake_calibrate_from_files(**kw):
        seen["out"] = kw["out"]
        return type(
            "R",
            (),
            {
                "path": kw["out"],
                "calibration": type("C", (), {"threshold": 0.5})(),
                "report": None,
            },
        )()

    monkeypatch.setattr("recall.setup.calibrate_from_files", fake_calibrate_from_files)
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "n",  # scaffold CLAUDE.md / memory/? declined
        "y",
        str(tmp_path / "queries.json"),
        str(tmp_path / "corpus"),
        str(tmp_path / "nested"),
        "",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert seen["out"] == tmp_path / "nested" / "calibration.json"
    assert "Calibration saved to" in output.getvalue()


def test_setup_wizard_rejects_windows_host_path_for_calibration_output(tmp_path, monkeypatch):
    monkeypatch.setattr("recall.setup.os.name", "posix")
    from recall.setup import _require_local_output_path

    with pytest.raises(ValueError, match="Calibration output path looks like a Windows host path"):
        _require_local_output_path(
            "C:\\Users\\gde00\\Music",
            label="Calibration output path",
            default=tmp_path / "calibration.json",
        )


def test_setup_wizard_can_enable_entailment_judge(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("calibration should be skipped")),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "y",
        "1",
        "n",  # scaffold CLAUDE.md / memory/? declined
        "n",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "RECALL_ENTAILMENT=1" in text
    assert "RECALL_ENTAILMENT_MODEL=cross-encoder/qnli-distilroberta-base" in text
    assert "RECALL_ENTAILMENT_REVISION=7dd04ee0a6040c06fb381ad7edcb8585f4d937fd" in text
    assert "Calibration skipped" in output.getvalue()


def test_splade_choice_requires_cuda_gpu():
    from recall.setup import sparse_choices

    no_cuda = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    with_cuda = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=True,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    assert all(choice.label != "splade" for choice in sparse_choices(no_cuda))
    assert any(choice.label == "splade" for choice in sparse_choices(with_cuda))


def test_update_markdown_block_creates_file_when_absent(tmp_path):
    from recall.setup import _update_markdown_block

    path = tmp_path / "CLAUDE.md"
    _update_markdown_block(path, "<!-- begin -->", "<!-- end -->", "hello")

    text = path.read_text(encoding="utf-8")
    assert text == "<!-- begin -->\nhello\n<!-- end -->\n"


def test_update_markdown_block_appends_when_markers_absent(tmp_path):
    from recall.setup import _update_markdown_block

    path = tmp_path / "CLAUDE.md"
    path.write_text("# My project\n\nSome notes.\n", encoding="utf-8")

    _update_markdown_block(path, "<!-- begin -->", "<!-- end -->", "hello")

    text = path.read_text(encoding="utf-8")
    assert text.startswith("# My project\n\nSome notes.\n")
    assert "<!-- begin -->\nhello\n<!-- end -->" in text


def test_update_markdown_block_replaces_in_place_on_rerun(tmp_path):
    from recall.setup import _update_markdown_block

    path = tmp_path / "CLAUDE.md"
    path.write_text(
        "# My project\n\n<!-- begin -->\nold\n<!-- end -->\n\nTrailer.\n",
        encoding="utf-8",
    )

    _update_markdown_block(path, "<!-- begin -->", "<!-- end -->", "new")

    text = path.read_text(encoding="utf-8")
    assert "old" not in text
    assert "<!-- begin -->\nnew\n<!-- end -->" in text
    assert text.startswith("# My project\n")
    assert text.rstrip().endswith("Trailer.")


def test_scaffold_claude_md_creates_file(tmp_path):
    from recall.setup import scaffold_claude_md

    path = tmp_path / "CLAUDE.md"
    scaffold_claude_md(path)

    text = path.read_text(encoding="utf-8")
    assert "<!-- recall setup begin -->" in text
    assert "recall_search" in text
    assert "recall_evidence" in text
    assert "memory/MEMORY.md" in text


def test_scaffold_claude_md_rerun_replaces_block_only(tmp_path):
    from recall.setup import scaffold_claude_md

    path = tmp_path / "CLAUDE.md"
    path.write_text("# My project\n\nCustom notes that must survive.\n", encoding="utf-8")

    scaffold_claude_md(path)
    scaffold_claude_md(path)

    text = path.read_text(encoding="utf-8")
    assert text.count("<!-- recall setup begin -->") == 1
    assert "Custom notes that must survive." in text


def test_scaffold_memory_index_creates_directory_and_file(tmp_path):
    from recall.setup import scaffold_memory_index

    memory_dir = tmp_path / "memory"
    created = scaffold_memory_index(memory_dir)

    assert created is True
    assert memory_dir.is_dir()
    text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "type: user | feedback | project | reference" in text


def test_scaffold_memory_index_leaves_existing_file_untouched(tmp_path):
    from recall.setup import scaffold_memory_index

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("# Real user facts\n", encoding="utf-8")

    created = scaffold_memory_index(memory_dir)

    assert created is False
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8") == "# Real user facts\n"


def test_index_memory_directory_skips_in_production(monkeypatch, tmp_path):
    from recall.setup import index_memory_directory

    monkeypatch.setenv("RECALL_ENV", "production")
    output = io.StringIO()

    index_memory_directory(
        dsn="postgresql://example/recall",
        embedder_name="hashing",
        memory_dir=tmp_path / "memory",
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert "Skipping auto-index" in output.getvalue()


def test_index_memory_directory_indexes_via_indexer(monkeypatch, tmp_path):
    from recall.index import IndexStats
    from recall.setup import index_memory_directory

    monkeypatch.delenv("RECALL_ENV", raising=False)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("# Memory index\n", encoding="utf-8")

    calls = {}

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def check_schema(self):
            calls["checked"] = True

    class FakeIndexer:
        def __init__(self, store, embedder, chunker=None):
            calls["store"] = store
            calls["embedder"] = embedder

        def index_path(self, path, glob=None):
            calls["path"] = path
            calls["glob"] = glob
            return IndexStats(files=1, chunks=3)

    monkeypatch.setattr("recall.store.PgVectorStore", lambda *a, **k: FakeStore())
    monkeypatch.setattr("recall.index.Indexer", FakeIndexer)
    output = io.StringIO()

    index_memory_directory(
        dsn="postgresql://example/recall",
        embedder_name="hashing",
        memory_dir=memory_dir,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert calls["path"] == memory_dir
    assert calls["glob"] == "**/*.md"
    assert "Indexed 3 chunks from 1 files" in output.getvalue()


def test_index_memory_directory_survives_indexing_failure(monkeypatch, tmp_path):
    from recall.setup import index_memory_directory

    monkeypatch.delenv("RECALL_ENV", raising=False)

    def boom(*a, **k):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr("recall.store.PgVectorStore", boom)
    output = io.StringIO()

    index_memory_directory(
        dsn="postgresql://example/recall",
        embedder_name="hashing",
        memory_dir=tmp_path / "memory",
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert "Could not auto-index" in output.getvalue()
    assert "db unreachable" in output.getvalue()


def test_index_memory_directory_survives_embedder_resolution_failure(monkeypatch, tmp_path):
    from recall.setup import index_memory_directory

    monkeypatch.delenv("RECALL_ENV", raising=False)

    def boom(*a, **k):
        raise ValueError("unknown embedder")

    monkeypatch.setattr("recall.setup.resolve_embedder", boom)
    output = io.StringIO()

    index_memory_directory(
        dsn="postgresql://example/recall",
        embedder_name="invalid_embedder",
        memory_dir=tmp_path / "memory",
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert "Could not auto-index" in output.getvalue()
    assert "unknown embedder" in output.getvalue()


def test_setup_wizard_scaffolds_claude_md_and_memory_and_indexes(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("calibration should be skipped")),
    )
    index_calls = {}
    monkeypatch.setattr(
        "recall.setup.index_memory_directory",
        lambda **kw: index_calls.update(kw),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "y",  # scaffold CLAUDE.md / memory/? accepted
        "n",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    claude_md_path = tmp_path / "CLAUDE.md"
    memory_dir = tmp_path / "memory"

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        claude_md_path=claude_md_path,
        memory_dir=memory_dir,
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert "recall_search" in claude_md_path.read_text(encoding="utf-8")
    assert (memory_dir / "MEMORY.md").exists()
    assert index_calls["memory_dir"] == memory_dir
    assert index_calls["embedder_name"] == "voyage:voyage-3"


def test_setup_wizard_survives_scaffold_failure_and_still_writes_env(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("calibration should be skipped")),
    )
    monkeypatch.setattr(
        "recall.setup.scaffold_claude_md",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("permission denied")),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "y",  # scaffold CLAUDE.md / memory/? accepted, but scaffold_claude_md raises
        "n",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    claude_md_path = tmp_path / "CLAUDE.md"
    memory_dir = tmp_path / "memory"

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        claude_md_path=claude_md_path,
        memory_dir=memory_dir,
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "RECALL_DSN=postgresql://example/recall" in text
    assert "RECALL_EMBEDDER=voyage:voyage-3" in text
    assert "RECALL_SPARSE=fts" in text
    assert "RECALL_ENTAILMENT=0" in text
    assert "Could not scaffold" in output.getvalue()


def test_setup_wizard_skips_scaffold_when_declined(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("calibration should be skipped")),
    )
    monkeypatch.setattr(
        "recall.setup.index_memory_directory",
        lambda **kw: (_ for _ in ()).throw(AssertionError("index should be skipped")),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "n",  # scaffold CLAUDE.md / memory/? declined
        "n",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    claude_md_path = tmp_path / "CLAUDE.md"
    memory_dir = tmp_path / "memory"

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        claude_md_path=claude_md_path,
        memory_dir=memory_dir,
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert not claude_md_path.exists()
    assert not memory_dir.exists()


def test_setup_wizard_scaffold_prompt_defaults_to_yes_on_blank_answer(tmp_path, monkeypatch):
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("calibration should be skipped")),
    )
    index_calls = {}
    monkeypatch.setattr(
        "recall.setup.index_memory_directory",
        lambda **kw: index_calls.update(kw),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "",  # scaffold CLAUDE.md / memory/? blank answer takes the default (yes)
        "n",
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    claude_md_path = tmp_path / "CLAUDE.md"
    memory_dir = tmp_path / "memory"

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        claude_md_path=claude_md_path,
        memory_dir=memory_dir,
        input_fn=fake_input,
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    assert claude_md_path.exists()
    assert (memory_dir / "MEMORY.md").exists()
    assert index_calls["memory_dir"] == memory_dir


def test_setup_wizard_still_scaffolds_when_calibration_output_path_is_invalid(
    tmp_path, monkeypatch
):
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(ValueError("bad calibration input")),
    )
    index_calls = {}
    monkeypatch.setattr(
        "recall.setup.index_memory_directory",
        lambda **kw: index_calls.update(kw),
    )
    answers = iter([
        "n",
        "voyage-key",
        "openai-key",
        "openrouter-key",
        "4",
        "1",  # reranker menu. The sparse backend is not prompted: only one
        "n",
        "y",  # scaffold CLAUDE.md / memory/? accepted
        "y",  # calibrate now? accepted
        str(tmp_path / "queries.json"),
        str(tmp_path / "corpus"),
        "",  # calibration output path, default — calibrate_from_files raises ValueError
    ])
    output = io.StringIO()

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    claude_md_path = tmp_path / "CLAUDE.md"
    memory_dir = tmp_path / "memory"

    with pytest.raises(SystemExit) as exc_info:
        run_setup_wizard(
            dsn="postgresql://example/recall",
            env_path=tmp_path / ".env",
            claude_md_path=claude_md_path,
            memory_dir=memory_dir,
            input_fn=fake_input,
            print_fn=lambda *a, **k: print(*a, **k, file=output),
        )

    assert exc_info.value.code == 2
    # A bad calibration path aborts the wizard before .env is ever written, unchanged from
    # before this refactor — but a scaffold the user separately asked for must not be silently
    # dropped just because a later, unrelated step failed.
    assert not (tmp_path / ".env").exists()
    assert claude_md_path.exists()
    assert (memory_dir / "MEMORY.md").exists()
    assert index_calls["memory_dir"] == memory_dir


def test_a_sole_backend_is_reported_rather_than_offered_as_a_menu(tmp_path, monkeypatch):
    """A menu of one is not a choice, so the wizard states it and moves on.

    This probe has no CUDA and no sentence-transformers, so reranking and the sparse backend each
    collapse to their baseline. The answer list below deliberately provides nothing for either.
    Delete the `sole_note` branch in `_choose` and this goes red: the wizard prompts twice more
    and runs the iterator dry.
    """
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=False,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    answers = iter([
        "n",  # security
        "", "", "",  # the three API keys
        "1",  # embedder
        "n",  # scaffold
        "n",  # calibrate
    ])
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert "Reranking stays off, because sentence-transformers is not installed" in text
    assert "SPLADE is unavailable" in text
    assert "Choose the sparse retrieval backend:" not in text
    assert "Choose whether to enable reranking:" not in text
    # Every answer was consumed by the prompt it was written for, and no prompt was answered
    # with a value meant for another one.
    assert next(answers, None) is None
    assert "Please answer yes or no." not in text
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "RECALL_RERANK=0" in env
    assert "RECALL_SPARSE=fts" in env


def test_a_real_choice_is_still_offered_as_a_menu(tmp_path, monkeypatch):
    """The shortcut must not swallow a decision the machine can actually make.

    With CUDA and sentence-transformers present, sparse has two backends and reranking has three,
    so both menus appear and both answers are consumed. Widen the `sole_note` branch to fire on
    more than one choice and this goes red.
    """
    probe = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=True,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    answers = iter([
        "n",  # security
        "", "", "",  # the three API keys
        "1",  # embedder
        "1",  # reranker menu, genuinely offered
        "2",  # sparse menu, genuinely offered: splade
        "n",  # entailment
        "n",  # scaffold
        "n",  # calibrate
    ])
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert "Choose the sparse retrieval backend:" in text
    assert "SPLADE is unavailable" not in text
    assert next(answers, None) is None
    assert "RECALL_SPARSE=splade" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_a_sole_embedder_says_what_it_costs_rather_than_offering_a_menu(tmp_path, monkeypatch):
    """The embedder list can collapse too, and hashing is the one collapse that costs quality.

    A menu of one never told anyone that. Drop the `sole_note` from the embedder call site and
    this goes red: the wizard prompts for a menu the answer list does not answer.
    """
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=False,
        sentence_transformers_available=False,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    answers = iter([
        "n",  # security
        "", "", "",  # the three API keys, all blank, so no cloud embedder appears
        "n",  # scaffold
        "n",  # calibrate
    ])
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert "Embedder: hashing, the only one this machine can use" in text
    assert "retrieves noticeably worse than a real model" in text
    assert "Choose the embedder you want to use:" not in text
    assert next(answers, None) is None
    assert "RECALL_EMBEDDER=hashing" in (tmp_path / ".env").read_text(encoding="utf-8")
