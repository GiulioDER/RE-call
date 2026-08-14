from __future__ import annotations

import io
import pytest

from recall.setup import (
    HardwareProbe,
    LOCAL_PROVIDER,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    embedder_choices,
    reasoning_provider_choices,
    run_setup_wizard,
)


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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
    # splade is listed either way now, because hiding it makes the product look like it does not
    # have the feature. What CUDA governs is whether it can be selected and run.
    without = next(c for c in sparse_choices(no_cuda) if c.label == "splade")
    withit = next(c for c in sparse_choices(with_cuda) if c.label == "splade")
    assert without.available is False
    assert "CUDA" in without.unavailable_note
    assert withit.available is True


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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "6",  # embedder: voyage cloud, now 6th since bge base and large were added
        "1",  # reranker menu
        "1",  # sparse backend menu
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
        "1",  # reranker menu, still offered because unavailable options are listed
        "1",  # sparse backend menu, same
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


def _no_extras_probe() -> HardwareProbe:
    """A default `pip install "recall-rag[fastembed]"` machine: no sentence-transformers, no CUDA."""
    return HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=10_000_000_000,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=False,
    )


def test_options_this_machine_cannot_run_are_still_listed_and_marked(tmp_path, monkeypatch) -> None:
    """Hiding them makes the product look like it lacks the feature, and leaves no way to ask.

    Drop the `(not installed yet)` suffix in `_choose` and the marker assertion goes red; stop
    appending the unavailable options in `reranker_choices` and the label assertions go red.
    """
    monkeypatch.setattr("recall.setup.probe_hardware", _no_extras_probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    answers = iter(["n", "", "", "", "1", "1", "1", "n", "n"])
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert "ms marco reranker" in text
    assert "bge reranker" in text
    assert "splade" in text
    assert "(not installed yet)" in text
    assert next(answers, None) is None


def test_picking_an_unavailable_reranker_explains_and_keeps_the_baseline(
    tmp_path, monkeypatch
) -> None:
    """The setting is deliberately NOT written: the module is absent, so it would fail at query
    time, in front of whoever inherits the deployment rather than the person who chose it.

    Delete the availability branch in `_choose` and this goes red with RECALL_RERANK=1.
    """
    monkeypatch.setattr("recall.setup.probe_hardware", _no_extras_probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    answers = iter(["n", "", "", "", "1", "2", "1", "n", "n"])  # reranker 2 is not installed
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert 'pip install "recall-rag[rerank]"' in text
    assert "Keeping none for now." in text
    assert "RECALL_RERANK=0" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert next(answers, None) is None


def test_picking_unavailable_splade_explains_and_keeps_postgres_fts(tmp_path, monkeypatch) -> None:
    """Same rule on the sparse menu, where the baseline is `postgres fts` rather than `none`."""
    monkeypatch.setattr("recall.setup.probe_hardware", _no_extras_probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    answers = iter(["n", "", "", "", "1", "1", "2", "n", "n"])  # sparse 2 is not installed
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert 'pip install "recall-rag[sparse]"' in text
    assert "Keeping postgres fts for now." in text
    assert "RECALL_SPARSE=fts" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert next(answers, None) is None


def test_the_note_names_the_condition_that_actually_failed() -> None:
    """Telling somebody to install what they already have sends them to fix the wrong thing.

    `runnable` folds together the package check and the disk/network check, so a fixed string
    blaming sentence-transformers is wrong whenever the real blocker is disk. Replace
    `_why_unavailable(probe)` with a constant and the disk case goes red.
    """
    from recall.setup import reranker_choices, sparse_choices

    no_package = HardwareProbe(
        cpu_count=8, gpu=None, cuda_available=False, free_bytes=10_000_000_000,
        internet=True, fastembed_available=True, sentence_transformers_available=False,
    )
    # Has the package, but no room to download the weights.
    no_disk = HardwareProbe(
        cpu_count=8, gpu=None, cuda_available=False, free_bytes=1_000_000,
        internet=True, fastembed_available=True, sentence_transformers_available=True,
    )

    missing_pkg = next(c for c in reranker_choices(no_package, security_required=False)
                       if c.label == "ms marco reranker")
    assert "sentence-transformers is not installed" in missing_pkg.unavailable_note
    assert "free disk" not in missing_pkg.unavailable_note

    missing_disk = next(c for c in reranker_choices(no_disk, security_required=False)
                        if c.label == "ms marco reranker")
    assert missing_disk.available is False
    assert "free disk" in missing_disk.unavailable_note
    assert "sentence-transformers is not installed" not in missing_disk.unavailable_note

    splade = next(c for c in sparse_choices(no_disk) if c.label == "splade")
    assert "no CUDA device was detected" in splade.unavailable_note


def test_a_choice_list_whose_baseline_cannot_run_is_refused() -> None:
    """The fallback returns choices[0], so an unrunnable first entry would write a broken value.

    Delete the guard in `_choose` and this goes red, silently returning the unrunnable baseline.
    """
    from recall.setup import Choice, _choose

    broken = [
        Choice(label="a", value="A", description="", available=False, unavailable_note="n"),
        Choice(label="b", value="B", description=""),
    ]
    with pytest.raises(ValueError, match="must be runnable"):
        _choose(lambda _p="": "1", lambda *a, **k: None, "Pick:", broken)


def _roomy_probe(free_bytes: int) -> HardwareProbe:
    return HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=free_bytes,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=False,
    )


def test_the_bigger_bge_models_appear_only_with_room_for_them() -> None:
    """bge-large is 1.2 GB against the shared 1.5 GB floor, so the floor alone is not enough.

    Drop the per-model size gates and the 1.6 GB case starts offering a download that cannot
    finish, so this goes red.
    """
    from recall.setup import embedder_choices

    roomy = [c.label for c in embedder_choices(
        _roomy_probe(10 * 1024**3), security_required=False, cloud_keys={})]
    tight = [c.label for c in embedder_choices(
        _roomy_probe(1_600_000_000), security_required=False, cloud_keys={})]

    assert "fastembed base" in roomy and "fastembed large" in roomy
    assert "fastembed" in tight  # bge-small is 67 MB, it still fits
    assert "fastembed base" not in tight
    assert "fastembed large" not in tight


def test_every_embedder_option_declares_its_width() -> None:
    """The dimension guard is only as good as the widths it compares, and a missing one is silent."""
    from recall.setup import embedder_choices

    choices = embedder_choices(
        _roomy_probe(10 * 1024**3), security_required=False, cloud_keys={})
    assert choices, "no embedders offered"
    assert all(c.dim for c in choices), [c.label for c in choices if not c.dim]


def test_setup_auto_prepares_an_empty_mismatched_table(tmp_path, monkeypatch):
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: _roomy_probe(10 * 1024**3))
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    monkeypatch.setattr(
        "recall.setup._schema_prepare_state",
        lambda dsn, dim, table=None: (
            "conflict",
            "recall_chunks_v1 uses vector(384), requested dimension is 1024",
        ),
    )
    monkeypatch.setattr(
        "recall.setup._table_row_counts",
        lambda dsn, tables: {name: 0 for name in tables},
    )
    dropped: list[str] = []
    applied: list[tuple[str, str, int]] = []

    monkeypatch.setattr("recall.setup._drop_default_schema_family", lambda dsn: dropped.append(dsn))
    monkeypatch.setattr(
        "recall.schema.apply_migrations",
        lambda dsn, *, table, dim: applied.append((dsn, table, dim)) or (),
    )
    answers = iter([
        "n",  # security
        "", "", "",  # API keys
        "4",  # fastembed large, 1024 wide
        "1",  # reranker
        "1",  # sparse
        "n",  # scaffold
        "n",  # calibrate
    ])
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://example/recall",
        migration_dsn="postgresql://owner/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert "uses vector(384), requested dimension is 1024" in text
    assert "default schema is empty, so the wizard will rebuild it now" in text
    assert dropped == ["postgresql://owner/recall"]
    assert applied == [("postgresql://owner/recall", "chunks", 1024)]
    assert next(answers, None) is None
    assert "RECALL_EMBEDDER=fastembed:BAAI/bge-large-en-v1.5" in (
        tmp_path / ".env"
    ).read_text(encoding="utf-8")


def test_an_unreachable_database_is_not_treated_as_a_conflict() -> None:
    """The wizard has always run before the schema exists, and must keep doing so."""
    from recall.setup import _schema_dim_conflict

    assert _schema_dim_conflict("postgresql://nobody@127.0.0.1:1/nothing", 384) is None


def test_setup_requires_a_migration_dsn_to_fix_a_dimension_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: _roomy_probe(10 * 1024**3))
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    monkeypatch.setattr(
        "recall.setup._schema_prepare_state",
        lambda dsn, dim, table=None: ("needs_apply", "table needs schema migration(s)"),
    )
    applied: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        "recall.schema.apply_migrations",
        lambda dsn, *, table, dim: applied.append((dsn, table, dim)) or (),
    )
    answers = iter(["n", "", "", "", "4", "1", "1", "n", "n"])

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: None,
    )

    assert applied == [("postgresql://example/recall", "chunks", 1024)]


def test_the_width_check_asks_about_the_table_the_caller_uses(tmp_path, monkeypatch):
    """Checking a hard-coded `chunks` would refuse over a table a `--table` user does not use."""
    seen: list[str | None] = []

    def spy(dsn: str, dim: int, table: str | None = None) -> tuple[str, None]:
        seen.append(table)
        return "compatible", None

    monkeypatch.setattr("recall.setup.probe_hardware", lambda: _roomy_probe(10 * 1024**3))
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    monkeypatch.setattr("recall.setup._schema_prepare_state", spy)
    answers = iter(["n", "", "", "", "2", "1", "1", "n", "n"])

    run_setup_wizard(
        dsn="postgresql://example/recall",
        table="my_project_chunks",
        env_path=tmp_path / ".env",
        input_fn=lambda _prompt="": next(answers),
        print_fn=lambda *a, **k: None,
    )

    assert seen == ["my_project_chunks"]


def test_setup_refuses_to_rebuild_a_populated_mismatched_table(tmp_path, monkeypatch):
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: _roomy_probe(10 * 1024**3))
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    monkeypatch.setattr(
        "recall.setup._schema_prepare_state",
        lambda dsn, dim, table=None: ("conflict", "mismatch" if dim == 1024 else None),
    )
    monkeypatch.setattr(
        "recall.setup._table_row_counts",
        lambda dsn, tables: {name: (17 if name == "chunks" else 0) for name in tables},
    )
    answers = iter(["n", "", "", "", "4"])

    with pytest.raises(SystemExit, match="already contains data: chunks=17"):
        run_setup_wizard(
            dsn="postgresql://example/recall",
            migration_dsn="postgresql://owner/recall",
            env_path=tmp_path / ".env",
            input_fn=lambda _prompt="": next(answers),
            print_fn=lambda *a, **k: None,
        )


def test_setup_surfaces_schema_apply_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: _roomy_probe(10 * 1024**3))
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    monkeypatch.setattr(
        "recall.setup._schema_prepare_state",
        lambda dsn, dim, table=None: ("needs_apply", "table needs schema migration(s)"),
    )
    monkeypatch.setattr(
        "recall.schema.apply_migrations",
        lambda dsn, *, table, dim: (_ for _ in ()).throw(PermissionError("no create privilege")),
    )
    answers = iter(["n", "", "", "", "4"])

    with pytest.raises(SystemExit, match="Original error: PermissionError: no create privilege"):
        run_setup_wizard(
            dsn="postgresql://example/recall",
            env_path=tmp_path / ".env",
            input_fn=lambda _prompt="": next(answers),
            print_fn=lambda *a, **k: None,
        )


def test_reasoning_providers_hide_cloud_when_security_is_required(monkeypatch):
    """Reasoning sends the query AND the retrieved evidence to the provider, which exposes more
    than embedding does. Somebody who said their data must not leave the machine must not be
    walked into a cloud provider three prompts later."""
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=100 * 1024**3,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    choices = reasoning_provider_choices(probe, security_required=True)
    assert [c.value for c in choices] == [LOCAL_PROVIDER]


def test_reasoning_providers_offer_cloud_when_security_is_not_required(monkeypatch):
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=100 * 1024**3,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    choices = reasoning_provider_choices(probe, security_required=False)
    assert [c.value for c in choices] == [
        LOCAL_PROVIDER,
        OPENROUTER_BASE_URL,
        OPENAI_BASE_URL,
    ]
    assert all(c.available for c in choices)


def test_reasoning_providers_mark_cloud_unavailable_without_the_openai_package(monkeypatch):
    """Offered but marked, never hidden: hiding makes the product look like it lacks the feature
    and leaves no way to ask for it. This is the same rule the embedder menu follows."""
    monkeypatch.setattr("recall.setup._module_available", lambda name: name != "openai")
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=100 * 1024**3,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    choices = reasoning_provider_choices(probe, security_required=False)
    assert choices[0].available is True
    assert [c.available for c in choices[1:]] == [False, False]
    assert 'pip install "recall-rag[extract]"' in choices[1].unavailable_note


def test_the_first_reasoning_provider_is_always_runnable(monkeypatch):
    """`_choose` raises unless choices[0].available. A local endpoint needs no key and no
    internet, so it is the only entry that can lead the menu unconditionally."""
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    probe = HardwareProbe(
        cpu_count=1,
        gpu=None,
        cuda_available=False,
        free_bytes=0,
        internet=False,
        fastembed_available=False,
        sentence_transformers_available=False,
    )
    choices = reasoning_provider_choices(probe, security_required=False)
    assert choices[0].available is True
