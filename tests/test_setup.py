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
        "1",
        "1",
        "n",
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
        "1",
        "1",
        "n",
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
        "1",
        "1",
        "n",
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
        "1",
        "1",
        "y",
        "1",
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
