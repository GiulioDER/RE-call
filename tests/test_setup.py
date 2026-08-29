from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

from recall.setup import (
    HardwareProbe,
    LOCAL_API_KEY,
    LOCAL_PROVIDER,
    MANUAL_MODEL,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    _prompt_twice,
    embedder_choices,
    probe_reasoning_model,
    reasoning_model_choices,
    reasoning_provider_choices,
    run_setup_wizard,
)
import recall.setup as recall_setup
from recall.seed import SeedPlan
from tests.conftest import requires_openai


@pytest.fixture(autouse=True)
def no_machine_dependent_prompts(monkeypatch):
    """Pin the two conditional prompts off for every wizard test in this module.

    Both appear only when the machine happens to be in a particular state: the Claude Code wiring
    prompt when a client is detected, the seeding prompt when the current directory has documents
    worth seeding. The answer scripts below are positional, so an unpinned conditional prompt
    shifts every later answer by one and the test hangs on an exhausted iterator. That failure
    reads as a flake, and it would appear or vanish depending on whether the person running the
    suite uses Claude Code, or on which directory they ran pytest from.

    Tests that exercise either prompt turn it back on explicitly.

    `plugin_skill_sources` is pinned empty for the same reason: the skill-copy prompt appears
    only when Claude Code is detected AND the repository's `plugin/` directory is on disk, and
    this suite always runs from a checkout where it is. Left unpinned, every test that turns
    detection back on would grow an extra prompt here and never under an installed wheel.
    """
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: False)
    monkeypatch.setattr("recall.setup.plugin_skill_sources", lambda: {})
    monkeypatch.setattr(
        "recall.setup.plan_seed",
        lambda root, **kw: SeedPlan(root=Path(root), files=(), total_bytes=0),
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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


def _template_block(starter: str) -> str:
    """The fenced memo template out of the starter index, as an author would copy it."""
    _, _, rest = starter.partition("```markdown\n")
    block, _, _ = rest.partition("```")
    return block


def test_starter_template_is_a_memo_recall_can_actually_read(tmp_path):
    """The taught format must round-trip through the reader that indexing fails fast on.

    This is the apparatus check, not a copy assertion. `recall/index.py` calls `validity_bounds`
    and raises, so a template teaching a shape the parser rejects would break `recall_index` on
    the very directory the wizard indexes moments later.
    """
    from datetime import date, datetime, timezone

    from recall.frontmatter import parse_frontmatter, validity_bounds
    from recall.setup import _memory_md_starter

    memo = _template_block(_memory_md_starter(date(2026, 8, 19)))
    memo = memo.replace("<short-kebab-case-slug>", "prefers-pnpm")
    memo = memo.replace("<one-line summary, used to judge relevance>", "package manager")
    memo = memo.replace("<the fact>", "This project installs with pnpm, never npm.")

    meta, body = parse_frontmatter(memo)

    assert meta["valid_from"] == "2026-08-19"
    assert "This project installs with pnpm" in body
    assert "valid_from" not in body  # the block was consumed, not left in the prose
    start, end = validity_bounds(meta)
    assert start == datetime(2026, 8, 19, tzinfo=timezone.utc)
    assert end is None  # valid_until is deliberately absent from the template


def test_starter_teaches_the_three_validity_keys_and_supersession(tmp_path):
    from datetime import date

    from recall.frontmatter import VALIDITY_KEYS
    from recall.setup import scaffold_memory_index

    memory_dir = tmp_path / "memory"
    scaffold_memory_index(memory_dir, today=date(2026, 8, 19))

    text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    for key in VALIDITY_KEYS:
        assert key in text, f"the starter must name {key}, the trust layer reads it"
    assert "valid_from: 2026-08-19" in text
    assert "Leave it out unless you know a real end date" in text
    assert "supersedes: <old-file>.md" in text


def test_starter_index_is_not_itself_frontmatter(tmp_path):
    """The fenced `---` lines inside the template must not pair into a block on MEMORY.md.

    If they did, the whole example would be stripped out of the indexed body and the file that
    teaches the format would stop containing it.
    """
    from datetime import date

    from recall.frontmatter import frontmatter_span, parse_frontmatter
    from recall.setup import _memory_md_starter

    starter = _memory_md_starter(date(2026, 8, 19))
    meta, body = parse_frontmatter(starter)

    assert frontmatter_span(starter) is None
    assert meta == {}
    assert body == starter
    assert "supersedes" in body


def test_scaffolded_memory_dir_lints_clean_including_a_real_supersession_edge(tmp_path):
    """`recall setup` must not write a corpus that `recall lint` then complains about.

    The first version of this scaffold did exactly that: teaching the `supersedes` key put the
    word in MEMORY.md's prose, and `closure-marker-unlinked` fired on the file the tool had just
    written. A linter that warns about its own tool's output teaches users to ignore the linter,
    so this asserts the whole scaffold-then-author path is clean, edge and all.
    """
    from datetime import date

    from recall.lint import lint_corpus
    from recall.setup import _memory_md_starter, scaffold_memory_index

    memory_dir = tmp_path / "memory"
    scaffold_memory_index(memory_dir, today=date(2026, 8, 19))
    template = _template_block(_memory_md_starter(date(2026, 8, 19)))

    def memo(slug: str, fact: str, extra: str = "") -> str:
        text = template.replace("<short-kebab-case-slug>", slug).replace("<the fact>", fact)
        text = text.replace("<one-line summary, used to judge relevance>", "package manager")
        return text.replace("valid_from: 2026-08-19", f"valid_from: 2026-08-19{extra}")

    (memory_dir / "pm-npm.md").write_text(memo("pm-npm", "Installs with npm."), encoding="utf-8")
    (memory_dir / "pm-pnpm.md").write_text(
        memo("pm-pnpm", "Installs with pnpm.", extra="\nsupersedes: pm-npm.md"), encoding="utf-8"
    )

    issues = lint_corpus(memory_dir)

    assert issues == [], [f"{i.file}: {i.code} {i.message}" for i in issues]


def test_prose_only_ignores_fenced_samples_but_not_real_prose():
    from recall.lint import CLOSURE_MARKERS, prose_only

    fenced = "A memo.\n\n```markdown\nsupersedes: old.md\n```\n\nNothing else.\n"
    assert CLOSURE_MARKERS.search(prose_only(fenced)) is None

    prose = "A memo.\n\nThis supersedes the old approach.\n"
    assert CLOSURE_MARKERS.search(prose_only(prose)) is not None

    # an unclosed fence runs to the end of the document, as CommonMark specifies
    unclosed = "A memo.\n\n```\nThis supersedes the old approach.\n"
    assert CLOSURE_MARKERS.search(prose_only(unclosed)) is None

    # a longer closing fence closes a shorter opener; an info string does not close anything
    tricky = "```\ncode\n````\n\nThis supersedes the old approach.\n"
    assert CLOSURE_MARKERS.search(prose_only(tricky)) is not None


def test_claude_md_block_teaches_closing_a_fact_rather_than_overwriting(tmp_path):
    from recall.setup import scaffold_claude_md

    path = tmp_path / "CLAUDE.md"
    scaffold_claude_md(path)

    text = path.read_text(encoding="utf-8")
    assert "valid_from" in text
    assert "supersedes" in text
    assert "do not edit or delete" in text


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
        def __init__(self, store, embedder, chunker=None, context_policy=None):
            calls["store"] = store
            calls["embedder"] = embedder
            calls["context_policy"] = context_policy

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
    assert calls["context_policy"].mode == "none"
    assert "Indexed 3 chunks from 1 files" in output.getvalue()


def test_index_memory_directory_honours_the_requested_tenant_and_table(monkeypatch, tmp_path):
    """It wrote to DEFAULT_TENANT unconditionally, so a `memory` tenant landed in `default`.

    The index succeeded, printed success, and put the rows where nothing would look for them. The
    existing test above cannot see this: its store stub is `lambda *a, **k: FakeStore()`, which
    discards the very keywords that decide where the rows go. Recording them is the whole test.
    """
    from recall.index import IndexStats
    from recall.setup import index_memory_directory

    monkeypatch.delenv("RECALL_ENV", raising=False)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("# Memory index\n", encoding="utf-8")

    opened: dict[str, object] = {}

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def check_schema(self):
            pass

    class FakeIndexer:
        def __init__(self, store, embedder, chunker=None):
            pass

        def index_path(self, path, glob=None):
            return IndexStats(files=1, chunks=3)

    def _store(*args, **kwargs):
        opened.update(kwargs)
        return FakeStore()

    monkeypatch.setattr("recall.store.PgVectorStore", _store)
    monkeypatch.setattr("recall.index.Indexer", FakeIndexer)

    index_memory_directory(
        dsn="postgresql://example/recall",
        embedder_name="hashing",
        memory_dir=memory_dir,
        tenant="memory",
        table="probe_chunks",
        print_fn=lambda *a, **k: None,
    )

    assert opened["tenant"] == "memory", "the caller's tenant must decide where the rows go"
    assert opened["table"] == "probe_chunks"

    # And the default is unchanged, so existing callers keep their behaviour.
    opened.clear()
    index_memory_directory(
        dsn="postgresql://example/recall",
        embedder_name="hashing",
        memory_dir=memory_dir,
        print_fn=lambda *a, **k: None,
    )
    from recall.store import DEFAULT_TABLE, DEFAULT_TENANT

    assert opened["tenant"] == DEFAULT_TENANT
    assert opened["table"] == DEFAULT_TABLE


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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
        "n",  # reasoning arm declined
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
    answers = iter(["n", "", "", "", "1", "1", "1", "n", "n", "n"])
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
    answers = iter(["n", "", "", "", "1", "2", "1", "n", "n", "n"])  # reranker 2 is not installed
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
    answers = iter(["n", "", "", "", "1", "1", "2", "n", "n", "n"])  # sparse 2 is not installed
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
        "n",  # reasoning arm declined
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
    answers = iter(["n", "", "", "", "4", "1", "1", "n", "n", "n"])

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
    answers = iter(["n", "", "", "", "2", "1", "1", "n", "n", "n"])

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
    and leaves no way to ask for it. This is the same rule `reranker_choices` and
    `sparse_choices` follow. `embedder_choices` differs: it omits cloud entries conditionally
    rather than offering them marked unavailable."""
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


def test_openrouter_reasoning_models_lead_with_gpt_4o_mini():
    """The first entry is what a reader gets by pressing Enter, so it must be the safe
    inexpensive default rather than the best or the cheapest."""
    choices = reasoning_model_choices(OPENROUTER_BASE_URL)
    assert choices[0].value == "openai/gpt-4o-mini"
    assert choices[0].available is True
    assert "deepseek/deepseek-chat" in [c.value for c in choices]
    assert "anthropic/claude-sonnet-4.5" in [c.value for c in choices]


def test_openai_reasoning_models_carry_no_vendor_prefix():
    """api.openai.com serves only OpenAI's own models and rejects OpenRouter's vendor/model
    form, so offering `deepseek/deepseek-chat` there would be a menu entry that cannot work."""
    choices = reasoning_model_choices(OPENAI_BASE_URL)
    ids = [c.value for c in choices if c.value != MANUAL_MODEL]
    assert ids == ["gpt-4o-mini", "gpt-4o"]
    assert all("/" not in i for i in ids)


def test_every_reasoning_model_menu_offers_manual_entry():
    """The catalogue is a static list in a released artifact and will go stale. Manual entry is
    what stops that being fatal, so it is not optional on any provider."""
    for base_url in (OPENROUTER_BASE_URL, OPENAI_BASE_URL):
        choices = reasoning_model_choices(base_url)
        assert choices[-1].value == MANUAL_MODEL
        assert choices[-1].available is True


def test_reasoning_model_descriptions_quote_no_prices():
    """A price baked into a shipped menu is a measurement nothing re-checks, and it goes stale on
    somebody else's release schedule."""
    for base_url in (OPENROUTER_BASE_URL, OPENAI_BASE_URL):
        for choice in reasoning_model_choices(base_url):
            assert "$" not in choice.description


class _FakeCompletions:
    def __init__(self, error: Exception | None):
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return object()


class _FakeOpenAI:
    last: "_FakeOpenAI | None" = None

    def __init__(self, error: Exception | None = None, **kwargs):
        self.kwargs = kwargs
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(error)})()
        _FakeOpenAI.last = self


def _install_fake_openai(monkeypatch, error: Exception | None = None):
    import openai

    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: _FakeOpenAI(error, **kw))


@requires_openai
def test_the_probe_returns_none_when_the_call_succeeds(monkeypatch):
    _install_fake_openai(monkeypatch)
    result = probe_reasoning_model(
        base_url="https://openrouter.ai/api/v1", api_key="k", model="deepseek/deepseek-chat"
    )
    assert result is None


@requires_openai
def test_the_probe_passes_the_endpoint_and_disables_client_retries(monkeypatch):
    """max_retries=0 matches the extraction engine: retries belong to the caller, and a wizard
    probe that silently retries three times reads as a hang."""
    _install_fake_openai(monkeypatch)
    probe_reasoning_model(base_url="http://localhost:11434/v1", api_key="local", model="qwen")
    assert _FakeOpenAI.last.kwargs["base_url"] == "http://localhost:11434/v1"
    assert _FakeOpenAI.last.kwargs["api_key"] == "local"
    assert _FakeOpenAI.last.kwargs["max_retries"] == 0


@requires_openai
def test_the_probe_reports_the_failure_instead_of_raising(monkeypatch):
    """The probe runs against three providers and an arbitrary user supplied base URL, so the
    set of reachable exception types is not knowable here. Any escape turns an optional step
    into a failed install."""
    _install_fake_openai(monkeypatch, error=RuntimeError("model not found"))
    result = probe_reasoning_model(
        base_url="https://openrouter.ai/api/v1", api_key="k", model="nope/nope"
    )
    assert result is not None
    assert "model not found" in result


def test_the_probe_declines_without_the_openai_package(monkeypatch):
    monkeypatch.setattr("recall.setup._module_available", lambda name: False)
    result = probe_reasoning_model(base_url="x", api_key="k", model="m")
    assert result is not None
    assert "openai" in result


@requires_openai
def test_the_probe_returns_promptly_when_the_call_never_finishes(monkeypatch):
    """httpx's read timeout only bounds the gap between chunks of a streamed response, not the
    whole call, so a client that trickles bytes could keep resetting it forever. The probe's own
    thread join is what actually bounds total duration, and it must not hang waiting for it."""
    _install_fake_openai(monkeypatch)
    never_set = threading.Event()
    monkeypatch.setattr(
        _FakeCompletions, "create", lambda self, **kwargs: never_set.wait() or object()
    )
    try:
        result = probe_reasoning_model(
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
            model="deepseek/deepseek-chat",
            timeout=0.2,
        )
    finally:
        never_set.set()
    assert result == "no response within 0s"


def _run_wizard(tmp_path, monkeypatch, answers, probe=None):
    probe = probe or HardwareProbe(
        cpu_count=8,
        gpu=None,
        cuda_available=False,
        free_bytes=100 * 1024**3,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=False,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    it = iter(answers)
    output = io.StringIO()
    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=lambda _p="": next(it),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )
    return (tmp_path / ".env").read_text(encoding="utf-8"), output.getvalue()


def test_prompt_twice_accepts_the_second_answer_after_a_blank():
    answers = iter(["", "qwen2.5"])
    assert _prompt_twice(lambda _p="": next(answers), lambda *a, **k: None, "Model id: ") == "qwen2.5"


def test_prompt_twice_gives_up_after_two_blanks():
    """Giving up rather than looping is what stops a piped stdin spinning forever."""
    answers = iter(["", ""])
    assert _prompt_twice(lambda _p="": next(answers), lambda *a, **k: None, "Model id: ") == ""


def test_declining_the_reasoning_arm_writes_only_the_off_flag(tmp_path, monkeypatch):
    """`off` and `never configured` must stay distinguishable in .env, which is why the flag is
    written rather than implied by the absence of a model."""
    env, _ = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",   # security required
            "2",   # embedder: fastembed
            "1",   # reranker: none
            "1",   # sparse: fts
            "n",   # reasoning arm declined
            "n",   # scaffold declined
            "n",   # calibrate declined
        ],
    )
    assert "RECALL_REASONING_EXPANSION=0" in env
    assert "RECALL_REASONING_EXPANSION_MODEL" not in env


def test_seeding_runs_before_the_hooks_are_installed(tmp_path, monkeypatch):
    """Order is the feature. A first session searching an empty corpus teaches the wrong lesson."""
    order = []
    plan = SeedPlan(root=tmp_path, files=(tmp_path / "CLAUDE.md",), total_bytes=12)
    monkeypatch.setattr("recall.setup.plan_seed", lambda root, **kw: plan)
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr(
        "recall.setup.scaffold_claude_md", lambda *a, **k: order.append("scaffold")
    )
    monkeypatch.setattr("recall.setup.scaffold_memory_index", lambda *a, **k: False)
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: None)
    monkeypatch.setattr("recall.setup.seed_corpus", lambda **kw: order.append("seed") or 3)
    monkeypatch.setattr(
        "recall.setup.register_mcp_server", lambda **kw: order.append("register")
    )
    monkeypatch.setattr("recall.setup.install_hooks", lambda **kw: order.append("hooks"))

    _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",   # security required
            "2",   # embedder: fastembed
            "1",   # reranker: none
            "1",   # sparse: fts
            "n",   # reasoning arm declined
            "y",   # scaffold
            "y",   # seed
            "y",   # wire up Claude Code
            "y",   # search memory on every write (the write-time hook)
            "n",   # calibrate declined
        ],
    )

    assert order == ["scaffold", "seed", "register", "hooks"]


def test_the_seed_prompt_names_what_it_would_ingest(tmp_path, monkeypatch):
    """Consent to an unspecified amount of your own project is not consent."""
    prompts = []
    plan = SeedPlan(root=tmp_path, files=(tmp_path / "a.md", tmp_path / "b.md"), total_bytes=4096)
    monkeypatch.setattr("recall.setup.plan_seed", lambda root, **kw: plan)
    monkeypatch.setattr("recall.setup.seed_corpus", lambda **kw: 0)

    real_prompt = recall_setup._prompt

    def spy(input_fn, print_fn, text, *a, **k):
        prompts.append(text)
        return real_prompt(input_fn, print_fn, text, *a, **k)

    monkeypatch.setattr("recall.setup._prompt", spy)
    _run_wizard(tmp_path, monkeypatch, ["y", "2", "1", "1", "n", "n", "n", "n"])

    seed_prompt = next(p for p in prompts if "Seed the corpus" in p)
    assert "2 files" in seed_prompt and "4 KB" in seed_prompt


def test_no_seed_prompt_when_there_is_nothing_to_seed(tmp_path, monkeypatch):
    """The answer script below has no answer for it, so an unwanted prompt exhausts the iterator."""
    monkeypatch.setattr("recall.setup.seed_corpus", lambda **kw: 0)
    # The autouse fixture already pins an empty plan; this asserts the wizard honours it.
    env, _ = _run_wizard(tmp_path, monkeypatch, ["y", "2", "1", "1", "n", "n", "n"])
    assert "RECALL_EMBEDDER=fastembed" in env


def test_declining_the_seed_prompt_indexes_nothing(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise AssertionError("a declined seed must not read or index the project")

    plan = SeedPlan(root=tmp_path, files=(tmp_path / "CLAUDE.md",), total_bytes=12)
    monkeypatch.setattr("recall.setup.plan_seed", lambda root, **kw: plan)
    monkeypatch.setattr("recall.setup.seed_corpus", fail)

    _run_wizard(tmp_path, monkeypatch, ["y", "2", "1", "1", "n", "n", "n", "n"])


def test_accepting_the_wiring_prompt_registers_the_server_and_installs_the_hooks(
    tmp_path, monkeypatch
):
    """The step that decides whether Claude uses recall at all in the session after this one."""
    calls = {}
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr(
        "recall.setup.register_mcp_server", lambda **kw: calls.setdefault("register", kw)
    )
    monkeypatch.setattr("recall.setup.install_hooks", lambda **kw: calls.setdefault("hooks", kw))

    env, output = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",   # security required
            "2",   # embedder: fastembed
            "1",   # reranker: none
            "1",   # sparse: fts
            "n",   # reasoning arm declined
            "n",   # scaffold declined
            "y",   # wire up Claude Code
            "y",   # search memory on every write (the write-time hook)
            "n",   # calibrate declined
        ],
    )

    assert calls["register"]["dsn"] == "postgresql://example/recall"
    # ⚠️ The embedder, not just the DSN. The first fix for this finding widened `server_env` and
    # never extended `register_mcp_server`, so nothing could pass the value and the suite stayed
    # green over a defect that was still open. Deleting `embedder=embedder.value` from
    # `recall/setup.py` restores exactly that inert state, and must turn this red.
    assert calls["register"].get("embedder"), (
        "recall setup registered a server without the embedder the interview chose; the server "
        "does not read .env, so it will silently fall back to fastembed"
    )
    assert calls["hooks"]["embedder"] == "fastembed"
    # The tools land in the NEXT session, and a user who does not know that reads a working
    # install as a broken one when the current session shows no recall tools.
    assert "NEXT session" in output
    assert "RECALL_EMBEDDER=fastembed" in env


def test_declining_the_wiring_prompt_touches_no_client_configuration(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise AssertionError("declined wiring must not write to the client's configuration")

    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr("recall.setup.register_mcp_server", fail)
    monkeypatch.setattr("recall.setup.install_hooks", fail)

    _run_wizard(
        tmp_path,
        monkeypatch,
        ["y", "2", "1", "1", "n", "n", "n", "n"],  # ...scaffold n, wiring n, calibrate n
    )


def test_a_failed_wiring_step_does_not_lose_the_completed_interview(tmp_path, monkeypatch):
    """`.env` is written before this runs, so a client whose config has moved costs a line."""
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr(
        "recall.setup.register_mcp_server",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("claude mcp add failed: nope")),
    )

    env, output = _run_wizard(
        tmp_path,
        monkeypatch,
        # security y, embedder 2, reranker 1, sparse 1, reasoning n, scaffold n,
        # wiring y, write-time hook y, calibrate n
        ["y", "2", "1", "1", "n", "n", "y", "y", "n"],
    )

    assert "RECALL_EMBEDDER=fastembed" in env
    assert "Could not wire up Claude Code" in output
    assert "USING_WITH_CLAUDE.md" in output


def test_the_plugin_install_lines_are_the_wizards_last_words(tmp_path, monkeypatch):
    """Print-only guidance, gated on a detected client and printed after everything else, so the
    two slash commands are the note left on screen for the user to type into Claude Code."""
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)

    _, output = _run_wizard(
        tmp_path,
        monkeypatch,
        ["y", "2", "1", "1", "n", "n", "n", "n"],  # ...wiring n, calibrate n
    )

    assert "/plugin marketplace add GiulioDER/RE-call" in output
    assert "/plugin install recall@re-call" in output
    # The fixture pins discovery empty, which is the installed-wheel case for a build that
    # does not carry the skills: no copy prompt appeared (the script above has no answer
    # for one), and the guidance says the plugin is how they arrive.
    assert "ship inside that plugin" in output
    assert output.rstrip().endswith("gets them.")


def test_no_plugin_guidance_when_claude_code_is_absent(tmp_path, monkeypatch):
    """Telling someone to type slash commands into a client they do not have is noise."""
    _, output = _run_wizard(tmp_path, monkeypatch, ["y", "2", "1", "1", "n", "n", "n"])
    assert "/plugin" not in output


def test_accepting_the_skill_copy_installs_it_under_the_config_home(tmp_path, monkeypatch):
    source = tmp_path / "SKILL.md"
    source.write_text("---\nname: check-memory-before-acting\n---\n\nSearch first.\n", encoding="utf-8")
    config_home = tmp_path / "claude-home"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_home))
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr(
        "recall.setup.plugin_skill_sources",
        lambda: {"check-memory-before-acting": source},
    )

    _, output = _run_wizard(
        tmp_path,
        monkeypatch,
        ["y", "2", "1", "1", "n", "n", "n", "y", "n"],  # ...wiring n, copy the skill y, calibrate n
    )

    dest = config_home / "skills" / "check-memory-before-acting" / "SKILL.md"
    assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert "Installed the check-memory-before-acting skill" in output


def test_declining_the_skill_copy_writes_nothing(tmp_path, monkeypatch):
    """The copy lands in a directory every project's sessions load, so silence means no."""
    source = tmp_path / "SKILL.md"
    source.write_text("skill body\n", encoding="utf-8")
    config_home = tmp_path / "claude-home"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_home))
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr(
        "recall.setup.plugin_skill_sources",
        lambda: {"check-memory-before-acting": source},
    )

    _, output = _run_wizard(
        tmp_path,
        monkeypatch,
        ["y", "2", "1", "1", "n", "n", "n", "n", "n"],  # ...wiring n, copy the skill n, calibrate n
    )

    assert not (config_home / "skills").exists()
    # The install lines still print: declining the copy is not declining the guidance.
    assert "/plugin install recall@re-call" in output


def test_a_failed_skill_copy_does_not_lose_the_completed_interview(tmp_path, monkeypatch):
    """Same contract as the wiring step: `.env` is written before this runs, so a filesystem
    refusal costs a printed line carrying the by-hand alternative."""
    source = tmp_path / "vanished" / "SKILL.md"  # never written, so the copy raises
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: True)
    monkeypatch.setattr(
        "recall.setup.plugin_skill_sources",
        lambda: {"check-memory-before-acting": source},
    )

    env, output = _run_wizard(
        tmp_path,
        monkeypatch,
        ["y", "2", "1", "1", "n", "n", "n", "y", "n"],
    )

    assert "RECALL_EMBEDDER=fastembed" in env
    assert "Could not install the check-memory-before-acting skill" in output
    assert "Not installed: check-memory-before-acting" in output
    assert "by hand" in output


def test_every_shipped_skill_is_where_the_wizard_looks_for_it():
    """Ties discovery to the real tree: if `plugin/skills/` moves, this is the test that says the
    wizard's copy offer silently became the installed-wheel path everywhere.

    Asserts the SET, not a count. A count passes while a newly added skill is never installed,
    which is exactly the failure that made this plural: the resolver named one skill by hand.
    """
    from recall.claude_code import plugin_skill_sources

    sources = plugin_skill_sources()
    assert set(sources) == {"check-memory-before-acting", "keep-memory-current"}
    for name, source in sources.items():
        assert source.name == "SKILL.md"
        assert source.parent.name == name, "a skill is loaded by its DIRECTORY name"
        assert source.is_file()


def test_install_user_skill_leaves_an_identical_copy_unchanged(tmp_path, monkeypatch):
    from recall.claude_code import install_user_skill

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "home"))
    source = tmp_path / "SKILL.md"
    source.write_text("same content\n", encoding="utf-8")
    lines: list[str] = []

    install_user_skill(
        source,
        # Explicit: the destination name now defaults to the source's DIRECTORY, and this
        # source sits in tmp_path rather than in a folder named after the skill.
        name="check-memory-before-acting",
        print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))),
    )
    dest = tmp_path / "home" / "skills" / "check-memory-before-acting" / "SKILL.md"
    before = dest.stat().st_mtime_ns

    install_user_skill(
        source,
        # Explicit: the destination name now defaults to the source's DIRECTORY, and this
        # source sits in tmp_path rather than in a folder named after the skill.
        name="check-memory-before-acting",
        print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))),
    )

    assert dest.stat().st_mtime_ns == before
    assert any("Installed" in line for line in lines)
    assert any("left unchanged" in line for line in lines)


def test_install_user_skill_replaces_a_stale_copy_and_says_so(tmp_path, monkeypatch):
    from recall.claude_code import install_user_skill

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "home"))
    dest = tmp_path / "home" / "skills" / "check-memory-before-acting" / "SKILL.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("old content\n", encoding="utf-8")
    source = tmp_path / "SKILL.md"
    source.write_text("new content\n", encoding="utf-8")
    lines: list[str] = []

    install_user_skill(
        source,
        # Explicit: the destination name now defaults to the source's DIRECTORY, and this
        # source sits in tmp_path rather than in a folder named after the skill.
        name="check-memory-before-acting",
        print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))),
    )

    assert dest.read_text(encoding="utf-8") == "new content\n"
    assert any("Replaced" in line for line in lines)


def test_choosing_openrouter_and_deepseek_writes_all_four_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    env, _ = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "n",              # security not required
            "",               # VOYAGE_API_KEY skipped
            "",               # OPENAI_API_KEY skipped
            "router-key",     # OPENROUTER_API_KEY
            "2",              # embedder: fastembed
            "1",              # reranker: none
            "1",              # sparse: fts
            "y",              # reasoning arm enabled
            "2",              # provider: openrouter
            "2",              # model: deepseek chat, second now that gpt-4o mini leads
            "n",              # scaffold declined
            "n",              # calibrate declined
        ],
    )
    assert "RECALL_REASONING_EXPANSION=1" in env
    assert "RECALL_REASONING_EXPANSION_MODEL=deepseek/deepseek-chat" in env
    assert "RECALL_REASONING_EXPANSION_BASE_URL=https://openrouter.ai/api/v1" in env
    assert "RECALL_REASONING_EXPANSION_API_KEY=router-key" in env


def test_a_key_captured_during_the_interview_is_also_written_under_its_provider_name(
    tmp_path, monkeypatch
):
    """Step 1b is not the only place a cloud key can be given: the interview itself asks for one
    when it was left blank there, and `_reasoning_interview` folds that answer back into
    `cloud_keys` so it lands in `.env` under its provider name too, not only under
    `RECALL_REASONING_EXPANSION_API_KEY`. The other cloud test supplies the key at step 1b, so it never
    exercises this capture path."""
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    env, _ = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "n",              # security not required
            "",               # VOYAGE_API_KEY skipped
            "",               # OPENAI_API_KEY skipped
            "",               # OPENROUTER_API_KEY skipped at step 1b
            "2",              # embedder: fastembed
            "1",              # reranker: none
            "1",              # sparse: fts
            "y",              # reasoning arm enabled
            "2",              # provider: openrouter
            "captured-key",   # OPENROUTER_API_KEY, typed during the interview
            "2",              # model: deepseek chat
            "n",              # scaffold declined
            "n",              # calibrate declined
        ],
    )
    assert "OPENROUTER_API_KEY=captured-key" in env
    assert "RECALL_REASONING_EXPANSION_API_KEY=captured-key" in env


def test_a_local_endpoint_takes_the_default_base_url(tmp_path, monkeypatch):
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    env, _ = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",        # security required, so only the local provider is offered
            "2",        # embedder: fastembed
            "1",        # reranker: none
            "1",        # sparse: fts
            "y",        # reasoning arm enabled
            # provider: local endpoint is the sole choice and is announced, not asked
            "",         # base URL: take the default
            "qwen2.5",  # model id
            "n",        # scaffold declined
            "n",        # calibrate declined
        ],
    )
    assert "RECALL_REASONING_EXPANSION_BASE_URL=http://localhost:11434/v1" in env
    assert "RECALL_REASONING_EXPANSION_MODEL=qwen2.5" in env
    assert f"RECALL_REASONING_EXPANSION_API_KEY={LOCAL_API_KEY}" in env


def test_a_malformed_base_url_does_not_end_the_interview(tmp_path, monkeypatch):
    """One mistyped bracket used to kill the whole wizard.

    `urlsplit("http://[::1:11434/v1")` raises `ValueError: Invalid IPv6 URL` on an unclosed
    bracket, and the locality check that decides whether to warn parsed the typed value with no
    guard. So a typo in an optional step raised out of `run_setup_wizard` before `.env` was
    written, taking every answer already given with it. The wizard is the thing you run to repair
    a broken configuration; it must not be the thing that dies on a typo.

    An unparseable URL is not obviously local, so the right outcome is the warning, not silence.
    """
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    env, output = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",                      # security required
            "2",                      # embedder: fastembed
            "1",                      # reranker: none
            "1",                      # sparse: fts
            "y",                      # reasoning arm enabled
            "http://[::1:11434/v1",   # base URL with an unclosed IPv6 bracket
            "qwen2.5",                # model id
            "n",                      # scaffold declined
            "n",                      # calibrate declined
        ],
    )
    assert "RECALL_REASONING_EXPANSION_BASE_URL=http://[::1:11434/v1" in env
    assert "RECALL_REASONING_EXPANSION_MODEL=qwen2.5" in env
    assert "retrieved evidence" in output  # it warned rather than staying silent


def test_a_failing_probe_still_writes_the_configuration(tmp_path, monkeypatch):
    """A transient fault must never block an install over a choice that is probably correct."""
    monkeypatch.setattr(
        "recall.setup.probe_reasoning_model", lambda **kw: "RuntimeError: model not found"
    )
    env, output = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",       # security required
            "2",       # embedder: fastembed
            "1",       # reranker: none
            "1",       # sparse: fts
            "y",       # reasoning arm enabled
            # provider: local endpoint is the sole choice and is announced, not asked
            "",        # base URL default
            "qwen2.5", # model id
            "n",       # scaffold declined
            "n",       # calibrate declined
        ],
    )
    assert "model not found" in output
    assert "RECALL_REASONING_EXPANSION=1" in env
    assert "RECALL_REASONING_EXPANSION_MODEL=qwen2.5" in env


def test_a_blank_model_id_twice_turns_the_arm_off(tmp_path, monkeypatch):
    env, output = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",   # security required
            "2",   # embedder: fastembed
            "1",   # reranker: none
            "1",   # sparse: fts
            "y",   # reasoning arm enabled
            # provider: local endpoint is the sole choice and is announced, not asked
            "",    # base URL default
            "",    # model id blank
            "",    # model id blank again
            "n",   # scaffold declined
            "n",   # calibrate declined
        ],
    )
    assert "RECALL_REASONING_EXPANSION=0" in env
    assert "RECALL_REASONING_EXPANSION_MODEL" not in env
    assert "no model id" in output


def test_a_blank_cloud_api_key_twice_turns_the_arm_off(tmp_path, monkeypatch):
    """A blank key must not be written as `RECALL_REASONING_EXPANSION=1`: that would enable an arm that
    cannot authenticate. The model id path already applies this rule through `_prompt_twice`, and
    the key path must follow it too."""
    env, output = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "n",   # security not required
            "",    # VOYAGE_API_KEY skipped
            "",    # OPENAI_API_KEY skipped
            "",    # OPENROUTER_API_KEY skipped
            "2",   # embedder: fastembed
            "1",   # reranker: none
            "1",   # sparse: fts
            "y",   # reasoning arm enabled
            "2",   # provider: openrouter
            "",    # OPENROUTER_API_KEY blank
            "",    # OPENROUTER_API_KEY blank again
            "n",   # scaffold declined
            "n",   # calibrate declined
        ],
    )
    assert "RECALL_REASONING_EXPANSION=0" in env
    assert "RECALL_REASONING_EXPANSION_API_KEY" not in env
    assert "no API key" in output


REASONING_ENV_KEYS = frozenset(
    {
        "RECALL_REASONING_EXPANSION",
        "RECALL_REASONING_EXPANSION_MODEL",
        "RECALL_REASONING_EXPANSION_BASE_URL",
        "RECALL_REASONING_EXPANSION_API_KEY",
    }
)


def test_the_wizard_writes_exactly_the_agreed_reasoning_variables(tmp_path, monkeypatch):
    """The reasoning arm is being built separately against these four names. Renaming one here
    without renaming it there produces a wizard that configures nothing, and no other test in
    this repository would notice.
    """
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    env, _ = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y", "2", "1", "1",
            # "y" enables the arm; the provider prompt is skipped because local endpoint is the
            # sole choice under security_required and is announced, not asked.
            "y", "", "qwen2.5",
            "n", "n",
        ],
    )
    written = {
        line.split("=", 1)[0]
        for line in env.splitlines()
        if line.startswith("RECALL_REASONING")
    }
    assert written == REASONING_ENV_KEYS


def test_the_interview_enables_the_flag_the_runtime_actually_reads(tmp_path, monkeypatch):
    """The runtime gate is `RECALL_REASONING_EXPANSION` plus `RECALL_REASONING_EXPANSION_MODEL`
    (`recall.reasoning_expansion.resolve_expansion_provider`). The interview used to write a bare
    `RECALL_REASONING` and `RECALL_REASONING_MODEL`, which nothing reads, so the wizard
    half-configured expansion (base URL and API key land under their real names) and fully enabled
    nothing."""
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    env, _ = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",        # security required, so only the local provider is offered
            "2",        # embedder: fastembed
            "1",        # reranker: none
            "1",        # sparse: fts
            "y",        # reasoning arm enabled
            "",         # base URL: take the default
            "qwen2.5",  # model id
            "n",        # scaffold declined
            "n",        # calibrate declined
        ],
    )
    written = {
        line.split("=", 1)[0]
        for line in env.splitlines()
        if line.startswith("RECALL_REASONING")
    }
    assert "RECALL_REASONING_EXPANSION" in written
    assert "RECALL_REASONING_EXPANSION_MODEL" in written
    assert "RECALL_REASONING" not in written
    assert "RECALL_REASONING_MODEL" not in written


def test_unreachable_database_reports_a_message_instead_of_staying_silent(tmp_path, monkeypatch):
    """`_prepare_schema_for_embedder` used to return silently on an unreachable database, so the
    wizard finished as though the schema were fine. The first sign of trouble was `index` or
    `search` failing much later with 'vector type not found in the database'. Point setup at a
    closed port and the wizard must say so, in words naming the actual command to run, while
    still completing and writing `.env`."""
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: _roomy_probe(100 * 1024**3))
    answers = iter([
        "y",  # security required, so no cloud API key prompts
        "2",  # embedder: fastembed
        "1",  # reranker: none
        "1",  # sparse: fts
        "n",  # reasoning arm declined
        "n",  # scaffold declined
        "n",  # calibrate declined
    ])
    output = io.StringIO()

    run_setup_wizard(
        dsn="postgresql://recall:recall@127.0.0.1:1/db",
        env_path=tmp_path / ".env",
        input_fn=lambda _p="": next(answers),
        print_fn=lambda *a, **k: print(*a, **k, file=output),
    )

    text = output.getvalue()
    assert "database could not be reached" in text
    assert "was not prepared" in text
    assert "recall schema apply" in text
    assert next(answers, None) is None
    # The whole point of the wizard is to be runnable when the configuration is broken: it must
    # neither raise nor exit, and it must still write .env.
    assert (tmp_path / ".env").exists()


def test_a_remote_local_endpoint_base_url_warns_about_the_tradeoff(tmp_path, monkeypatch):
    """`security_required` withholds cloud providers from the menu, but the base URL prompt still
    accepts any free text, and a model server on a private LAN or an internal DNS name is a
    legitimate local setup this cannot recognize. So a remote looking URL is not refused, but the
    reader must be told what answering yes to the security question actually bought them."""
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    _, output = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",  # security required, so only the local provider is offered
            "2",  # embedder: fastembed
            "1",  # reranker: none
            "1",  # sparse: fts
            "y",  # reasoning arm enabled
            "https://reasoning.example.com/v1",  # base URL, not obviously local
            "qwen2.5",  # model id
            "n",  # scaffold declined
            "n",  # calibrate declined
        ],
    )
    assert "withheld the cloud providers" in output
    assert "reasoning.example.com" in output
    assert "query and the retrieved evidence" in output


def test_the_default_local_base_url_prints_no_warning(tmp_path, monkeypatch):
    """The warning must not fire on the ordinary path, where the reader took the offered default."""
    monkeypatch.setattr("recall.setup.probe_reasoning_model", lambda **kw: None)
    _, output = _run_wizard(
        tmp_path,
        monkeypatch,
        [
            "y",  # security required, so only the local provider is offered
            "2",  # embedder: fastembed
            "1",  # reranker: none
            "1",  # sparse: fts
            "y",  # reasoning arm enabled
            "",  # base URL, take the default, obviously local
            "qwen2.5",  # model id
            "n",  # scaffold declined
            "n",  # calibrate declined
        ],
    )
    assert "withheld the cloud providers" not in output


def test_quote_env_escapes_newlines_so_a_value_cannot_span_lines(tmp_path, monkeypatch):
    """Without escaping, a raw newline inside a value broke out of its quoted line, and
    `recall/_env.py`'s line based parser read the continuation as its own unrelated `KEY=VALUE`
    pair. Proved here by constructing a value that tries to smuggle in RECALL_SERVING_DSN
    alongside a legitimate key, the same shape as the finding this fixes."""
    import os

    from recall._env import load_dotenv
    from recall.setup import SETUP_BEGIN, SETUP_END, _update_env_block

    payload = "attacker-model\nRECALL_SERVING_DSN=postgresql://attacker@evil/db"
    env_path = tmp_path / ".env"
    _update_env_block(env_path, {"RECALL_REASONING_MODEL": payload})

    lines = env_path.read_text(encoding="utf-8").splitlines()
    data_lines = [line for line in lines if line not in (SETUP_BEGIN, SETUP_END)]
    assert len(data_lines) == 1, data_lines

    monkeypatch.delenv("RECALL_SERVING_DSN", raising=False)
    monkeypatch.delenv("RECALL_REASONING_MODEL", raising=False)
    load_dotenv(env_path)
    try:
        assert "RECALL_SERVING_DSN" not in os.environ
        assert os.environ["RECALL_REASONING_MODEL"] == payload
    finally:
        os.environ.pop("RECALL_REASONING_MODEL", None)
        os.environ.pop("RECALL_SERVING_DSN", None)
