"""Tests for install-time corpus seeding.

The properties that matter are what gets in, what stays out, and that the plan the user is shown is
the plan that runs. The exclusions carry more weight than the inclusions here: seeding reads a
user's project without further prompting, so a directory that should never be walked is a bug with
a privacy shape rather than a correctness one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from recall.seed import SeedPlan, plan_seed, seed_corpus


def _project(root: Path) -> Path:
    """A project with one of everything the seeder is supposed to care about."""
    (root / "CLAUDE.md").write_text("# project rules\n", encoding="utf-8")
    (root / "README.md").write_text("# readme\n", encoding="utf-8")
    (root / "memory").mkdir()
    (root / "memory" / "MEMORY.md").write_text("# index\n", encoding="utf-8")
    (root / "memory" / "a-fact.md").write_text("a fact\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "GUIDE.md").write_text("# guide\n", encoding="utf-8")
    (root / "docs" / "nested").mkdir()
    (root / "docs" / "nested" / "deep.md").write_text("# deep\n", encoding="utf-8")
    return root


def _names(plan: SeedPlan) -> set[str]:
    return {p.name for p in plan.files}


def test_the_durable_documents_are_seeded(tmp_path: Path) -> None:
    plan = plan_seed(_project(tmp_path))
    assert _names(plan) == {"CLAUDE.md", "README.md", "MEMORY.md", "a-fact.md", "GUIDE.md",
                            "deep.md"}
    assert plan.total_bytes > 0
    assert not plan.is_empty


def test_memory_comes_before_docs_so_the_budget_keeps_it(tmp_path: Path) -> None:
    """Priority order decides what survives a budget, so it is asserted rather than assumed."""
    plan = plan_seed(_project(tmp_path))
    order = [p.name for p in plan.files]
    assert order.index("MEMORY.md") < order.index("GUIDE.md")


def test_a_file_is_never_offered_twice(tmp_path: Path) -> None:
    """`CLAUDE.md` and `README.md` are matched by name AND by the top-level `*.md` sweep."""
    plan = plan_seed(_project(tmp_path))
    assert len(plan.files) == len(set(plan.files))


def test_transcripts_and_other_checkouts_are_never_walked(tmp_path: Path) -> None:
    """The `.claude` exclusion is load-bearing twice over.

    It holds this user's session transcripts, and it holds `worktrees/`, so descending would seed
    every other checkout of the repository through this one.
    """
    _project(tmp_path)
    claude = tmp_path / ".claude"
    (claude / "projects" / "slug").mkdir(parents=True)
    (claude / "projects" / "slug" / "session.md").write_text("secret\n", encoding="utf-8")
    (claude / "worktrees" / "other" / "docs").mkdir(parents=True)
    (claude / "worktrees" / "other" / "docs" / "GUIDE.md").write_text("other\n", encoding="utf-8")

    plan = plan_seed(tmp_path)

    assert not any(".claude" in p.parts for p in plan.files)


def test_noise_directories_are_skipped_even_under_docs(tmp_path: Path) -> None:
    _project(tmp_path)
    vendored = tmp_path / "docs" / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "README.md").write_text("vendored\n", encoding="utf-8")

    assert not any("node_modules" in p.parts for p in plan_seed(tmp_path).files)


def test_non_prose_files_are_left_alone(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "docs" / "diagram.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "docs" / "script.py").write_text("print(1)\n", encoding="utf-8")

    assert not any(p.suffix in {".png", ".py"} for p in plan_seed(tmp_path).files)


def test_an_empty_file_is_not_seeded(tmp_path: Path) -> None:
    """Zero bytes cannot produce a chunk, so it is a file the user would see counted for nothing."""
    (tmp_path / "CLAUDE.md").write_text("", encoding="utf-8")
    assert plan_seed(tmp_path).is_empty


def test_a_project_with_nothing_to_seed_plans_nothing(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)\n", encoding="utf-8")
    plan = plan_seed(tmp_path)
    assert plan.is_empty
    assert plan.files == ()


def test_the_budget_drops_files_and_reports_them_rather_than_going_quiet(tmp_path: Path) -> None:
    """A cap that says nothing reads as 'everything was covered' when it was not."""
    _project(tmp_path)
    plan = plan_seed(tmp_path, max_bytes=20)

    assert plan.dropped, "the budget must record what it left out"
    assert plan.total_bytes <= 20
    assert "left out" in plan.describe()
    # And what survived is the highest priority, not an arbitrary prefix of the walk.
    assert plan.files[0].name == "CLAUDE.md"


def test_the_file_cap_is_honoured(tmp_path: Path) -> None:
    plan = plan_seed(_project(tmp_path), max_files=2)
    assert len(plan.files) == 2
    assert len(plan.dropped) == 4


def test_the_plan_is_deterministic(tmp_path: Path) -> None:
    """A user who is shown a plan and accepts it must get the thing they were shown."""
    _project(tmp_path)
    assert plan_seed(tmp_path).files == plan_seed(tmp_path).files


def test_seeding_an_empty_plan_says_so_and_indexes_nothing(tmp_path: Path) -> None:
    lines: list[str] = []
    written = seed_corpus(
        dsn="postgresql://unreachable/db",
        embedder_name="hashing",
        plan=SeedPlan(root=tmp_path, files=(), total_bytes=0),
        print_fn=lambda *a, **k: lines.append(" ".join(str(x) for x in a)),
    )
    assert written == 0
    assert any("Nothing to seed" in line for line in lines)


def test_a_seeding_failure_is_reported_and_never_raised(tmp_path: Path) -> None:
    """The interview is already persisted by the time this runs; it must cost a line, not a run."""
    _project(tmp_path)
    lines: list[str] = []
    written = seed_corpus(
        dsn="postgresql://recall:recall@127.0.0.1:1/recall",
        embedder_name="hashing",
        plan=plan_seed(tmp_path),
        print_fn=lambda *a, **k: lines.append(" ".join(str(x) for x in a)),
    )
    assert written == 0
    assert any("Could not seed the corpus" in line for line in lines)
    assert any("recall.cli index" in line for line in lines)


def test_the_plan_hands_the_indexer_the_set_it_measured(tmp_path: Path, monkeypatch: Any) -> None:
    """Re-walking would index a set nobody counted, which is what `files=` exists to prevent."""
    _project(tmp_path)
    plan = plan_seed(tmp_path)
    captured: dict[str, Any] = {}

    class FakeIndexer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def index_path(self, root: Any, files: Any = None, glob: Any = None) -> Any:
            captured["root"] = root
            captured["files"] = files
            captured["glob"] = glob

            class Stats:
                chunks = 7
                files = 6

            return Stats()

    import recall.index

    monkeypatch.setattr(recall.index, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        "recall.store.PgVectorStore",
        lambda *a, **k: type(
            "S",
            (),
            {
                "__enter__": lambda self: self,
                "__exit__": lambda self, *e: False,
                "check_schema": lambda self: None,
            },
        )(),
    )

    written = seed_corpus(
        dsn="postgresql://example/db",
        embedder_name="hashing",
        plan=plan,
        print_fn=lambda *a, **k: None,
    )

    assert written == 7
    assert captured["files"] == list(plan.files)
    assert captured["glob"] is None, "passing both files and glob is rejected by the indexer"
    assert Path(captured["root"]) == plan.root
