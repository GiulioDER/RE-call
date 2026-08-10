"""The README should describe quality gates without using a stale test-count badge."""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_the_readme_does_not_advertise_a_numeric_test_count() -> None:
    text = README.read_text(encoding="utf-8")
    assert "tests-1300" not in text
    assert "1,300+ tests" not in text


def test_the_schema_migrations_claim_matches_the_readme_body() -> None:
    text = README.read_text(encoding="utf-8")
    assert "no versioned upgrade path" not in text
    assert "ordered SQL migration path" in text
    assert "pre-tenancy tables are migrated in place" in text


def test_the_readme_has_a_clear_quickstart_and_surface_split() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## Quickstart" in text
    assert "## Product surface" in text
    assert "python -m recall.cli setup" in text
    assert "When the wizard asks whether to calibrate" in text
    assert "Run the guided setup wizard for your own corpus" in text
    assert "Declared supersession makes the current memory win" in text


def test_the_readme_says_its_numbers_are_claim_gated() -> None:
    text = README.read_text(encoding="utf-8")
    assert "tied to committed artifacts" in text
    assert "claim gate checks them in CI" in text


def test_the_readme_names_apache_and_the_citation_path() -> None:
    text = README.read_text(encoding="utf-8")
    assert "Apache 2.0 license" in text
    assert "## Citation" in text
    assert "NOTICE" in text


def test_the_readme_names_the_actual_gate_shapes() -> None:
    text = README.read_text(encoding="utf-8")
    assert "Real pgvector integration tests" in text
    assert "type checking" in text
    assert "dependency audit" in text
