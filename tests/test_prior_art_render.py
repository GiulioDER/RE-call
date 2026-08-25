from __future__ import annotations

from tools.prior_art.loader import load_dataset
from tools.prior_art.render import render_gap_report, render_matrix, render_files, render_summary


def test_matrix_render_is_deterministic() -> None:
    dataset = load_dataset()
    assert render_matrix(dataset) == render_matrix(dataset)
    assert render_gap_report(dataset) == render_gap_report(dataset)


def test_matrix_contains_uncertainty_states_and_systems() -> None:
    dataset = load_dataset()
    matrix = render_matrix(dataset)
    assert "`unknown`" in matrix
    assert "`not_evidenced`" in matrix
    assert "Graphiti" in matrix
    assert "RE-call" in matrix
    assert "## System overview" in matrix
    assert "| System | representation | write_path | retrieval |" in matrix
    assert "## Incomplete or unresolved claims" in matrix
    assert "clm_langmem_revision_001" in matrix
    assert "## Conflicting evidence" in matrix


def test_summary_contains_unresolved_claims_and_statuses() -> None:
    summary = render_summary(load_dataset())
    assert '"unresolved_claim_ids"' in summary
    assert '"unverified_gap"' in summary
    assert '"target_combination"' in summary


def test_gap_report_renders_target_combination_without_uniqueness_claim() -> None:
    report = render_gap_report(load_dataset())
    assert "## Target combination analysis" in report
    assert "`sys_recall`" in report
    assert "`unverified_combination`" in report or "`partial_combination`" in report
    assert "not a novelty claim" in report


def test_render_check_passes_against_committed_reports() -> None:
    dataset = load_dataset()
    assert render_files(dataset, check=True) == []
