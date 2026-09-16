"""The compatibility policy names the repository's executable upgrade evidence."""

from pathlib import Path


POLICY = Path(__file__).resolve().parent.parent / "docs" / "COMPATIBILITY.md"


def test_compatibility_policy_is_present_and_points_at_executable_contracts() -> None:
    text = POLICY.read_text(encoding="utf-8")
    required = (
        "## Versioning",
        "## Python API",
        "## CLI and MCP contracts",
        "## PostgreSQL schema and upgrades",
        "## Lineage and calibration",
        "0.MINOR.PATCH",
        "forward-only",
        "immutable",
        "tests/test_schema_migrations.py",
        "tests/test_api_doc_drift.py",
    )
    missing = [claim for claim in required if claim not in text]
    assert not missing, f"COMPATIBILITY.md is missing policy anchors: {missing}"
