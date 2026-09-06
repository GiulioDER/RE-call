from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def checker():
    path = Path(__file__).parents[1] / "scripts" / "check_broad_exception_intent.py"
    spec = importlib.util.spec_from_file_location("check_broad_exception_intent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_broad_catches_are_classified(checker) -> None:
    assert checker.find_violations() == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("try:\n    work()\nexcept Exception:\n    pass\n", "exactly one"),
        ("try:\n    work()\nexcept Exception:  # BROAD-CATCH: maybe\n    pass\n", "unknown"),
        ("try:\n    work()\nexcept Exception:  # BROAD-CATCH: fail-open\n    pass\n", None),
    ],
)
def test_marker_policy_is_strict(tmp_path: Path, checker, source: str, expected: str | None) -> None:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    violations = checker.find_violations([str(path)])
    if expected is None:
        assert violations == []
    else:
        assert expected in violations[0]
