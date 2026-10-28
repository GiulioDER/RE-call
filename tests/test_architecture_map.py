"""Tests for `tools/architecture_map.py`.

The committed map is a CI contract (`--check` in the architecture job), so what it depends on
decides which PRs have to regenerate it. It must change when the import graph changes and must not
change otherwise: a map that hashed source bytes went stale on every code PR and made any two open
code PRs conflict on its fingerprint line.

grimp is in the `analysis` extra only, so these run in CI's architecture job, which installs it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("grimp")

_SPEC = importlib.util.spec_from_file_location(
    "architecture_map", Path(__file__).resolve().parents[1] / "tools" / "architecture_map.py"
)
assert _SPEC is not None and _SPEC.loader is not None
architecture_map = importlib.util.module_from_spec(_SPEC)
sys.modules["architecture_map"] = architecture_map
_SPEC.loader.exec_module(architecture_map)


def _make_packages(root: Path, prefix: str) -> tuple[str, str]:
    first, second = f"{prefix}_a", f"{prefix}_b"
    for name in (first, second):
        (root / name).mkdir()
        (root / name / "__init__.py").write_text("", encoding="utf-8")
    (root / first / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / second / "user.py").write_text("VALUE = 2\n", encoding="utf-8")
    return first, second


def test_a_source_edit_that_changes_no_import_leaves_the_map_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: the map depends on the import graph only, not on source bytes.

    Red proof: restoring the line `284b10ce9` rendered in `build_map`, a `Source tree fingerprint`
    sha256 over every scanned file's bytes, made this fail on the equality assertion, because the
    appended comment changed that line. (The old `build_map` took no `packages` argument, so it
    was restored into the new signature rather than run as it stood.)
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    packages = _make_packages(tmp_path, "archmap_bytes")
    before = architecture_map.build_map(packages)

    core = tmp_path / packages[0] / "core.py"
    core.write_text(core.read_text(encoding="utf-8") + "\n# a comment, no import\n", encoding="utf-8")

    assert architecture_map.build_map(packages) == before


def test_a_new_cross_package_import_changes_the_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: a new import edge still makes the committed map stale.

    This is the guarantee the check exists for, and the one dropping the byte hash must not lose.
    Red proof: deleting the `lines.extend(...)` that renders the cross package edge table in
    `build_map` made this fail on the row assertion.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    first, second = _make_packages(tmp_path, "archmap_edges")
    before = architecture_map.build_map((first, second))

    (tmp_path / second / "user.py").write_text(
        f"from {first}.core import VALUE\n", encoding="utf-8"
    )
    after = architecture_map.build_map((first, second))

    assert after != before
    assert f"| `{second}.user` | `{first}.core` |" in after
