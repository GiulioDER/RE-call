"""The verify-recall skill (``.claude/skills/verify-recall/``): its feature map stays tied to the
feature ids CI already checks, and its control CLI can only touch runs it owns, keeps every piece
of evidence through cleanup, and reports a strict refusal by its code.

Invariants: every ``references/features/*.md`` but the index is named after a feature id in
``docs/feature_map.toml``, has the four sections the map's readers rely on, and is linked from the
index; ``control_recall`` refuses a run outside its root or launched by another checkout, a
cleanup leaves every evidence file, and a strict search's code is read from stderr. Failure modes
caught: a verification map that drifts from the feature map, and a driver that deletes proof or
another session's data.

Red proof, 2026-11-25, one mutation per test, run alone and restored; each failed at its own
assertion:

- a feature file renamed to an id that does not exist (``lite.md`` -> ``lite-store.md``):
  ``test_every_feature_file_is_a_feature_map_id``.
- the ``## Gotchas`` heading removed from ``forget.md``: ``test_every_feature_file_has_the_four_sections``.
- the ``forget`` line removed from the index: ``test_the_index_links_every_feature_file``.
- ``_run_dir`` without its ``path.parent != ROOT`` test: ``test_a_run_outside_the_root_is_refused``.
- ``_run_dir`` without its checkout test: ``test_a_run_another_checkout_launched_is_refused``.
- ``cleanup`` removing the whole run instead of ``scratch/``:
  ``test_a_strict_search_refuses_by_code_and_cleanup_keeps_the_evidence``.
- ``search`` reading only stdout for the code: the same test, on its ``code`` assertion.
"""

from __future__ import annotations

import importlib.util
import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SKILL = ROOT / ".claude" / "skills" / "verify-recall"
FEATURES = SKILL / "references" / "features"
SECTIONS = ["Sub-features", "How to get to it (user POV)", "Driving it with control_recall", "Gotchas"]

SPEC = importlib.util.spec_from_file_location("control_recall", SKILL / "control_recall.py")
assert SPEC and SPEC.loader
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def _feature_files() -> list[Path]:
    return sorted(p for p in FEATURES.glob("*.md") if p.name != "README.md")


def _call(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[dict[str, object], int]:
    status = control.main(list(argv))
    return json.loads(capsys.readouterr().out), status


def test_every_feature_file_is_a_feature_map_id() -> None:
    ids = {f["id"] for f in tomllib.loads((ROOT / "docs" / "feature_map.toml").read_text(encoding="utf-8"))["feature"]}
    assert [p.stem for p in _feature_files() if p.stem not in ids] == []


def test_every_feature_file_has_the_four_sections() -> None:
    for path in _feature_files():
        headings = re.findall(r"^## (.+)$", path.read_text(encoding="utf-8"), re.M)
        assert headings == SECTIONS, path.name


def test_the_index_links_every_feature_file() -> None:
    linked = set(re.findall(r"\]\(([\w-]+\.md)\)", (FEATURES / "README.md").read_text(encoding="utf-8")))
    assert linked == {p.name for p in _feature_files()}


def test_a_run_outside_the_root_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path / "root")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "run.json").write_text(json.dumps({"checkout": str(control.CHECKOUT)}), encoding="utf-8")
    out, status = _call(capsys, "cleanup", "--run", "../elsewhere")
    assert status == 2 and "fix" in out
    assert (tmp_path / "elsewhere" / "run.json").is_file()


def test_a_run_another_checkout_launched_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    run = tmp_path / "theirs"
    (run / "scratch").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"checkout": str(tmp_path / "another-checkout")}), encoding="utf-8")
    out, status = _call(capsys, "cleanup", "--run", "theirs")
    assert status == 2 and "another" in str(out["error"])
    assert (run / "scratch").is_dir()


def test_a_strict_search_refuses_by_code_and_cleanup_keeps_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    launched, status = _call(capsys, "launch")
    assert status == 0 and launched["calibration"] == "uncertified"
    run = str(launched["run"])
    searched, _ = _call(capsys, "search", "--run", run, "how long do pricing snapshot cache entries live")
    assert (searched["exit"], searched["code"], searched["rows"]) == (1, "CALIBRATION_UNCERTIFIED", [])
    cleaned, _ = _call(capsys, "cleanup", "--run", run)
    assert not (tmp_path / run / "scratch").exists()
    assert all(Path(str(e)).is_file() for e in (launched["evidence"], searched["evidence"]))
    assert cleaned["evidence_kept"] == ["001-launch-index.json", "002-search.json"]
