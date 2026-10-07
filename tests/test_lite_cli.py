"""`recall index` and `recall search` on a `sqlite:///` DSN: no database server, no Docker.

Invariants and the failure each one catches:
- L1 `recall index` then `recall search` work on a lite DSN, and the search answers from the memo
  that was indexed.
- L2 `recall index` on a lite store fits and stores a calibration once the corpus can test itself,
  because a local install has nobody to run one by hand.
- L3 the generation route is refused on a lite DSN with a message that says why and what to set,
  before anything is opened.

Red proof, 2026-10-07, each failing in its intended assertion (JUnit XML), then restored and green:
- B1 (L1, L3) baseline `390447ec`, the commit before this wiring: `recall index` on a lite DSN
  raised from psycopg, so L1 failed "recall index did not index the lite store"; L3 failed
  "the generation route was not refused for the lite store".
- R1 (L2) the `ensure_calibrated` call after indexing removed: "indexing did not calibrate".
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from recall.cli import main
from recall.lite import LiteStore
from tests.test_lite_calibration import _write_memos

_MEMOS = {
    "rate.md": "# Rate limit\n\nThe public API rate limit is 100 requests per second per key.\n",
    "deploy.md": "# Deploy day\n\nDeploys go out on Tuesdays from the release branch.\n",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _run(argv: list[str]) -> str:
    """The command's stdout, or the exit message it stopped with: either way something to assert."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            main(argv)
    except SystemExit as exc:
        return out.getvalue() + f"\nEXIT: {exc.code}"
    except Exception as exc:  # noqa: BLE001  # the pre-wiring failure is an arbitrary driver error
        return out.getvalue() + f"\nRAISED: {type(exc).__name__}: {exc}"
    return out.getvalue()


def _memos(root: Path, memos: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in memos.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def _dsn(tmp_path: Path) -> str:
    return "sqlite:///" + (tmp_path / "store" / "memory.db").as_posix()


def test_index_then_search_on_a_lite_dsn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L1."""
    folder = _memos(tmp_path / "mem", _MEMOS)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    indexed = _run([*base, "index", str(folder)])
    assert "indexed 2 chunks from 2 files" in indexed, f"recall index did not index the lite store: {indexed}"
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")
    found = _run([*base, "search", "what is the API rate limit", "-k", "2"])
    hits = [line for line in found.splitlines() if ".md" in line and "cos=" in line]
    assert hits and "rate.md" in hits[0], f"the search did not answer from the indexed memo: {found}"


def test_indexing_a_lite_store_calibrates_it(tmp_path: Path) -> None:
    """L2."""
    folder = tmp_path / "mem"
    _write_memos(folder, 48)  # distinct made-up subjects, so questions can be generated from them
    out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path), "index", str(folder)])
    assert "calibration: " in out, f"indexing did not calibrate: {out}"
    with LiteStore(tmp_path / "store" / "memory.db", dim=64) as store:
        assert store.latest_calibration_json() is not None, "indexing did not calibrate"


def test_the_generation_route_is_refused_on_a_lite_dsn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L3."""
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path), "search", "anything"])
    assert "has no generations" in out and "RECALL_INDEX_MODE=legacy" in out, (
        f"the generation route was not refused for the lite store: {out}"
    )
    assert not (tmp_path / "store" / "memory.db").exists(), "the refused command still created the store"
