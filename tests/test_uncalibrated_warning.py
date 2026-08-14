"""An untuned constant gating abstention must announce itself.

`DEFAULT_GAP_THRESHOLD` is an absolute cosine, and cosine levels are a property of how a model was
trained rather than a quantity comparable across models. Measured over two corpora and three
embedders: the default sits at the 0th percentile of five of those six top-1 distributions and at
the 16th of the sixth — inert for most models, and for one discarding roughly a sixth of queries
into empty retrieval, which the answerer turns into a refusal and the caller reads as a wrong
answer.

The failure is invisible on the DEFAULT embedder (bge-small's observed minimum is 0.63, above the
0.50 floor), so only a user who changes embedder is exposed — and got no signal at all. These tests
pin the signal.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import subprocess
import sys

import pytest

from recall.calibration import ENV_VAR, Calibration, save
from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.trust import _WARNED_UNCALIBRATED, _warn_uncalibrated
from tests.conftest import dev_search_uncalibrated
from tests.conftest import requires_db

DOC = "The API rate limit is one hundred requests per second per client key.\n"


@pytest.fixture(autouse=True)
def _isolated_calibration(tmp_path, monkeypatch):
    """Per-test calibration path, and a cleared warn-once registry.

    The registry is process-global by design (a long-running server should say this once per model,
    not once per query), so it must be reset around each test or the second test to run would see
    a silence produced by the first rather than by the code under test.
    """
    from recall.calibration import _LOAD_CACHE

    monkeypatch.setenv(ENV_VAR, str(tmp_path / "calibration.json"))
    _LOAD_CACHE.clear()
    _WARNED_UNCALIBRATED.clear()
    yield
    _LOAD_CACHE.clear()
    _WARNED_UNCALIBRATED.clear()


def _index(tmp_path, store):
    (tmp_path / "rate.md").write_text(DOC, encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=64)).index_path(tmp_path)


@requires_db
def test_warns_when_the_embedder_has_no_calibration(tmp_path, make_store, caplog):
    """The default is applied to an unmeasured model — say so."""
    store = make_store(64)
    _index(tmp_path, store)

    with caplog.at_level(logging.WARNING):
        dev_search_uncalibrated(store, HashingEmbedder(dim=64), "API rate limit", k=5)

    assert any("no calibration found" in r.message for r in caplog.records), (
        "an untuned constant gated abstention and nothing said so"
    )


@requires_db
def test_warns_when_only_a_legacy_unbound_file_exists(tmp_path, make_store, caplog):
    store = make_store(64)
    _index(tmp_path, store)
    embedder = HashingEmbedder(dim=64)
    save(Calibration(embedder=embedder.name, threshold=0.11, scale=0.05))

    with caplog.at_level(logging.WARNING):
        dev_search_uncalibrated(store, embedder, "API rate limit", k=5)

    assert any("no calibration found" in r.message for r in caplog.records)


@requires_db
def test_does_not_warn_when_a_calibration_is_passed_explicitly(tmp_path, make_store, caplog):
    """An explicit argument is a deliberate choice, not an omission."""
    store = make_store(64)
    _index(tmp_path, store)
    embedder = HashingEmbedder(dim=64)
    explicit = Calibration(embedder=embedder.name, threshold=0.11, scale=0.05)

    with caplog.at_level(logging.WARNING):
        dev_search_uncalibrated(store, embedder, "API rate limit", k=5, calibration=explicit)

    assert not any("no calibration found" in r.message for r in caplog.records)


@requires_db
def test_warns_once_per_embedder_not_once_per_query(tmp_path, make_store, caplog):
    """Retrieval is a hot path; a per-query warning would be unusable in a server and would train
    operators to filter the whole category out."""
    store = make_store(64)
    _index(tmp_path, store)
    embedder = HashingEmbedder(dim=64)

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            dev_search_uncalibrated(store, embedder, "API rate limit", k=5)

    hits = [r for r in caplog.records if "no calibration found" in r.message]
    assert len(hits) == 1, f"expected exactly one warning across five queries, got {len(hits)}"


def test_an_unreadable_calibration_is_announced_once_not_once_per_query(
    tmp_path, monkeypatch, caplog
):
    """Same rule one layer down: `load_for` runs per query, so its own warning must not repeat.

    A file that is present but unreadable (EACCES, a dropped mount) has no content to key a cache
    entry on, so the loader records a sentinel instead. Without it, keying the cache on the file's
    bytes means the bytes are never obtained, nothing is ever cached, and the warning fires on
    every search — the exact noise `test_warns_once_per_embedder_not_once_per_query` forbids one
    layer up. This test fails against a loader that returns early on OSError without recording it.
    """
    from pathlib import Path

    from recall.calibration import load_for

    path = tmp_path / "calibration.json"
    save(Calibration(embedder="e1", threshold=0.42, scale=0.05), path)

    def _denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_bytes", _denied)

    with caplog.at_level(logging.WARNING):
        results = [load_for("e1", path) for _ in range(5)]

    assert results == [None] * 5, "an unreadable file must degrade to the uncalibrated fallback"
    hits = [r for r in caplog.records if "unreadable calibration file" in r.message]
    assert len(hits) == 1, f"expected exactly one warning across five loads, got {len(hits)}"


@requires_db
def test_the_warning_names_the_embedder_and_the_remedy(tmp_path, make_store, caplog):
    """A warning that does not say WHICH model is unmeasured, or what to do, is a nuisance rather
    than a diagnosis — the reader has to reproduce the investigation to act on it."""
    store = make_store(64)
    _index(tmp_path, store)
    embedder = HashingEmbedder(dim=64)

    with caplog.at_level(logging.WARNING):
        dev_search_uncalibrated(store, embedder, "API rate limit", k=5)

    msg = next(r.getMessage() for r in caplog.records if "no calibration found" in r.getMessage())
    assert embedder.name in msg
    assert "recall calibration calibrate" in msg


def _argparse_rejects(command: str, cwd) -> bool:
    """True when the CLI refuses `command` as unparseable, rather than for any later reason.

    Only argparse is under test here, and it runs before the DSN is ever dialled, so the run is
    pointed at a closed local port: whatever happens after parsing is irrelevant and must not be
    allowed to hang.

    `cwd` is a scratch directory so no repository `.env` is picked up, and PYTHONPATH pins the
    import to THIS checkout. One editable install serves many worktrees here, so a bare
    `python -m recall.cli` run from elsewhere would parse a different tree's CLI and the result
    would say nothing about this one.
    """
    tokens = command.split()
    assert tokens[0] == "recall", f"expected a `recall` command, got {command!r}"
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    argv = [
        sys.executable,
        "-m",
        "recall.cli",
        "--serving-dsn",
        "postgresql://u:p@127.0.0.1:1/db?connect_timeout=1",
        *tokens[1:],
    ]
    done = subprocess.run(
        argv, capture_output=True, text=True, cwd=cwd, env=env, timeout=120
    )
    return "unrecognized arguments" in done.stderr or "invalid choice" in done.stderr


def test_the_remedy_the_warning_prints_is_a_command_that_actually_parses(tmp_path):
    """The warning names a command to run. For several releases it named one that cannot run.

    The generation-bound measurement lives under the `calibration` subcommand group. Top-level
    `calibrate` takes a POSITIONAL queries file and defines neither `--generation` nor `--queries`,
    so `recall calibrate --generation G --queries FILE --publish` exits 2 on argparse. The test
    that was supposed to pin this asserted the substring "recall calibrate", which the broken form
    satisfies, so the guard could never have caught it.

    Run the string the warning actually prints. It may fail for any later reason, the closed port
    being the likely one, but it must not fail because the CLI does not understand it.
    """
    _WARNED_UNCALIBRATED.discard("probe-embedder")
    logger = logging.getLogger("recall.trust")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        _warn_uncalibrated("probe-embedder")
    finally:
        logger.removeHandler(handler)
        _WARNED_UNCALIBRATED.discard("probe-embedder")

    msg = next(r.getMessage() for r in records if "no calibration found" in r.getMessage())
    quoted = re.findall(r"`([^`]+)`", msg)
    assert quoted, f"the warning names no command at all: {msg!r}"
    command = next(c for c in quoted if c.startswith("recall "))

    assert not _argparse_rejects(command, tmp_path), (
        f"the warning tells the reader to run {command!r}, and the CLI rejects it as unparseable"
    )

    # Canary. The check above is only meaningful if it can fail, so prove it still catches the
    # exact form this test exists to prevent. Without this, a checker that silently stopped
    # detecting anything would look identical to a clean run.
    broken = "recall calibrate --generation G --queries FILE --publish"
    assert _argparse_rejects(broken, tmp_path), (
        "the parse check no longer detects the historical broken form, so it is proving nothing"
    )
