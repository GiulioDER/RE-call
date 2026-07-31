"""Guards for the mem-bench submitter adapters (`benchmarks/membench/`).

These exist because the version of this code that produced the currently-published RE-call rows
**could not have run its own tuned arm**: it reached the Voyage reranker via
`sys.path.insert(0, "/opt/recall-ladder-v4/benchmarks/ladder/systems")` -- an absolute path on a
host that no longer exists, pointing at a branch-only duplicate of a file that is on master. Nothing
caught it because the code lived in a scratchpad, outside any repo, and was therefore never linted,
never imported by a test, and very nearly lost with the host.

So the assertions here are deliberately unglamorous: *the imports resolve*. That is the class of
failure that actually happened.

None of these need a database or credentials, so they run in CI. The two that need mem-bench
importable skip cleanly when it is not -- mem-bench is a separate repo and is not a dependency of
this one, by design (`benchmarks/membench/__init__.py` explains why).
"""
from __future__ import annotations

import importlib.util

import pytest

import recall
from benchmarks.membench import _env

_HAS_MEMBENCH = importlib.util.find_spec("membench") is not None
requires_membench = pytest.mark.skipif(
    not _HAS_MEMBENCH,
    reason="mem-bench is a separate repo; put it on PYTHONPATH to exercise the adapters",
)


def test_the_reranker_import_resolves():
    """THE regression guard. The old path could not import; this asserts the module is reachable
    from a plain checkout, with no `sys.path` surgery and no absolute host path."""
    from benchmarks.voyage_rerank import VoyageReranker

    assert VoyageReranker.__module__ == "benchmarks.voyage_rerank"


def test_no_reranker_unless_asked(monkeypatch):
    monkeypatch.delenv("MEMBENCH_RERANKER", raising=False)
    assert _env._build_reranker() is None


def test_a_non_voyage_reranker_spec_is_ignored_not_guessed_at(monkeypatch):
    """Anything that is not `voyage:<model>` yields None rather than a best-effort lookup: a
    silently-wrong reranker would change the measurement without changing the config string."""
    monkeypatch.setenv("MEMBENCH_RERANKER", "cohere:rerank-3")
    assert _env._build_reranker() is None


def test_the_voyage_branch_gets_past_the_import(monkeypatch):
    """Construction may fail for want of credentials -- that is fine and not what this pins. What
    must never happen again is `ImportError`/`ModuleNotFoundError` on the way there."""
    monkeypatch.setenv("MEMBENCH_RERANKER", "voyage:rerank-2.5")
    try:
        _env._build_reranker()
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - the regression
        pytest.fail(f"reranker import broke again: {exc}")
    except Exception:
        pass


def test_from_env_reads_k_and_candidate_k(monkeypatch):
    """The tuned arm is this same code path with different env values -- so the env actually
    reaching the config is the thing that makes "one code path, two arms" true rather than claimed.

    `_build_embedder` is stubbed because constructing a real one downloads a model; the substitution
    is what keeps this a unit test rather than a network test.
    """
    monkeypatch.delenv("MEMBENCH_RERANKER", raising=False)
    monkeypatch.setenv("MEMBENCH_K", "45")
    monkeypatch.setenv("MEMBENCH_CANDIDATE_K", "250")
    monkeypatch.setattr(_env, "_build_embedder", lambda: object())

    cfg = _env.from_env()
    assert (cfg.k, cfg.candidate_k) == (45, 250)
    assert cfg.reranker is None


def test_from_env_defaults_when_nothing_is_set(monkeypatch):
    for var in ("MEMBENCH_K", "MEMBENCH_CANDIDATE_K", "MEMBENCH_RERANKER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_env, "_build_embedder", lambda: object())

    cfg = _env.from_env()
    assert (cfg.k, cfg.candidate_k) == (5, 20)


def test_system_version_is_read_live_not_stored():
    """It must track the package, not a constant written down beside it -- that duplication is
    exactly how the board ended up carrying eight artifacts labelled with a version nothing
    confirmed. See mem-bench's `membench/sysversion.py`."""
    assert _env.system_version() == recall.__version__
    assert _env.system_version().strip()


@requires_membench
def test_both_adapters_import_and_report_the_live_version():
    from benchmarks.membench.recall_isolation import RecallIsolation
    from benchmarks.membench.recall_temporal import RecallTemporal

    for cls in (RecallIsolation, RecallTemporal):
        assert cls.name == "RE-call"
        # Read off the property without running __init__, which would want a database.
        assert cls.system_version.fget(object.__new__(cls)) == recall.__version__
