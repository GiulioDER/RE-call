"""L3: ``RECALL_AML_WORD_WINDOW_SIZE`` / ``_STRIDE`` override a windowing variant's raw windows.

Pre-registration: recall-lab ``research/preregistrations/2026-10-08-l3-smaller-windows.md``.
Default off: unset, the variant's 160/120 windows are untouched.

Red proofs, each run 2026-10-08 against the deliberate mutation named, failing at the assertion
named, then green after restoring the line:

* ``test_the_override_reaches_the_stored_windows``: the ``self._behavior = replace(...)`` line
  after ``_window_override`` removed; fails at ``service.word_window_size == 64``.
* ``test_unset_keeps_the_variant_windows``: ``_window_override`` returns ``(64, 48)`` when both
  variables are unset; fails at ``service.word_window_size == 160``.
* ``test_a_stride_larger_than_the_size_stops_startup``: the ``new_stride > new_size`` clause
  removed; fails with DID NOT RAISE.
* ``test_a_variant_without_windows_refuses_the_override``: the ``size is None`` refusal replaced
  by a 160-word default; fails at the ``match`` (the stride check raises a different message).
"""

from __future__ import annotations

import pytest

from recall_aml.service import _window_override
from tests.test_aml_specialist_fusion import _service


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("RECALL_AML_WORD_WINDOW_SIZE", "RECALL_AML_WORD_WINDOW_STRIDE"):
        monkeypatch.delenv(name, raising=False)


def test_the_override_reaches_the_stored_windows(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_SIZE", "64")
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_STRIDE", "48")
    service, *_ = _service("C9_routed_specialists_grounded_graph_atomic")
    assert service.word_window_size == 64
    assert service.word_window_stride == 48


def test_unset_keeps_the_variant_windows() -> None:
    service, *_ = _service("C9_routed_specialists_grounded_graph_atomic")
    assert service.word_window_size == 160
    assert service.word_window_stride == 120


def test_a_stride_larger_than_the_size_stops_startup(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_SIZE", "40")
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_STRIDE", "60")
    with pytest.raises(ValueError, match="stride may not exceed"):
        _service("C9_routed_specialists_grounded_graph_atomic")


def test_a_variant_without_windows_refuses_the_override(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_SIZE", "64")
    with pytest.raises(ValueError, match="needs a variant that builds word windows"):
        _window_override(None, None)


def test_size_alone_keeps_the_variant_stride_when_it_fits(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_SIZE", "200")
    assert _window_override(160, 120) == (200, 120)
