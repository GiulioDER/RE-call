"""L3: ``RECALL_AML_WORD_WINDOW_SIZE`` / ``_STRIDE`` override a windowing variant's raw windows.

Pre-registration: recall-lab ``research/preregistrations/2026-10-08-l3-smaller-windows.md``.
Default off: unset, the variant's 160/120 windows are untouched.

Red proofs, each run 2026-10-08 against the deliberate mutation named, failing at the assertion
named, then green after restoring the line:

* ``test_the_override_reaches_the_stored_windows``: the ``self._behavior = replace(...)`` line
  after ``_window_override`` removed; fails at ``service.word_window_size == 64``.
* ``test_an_overlap_too_small_for_atomic_views_stops_startup`` (added after the first version
  shipped 64/48, which passed every config test and then failed EVERY Add on the testbench with
  ``micro views must fit inside the window overlap``): the ``min_overlap`` check removed; fails
  with DID NOT RAISE. ``test_an_accepted_override_builds_atomic_views`` exercises the consumer
  (``window_views``) at the accepted 64/40.
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
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_STRIDE", "40")
    service, *_ = _service("C9_routed_specialists_grounded_graph_atomic")
    assert service.word_window_size == 64
    assert service.word_window_stride == 40


def test_an_overlap_too_small_for_atomic_views_stops_startup(monkeypatch) -> None:
    """64/48 leaves a 16-word overlap; C9's micro views need 24, so every Add would fail."""
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_SIZE", "64")
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_STRIDE", "48")
    with pytest.raises(ValueError, match="atomic views need 24"):
        _service("C9_routed_specialists_grounded_graph_atomic")


def test_an_accepted_override_builds_atomic_views(monkeypatch) -> None:
    """The consumer boundary the first version missed: an Add's atomic views at 64/40."""
    from recall.atomizer import window_views
    from recall_aml.atomic_views import ATOMIC_VIEW_STRATEGY

    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_SIZE", "64")
    monkeypatch.setenv("RECALL_AML_WORD_WINDOW_STRIDE", "40")
    size, stride = _window_override(160, 120, min_overlap=24)
    text = " ".join(f"word{i} is a fact about topic {i % 7}." for i in range(200))
    views = window_views(text, window_size=size, window_stride=stride, strategy=ATOMIC_VIEW_STRATEGY)
    assert views


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
