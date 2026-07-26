"""Thread budget for the local embedder.

Not a performance nicety: in an unprivileged container `os.cpu_count()` reports the HOST's cores
while a cgroup quota caps actual runtime. fastembed sizes its pool from the former, so running N
worker processes multiplies N x (host cores) threads onto a fraction of that — measured at ~945
threads against a ~61-CPU quota, which thrashes.
"""
from __future__ import annotations

from recall.embeddings import resolve_thread_budget


def test_resolve_thread_budget_prefers_an_explicit_setting():
    assert resolve_thread_budget(env={"RECALL_EMBED_THREADS": "8"}, cpu_count=256) == 8


def test_resolve_thread_budget_is_none_when_unset_so_fastembed_keeps_its_default():
    # Absent an explicit budget this must not silently impose one: the single-process sweep showed
    # throughput rising monotonically to the default (2.2 -> 5.7 -> 9.0 -> 10.3 docs/s at 1/8/32/
    # default threads), so capping by default would SLOW DOWN every ordinary single-process user.
    assert resolve_thread_budget(env={}, cpu_count=256) is None


def test_resolve_thread_budget_ignores_junk_rather_than_crashing_a_long_run():
    # A typo'd env var must not take down hour-eight of an overnight job.
    assert resolve_thread_budget(env={"RECALL_EMBED_THREADS": "eight"}, cpu_count=64) is None
    assert resolve_thread_budget(env={"RECALL_EMBED_THREADS": "0"}, cpu_count=64) is None
    assert resolve_thread_budget(env={"RECALL_EMBED_THREADS": "-4"}, cpu_count=64) is None


def test_resolve_thread_budget_caps_at_the_visible_cpu_count():
    # Asking for more threads than there are CPUs is always a mistake; honouring it would recreate
    # the oversubscription this exists to prevent.
    assert resolve_thread_budget(env={"RECALL_EMBED_THREADS": "999"}, cpu_count=64) == 64
