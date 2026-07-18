"""Harness logic for the near-miss abstention evaluation (arms A/B/C), judged with fake judges.

These tests pin the ARM SEMANTICS deterministically (a real QNLI judge's quality is what the
actual evaluation run measures, not what CI asserts):
- with an accept-all judge the entailment arms cannot differ from the threshold arm on FCR;
- with a reject-all judge the entailment arms must abstain on everything.
The near-miss query set must also stay OUT of the calibration inputs — the challenge set must
not tune the threshold it is challenging.
"""
from __future__ import annotations

import math

from recall.embeddings import HashingEmbedder
from recall.eval.harness import ARMS, run_nearmiss_eval

from tests.conftest import TEST_DSN, requires_db


class AcceptAll:
    def judge(self, query: str, texts: list[str]) -> list[bool]:
        return [True] * len(texts)


class RejectAll:
    def judge(self, query: str, texts: list[str]) -> list[bool]:
        return [False] * len(texts)


def _by_arm(results, emb_name):
    return {r.arm: r for r in results if r.embedder == emb_name}


@requires_db
def test_arms_present_and_metrics_bounded():
    emb = HashingEmbedder(dim=64)
    results = run_nearmiss_eval(TEST_DSN, [emb], judge=AcceptAll())
    arms = _by_arm(results, emb.name)
    assert set(arms) == set(ARMS)
    for r in arms.values():
        for v in (r.nearmiss_fcr, r.gap_fcr, r.false_abstain, r.mrr_answerable):
            assert math.isnan(v) or 0.0 <= v <= 1.0
        assert r.entail_latency_ms_mean >= 0.0
        assert r.query_latency_ms_mean > 0.0


@requires_db
def test_accept_all_judge_cannot_change_the_threshold_arm():
    emb = HashingEmbedder(dim=64)
    arms = _by_arm(run_nearmiss_eval(TEST_DSN, [emb], judge=AcceptAll()), emb.name)
    # a judge that trusts everything degenerates arm B to arm A exactly
    assert arms["threshold+entail"].nearmiss_fcr == arms["threshold"].nearmiss_fcr
    assert arms["threshold+entail"].false_abstain == arms["threshold"].false_abstain
    assert arms["threshold+entail"].mrr_answerable == arms["threshold"].mrr_answerable
    # the threshold arm pays zero judge latency
    assert arms["threshold"].entail_latency_ms_mean == 0.0


def test_near_miss_distractor_ids_resolve_against_the_eval_corpus():
    # deferred DAT-001: distractor_ids were documentation-only — a renamed corpus file would
    # silently degrade the near-miss set into far-gap queries, flattering the threshold arm
    import json
    from pathlib import Path

    from recall.index import chunk_text

    eval_dir = Path("recall/eval")
    nearmiss = json.loads((eval_dir / "near_miss.json").read_text(encoding="utf-8"))
    assert nearmiss, "near-miss set must not be empty"
    for q in nearmiss:
        assert q["distractor_ids"], f"{q['id']}: no distractor declared"
        for did in q["distractor_ids"]:
            fname, ord_s = did.rsplit(":", 1)
            f = eval_dir / "corpus" / fname
            assert f.is_file(), f"{q['id']}: distractor file {fname} missing from corpus"
            n_chunks = len(chunk_text(f.read_text(encoding="utf-8-sig")))
            assert int(ord_s) < n_chunks, f"{q['id']}: chunk ord {ord_s} out of range"


@requires_db
def test_reject_all_judge_abstains_on_everything():
    emb = HashingEmbedder(dim=64)
    arms = _by_arm(run_nearmiss_eval(TEST_DSN, [emb], judge=RejectAll()), emb.name)
    for arm in ("threshold+entail", "entail-only"):
        assert arms[arm].nearmiss_fcr == 0.0   # nothing near-miss survives
        assert arms[arm].false_abstain == 1.0  # ...at the cost of abstaining on every answerable
