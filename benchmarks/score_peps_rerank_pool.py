"""Score the PEPs rerank x pool-width 2x2 against its preregistration.

Prior work (searched before writing this, per `benchmarks/EXPERIMENT-CONVENTION.md`):
`docs_search(source_type='memory')` on "cross-encoder reranking candidate pool width hit@5
interaction PEPs corpus" and on "reranker MiniLM versus voyage rerank-2.5 measured gain, retrieval
levers falsified private corpus" ->
[[closed-hypothesis-recall-rerank-pool-interaction-2026-08-05]] (this experiment's own memo),
[[closed-hypothesis-recall-reranker-finetune-2026-07-30]] (the LOCOMO pool-level baseline: MiniLM
+0.145, voyage +0.230, produced by `benchmarks/rerank_pool_arms.py`), and
[[project-docs-rag-reranker-voyage-vs-minilm-2026-07-30]] (the same MiniLM collapses hit@1
0.590 -> 0.190 on a bilingual corpus; it is English-only). Plus the 2026-07-22 `RE-call retrieval
levers` row in `closed_hypotheses_index.md` (rerank +0.043 within noise, pool 20->100 +0.000).
`gap_warning: false` on both queries. **The cell none of them measured is the INTERACTION** --
07-22 moved rerank and pool width one at a time -- which is what the preregistration targets.

WHY THIS EXISTS. This is the scorer, not the runner: the arms were produced by the shipped
`python -m recall.eval.labelled` (invocation recorded in `results/peps_rerank_pool/README.md`), and
their reports are committed under `results/peps_rerank_pool/`. This script turns those reports into
the preregistered verdict, and it re-derives every published figure so they can be checked rather
than merely read -- the provenance gap `rerank_pool_arms.py` calls out in its own docstring.

THE STATISTIC. All four arms ran the SAME questions against the SAME index from ONE commit, so
every delta is PAIRED. Differencing two independent per-arm CIs is the wrong statistic and badly
overstates the uncertainty on a delta. This bootstraps the delta itself by resampling questions,
and adds McNemar's exact test on the discordant pairs, which is what exposes that the pool-20
"+0.0114" is churn (broke 6, fixed 7, p=1.00) rather than a small gain.

Per-question hit vectors are reconstructed from each arm's `misses` list, which `labelled.py`
emits untruncated with question ids. That is what makes this possible without re-running anything.

INVARIANTS. Asserted, exit 1 on failure:
  - the apparatus invariant is on the SAMPLE SIZES, not the rates: `retrieval_scored_on` 88,
    `false_abstain.n` 44, `abstention_accuracy.n` 11. `false_abstain.rate` legitimately MOVES with
    `candidate_k` (0.0227 at ck20, 0.0909 at ck250) because `research_search` shares the pool
    (EVAL-002), so asserting the rate would false-void a valid arm.
  - I1 one index across arms, I2 the reranker is not inert, I4 `candidate_k` echoed as requested.
  - every published hit@5, delta and McNemar count reproduces exactly.
Bootstrap CIs are seeded (`SEED`) and reproduce, but are printed rather than asserted: they are the
one output that depends on the RNG.

Usage:  python -m benchmarks.score_peps_rerank_pool
"""
from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "peps_rerank_pool"
QUESTIONS = ROOT / "recall" / "eval" / "peps_questions.json"

REPS = 20_000
SEED = 20260805

#: Published figures, from the memo and `results/peps_rerank_pool/README.md`. Asserted below.
PUBLISHED = {
    88: {"A1": 0.6250, "A2": 0.6364, "A3": 0.6477, "A4": 0.5795},
    44: {"A1": 0.6818, "A2": 0.6818, "A3": 0.7273, "A4": 0.6136},
}
#: (broke, fixed) for rerank at each pool, n=88. Deterministic, no RNG.
PUBLISHED_MCNEMAR = {("20", 88): (6, 7), ("250", 88): (12, 6)}

#: name, baseline arm, treatment arm, comparator, threshold, what it discriminates
PREDICTIONS = (
    ("P1", "A3", "A4", ">=", 0.08, "rerank helps at a wide pool"),
    ("P2", "A2", "A4", ">=", 0.04, "THE INTERACTION -- the whole question"),
    ("P3", "A1", "A3", "<=", 0.02, "pool width alone does little"),
    ("P4", "A1", "A2", "<=", 0.06, "the weak form is weak"),
)


def rate(v: list[int]) -> float:
    return sum(v) / len(v)


def hit_vector(report: dict, arm: str, ids: list[str]) -> list[int]:
    """1 where the arm hit, 0 where the question is in the arm's `misses`."""
    missed = {m["id"] for m in report["arms"][arm]["misses"]}
    stray = missed - set(ids)
    if stray:
        raise SystemExit(f"{arm}: misses fall outside the scored set: {sorted(stray)}")
    return [0 if qid in missed else 1 for qid in ids]


def paired_delta_ci(a: list[int], b: list[int], reps: int = REPS, seed: int = SEED):
    """Bootstrap rate(b) - rate(a) by resampling QUESTIONS, preserving the pairing.

    ⚠️ The percentile convention here is the plain 0-based percentile index
    (`deltas[int(0.025*reps)]` / `deltas[int(0.975*reps)-1]`), NOT
    `recall.observability.percentile` and NOT `benchmarks/latency.py:_percentile`. Named
    explicitly because this repo has already been bitten twice by two percentile conventions
    diverging silently (see `recall/eval/metrics.py:latency_report`, which documents the same
    hazard for `benchmarks/latency.py`). At reps=20_000 the conventions differ by at most one
    sample of twenty thousand, but the published CIs were produced by THIS one, so changing it
    would move numbers that are already cited in the memo and the README.
    """
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(reps):
        pick = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(b[i] - a[i] for i in pick) / n)
    deltas.sort()
    return rate(b) - rate(a), deltas[int(0.025 * reps)], deltas[int(0.975 * reps) - 1]


def mcnemar(a: list[int], b: list[int]):
    """(a-only hits, b-only hits, exact two-sided p) -- i.e. (broke, fixed) when b adds rerank."""
    only_a = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    only_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    return only_a, only_b, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(n: int):
    """Return (ck20_report, ck250_report, scored_ids) for the n=88 or n=44 pair."""
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if n == 88:
        ids = [q["id"] for q in questions if q.get("answerable")]
    else:
        # `labelled.py` splits `fit, held = questions[::2], questions[1::2]` -- a deterministic
        # stride, no RNG -- so the held set is identical across runs and the n=44 pair is still
        # PAIRED across the two invocations.
        ids = [q["id"] for q in questions[1::2] if q.get("answerable")]
    r20 = json.loads((RESULTS / f"peps_n{n}_ck20.json").read_text(encoding="utf-8"))
    r250 = json.loads((RESULTS / f"peps_n{n}_ck250.json").read_text(encoding="utf-8"))
    return r20, r250, ids


def score(n: int, strict: bool) -> list[str]:
    failures: list[str] = []
    r20, r250, ids = load(n)
    print(f"\n{'=' * 72}\n=== n={n}  (scored set: {len(ids)} answerable questions)\n{'=' * 72}")

    if strict:
        print("\n-- apparatus invariant (on the N's, NOT the rates) --")
        for lbl, r in (("ck20 ", r20), ("ck250", r250)):
            q, fa, aa = r["questions"], r["false_abstain"], r["abstention_accuracy"]
            ok = q.get("retrieval_scored_on") == 88 and fa["n"] == 44 and aa["n"] == 11
            print(f"   {lbl}: retrieval_scored_on={q.get('retrieval_scored_on')} "
                  f"false_abstain.n={fa['n']} (rate {fa['rate']}) "
                  f"abstention_accuracy.n={aa['n']} -> {'PASS' if ok else 'VOID'}")
            if not ok:
                failures.append(f"n={n} {lbl.strip()} apparatus invariant")

    print("\n-- I1 one index / I4 candidate_k echoed --")
    for lbl, r, want in (("ck20 ", r20, 20), ("ck250", r250, 250)):
        c = r["corpus"]
        ok = r["candidate_k"] == want
        print(f"   {lbl}: table={c['table']} sources={c['sources_indexed']} chunks={c['chunks']} "
              f"candidate_k={r['candidate_k']} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"n={n} {lbl.strip()} I4")
    i1 = all(r20["corpus"][f] == r250["corpus"][f] for f in ("table", "chunks", "sources_indexed"))
    print(f"   I1 -> {'PASS' if i1 else 'FAIL'}")
    if not i1:
        failures.append(f"n={n} I1")

    vec = {"A1": hit_vector(r20, "hybrid", ids), "A2": hit_vector(r20, "hybrid+rerank", ids),
           "A3": hit_vector(r250, "hybrid", ids), "A4": hit_vector(r250, "hybrid+rerank", ids)}

    print("\n-- arms (reconstructed from `misses`; must match the published rate) --")
    for arm in sorted(vec):
        got, want = rate(vec[arm]), PUBLISHED[n][arm]
        ok = abs(got - want) < 5e-4
        print(f"   {arm}: {sum(vec[arm])}/{len(ids)} = {got:.4f}  published {want:.4f} "
              f"-> {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"n={n} {arm} hit@5 {got:.4f} != published {want:.4f}")

    print("\n-- I2 reranker live? --")
    for pool, a, b in (("20", "A1", "A2"), ("250", "A3", "A4")):
        broke, fixed, _ = mcnemar(vec[a], vec[b])
        moved = broke + fixed
        print(f"   pool {pool:>3}: broke {broke}, fixed {fixed}, moved {moved} "
              f"-> {'PASS' if moved else 'VOID (inert)'}")
        if not moved:
            failures.append(f"n={n} pool {pool} I2 inert")
        want = PUBLISHED_MCNEMAR.get((pool, n))
        if want and (broke, fixed) != want:
            failures.append(f"n={n} pool {pool} McNemar {(broke, fixed)} != published {want}")

    print("\n-- predictions (paired bootstrap 20k, 95% percentile CI) --")
    for name, a, b, op, thr, why in PREDICTIONS:
        d, lo, hi = paired_delta_ci(vec[a], vec[b])
        broke, fixed, p = mcnemar(vec[a], vec[b])
        holds = d >= thr if op == ">=" else d <= thr
        # Where the WHOLE CI sits relative to the threshold is the load-bearing fact for the
        # decision: a delta indistinguishable from zero and a delta clearing +0.04 are different
        # claims, and only the second one justified shipping.
        if op == ">=":
            band = ("CI excludes the threshold (above)" if lo >= thr
                    else "CI spans the threshold" if hi >= thr
                    else "ENTIRE CI below the threshold")
        else:
            band = ("ENTIRE CI below the threshold" if hi <= thr
                    else "CI spans the threshold" if lo <= thr
                    else "ENTIRE CI above the threshold")
        sign = "  (WRONG SIGN)" if op == ">=" and d < 0 else ""
        print(f"   {name}: {b}-{a} = {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  need {op} {thr:+.2f} "
              f"-> {'HOLDS' if holds else 'FAILS'}{sign}")
        print(f"        {band}; McNemar broke/fixed {broke}/{fixed}, exact two-sided p={p:.4f}"
              f"  | {why}")
    return failures


def main() -> int:
    if not RESULTS.is_dir():
        raise SystemExit(f"missing {RESULTS} -- run from a checkout that has the committed reports")
    failures = score(88, strict=True) + score(44, strict=False)
    print(f"\n{'=' * 72}")
    if failures:
        print("REPRODUCTION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All invariants hold and every published figure reproduces.")
    print("Verdict: P1 and P2 FAIL with the entire 95% CI below the preregistered threshold and")
    print("the wrong sign; P4 holds; P3 misses by ~1/4 of one question. H refuted, R supported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
