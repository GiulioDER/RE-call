# Verdict — Arm R: which regime is MTRAG?

**Date:** 2026-08-05 · Pre-registration:
[`benchmarks/PREREGISTRATION-mtrag-abstention.md`](../../benchmarks/PREREGISTRATION-mtrag-abstention.md)
§1–§2, written before this ran and **not edited**. Runner:
[`benchmarks/mtrag/probe/arm_r_separability.py`](../../benchmarks/mtrag/probe/arm_r_separability.py).
Artifact archived on VPS2 at `/var/lib/recall-benchmarks/2026-08-05-mtrag-armR-separability/`
with a SHA256 manifest, because `/var/tmp` is not durable.

## P1 CONFIRMED, and not marginally

| | |
|---|---|
| signal | dense top-1 cosine, the shipped one |
| separability (Mann-Whitney AUC) | **0.6212**, 95% CI **[0.5504, 0.6920]** |
| answerable | n=709, mean cosine 0.7595 |
| unanswerable | n=55, mean cosine 0.7320 |
| `Calibration.certified` | **False** |

Preregistered: **≤ 0.80 → near-miss regime**; ≥ 0.90 → P1 falsified. The point estimate is 0.621
and **the entire confidence interval sits below 0.80**, so this is not a boundary call.

The library refused certification in its own words, which is the behaviour
[[project-recall-abstention-bounded-domain-2026-07-24]] shipped instead of a fix:

> separability 0.621 [0.550, 0.692] < 0.9: answerable and unanswerable scores overlap, so **NO
> threshold separates them** — this one will refuse real answers, reject unanswerable queries, or
> both, and moving it only trades one error for the other.

**MTRAG is harder than the corpora where the six signals were already closed.** Dense cosine scored
0.753 there and 0.78 on the 2026-07-28 replication. Here it is **0.621**, materially closer to no
signal at all (0.50) than to either prior figure. The cheap signal is *weaker* on MTRAG than on the
benchmarks where it had already been judged inadequate.

## What that costs, in the currency the benchmark pays in

Preregistration §3 gives the payoffs: a correct abstention scores exactly 1.0, a false abstention
exactly 0.0, answering an unanswerable exactly 0.0, and answering pays **a = 0.4199**. That
constant is measured over **ANSWERABLE + PARTIAL**, so the population here is the same one:
**n = 777 answer-pays against 55 unanswerable**, with CONVERSATIONAL excluded as payoff-neutral.

| rule | threshold | false abstention | caught | mean | vs always-answer |
|---|---|---|---|---|---|
| always answer | — | — | — | 0.3921 | — |
| library `best_threshold` (q05/q95 mid) | 0.7340 | **34.5%** | 52.7% | 0.2917 | **−0.1004** |
| **preregistered break-even, p\* = 0.2957** | 0.7429 | **38.9%** | 56.4% | 0.2770 | **−0.1152** |
| in-sample optimum over breakpoints | 0.5665 | 0.3% | 1.8% | 0.3923 | **+0.0002** |
| **oracle** (perfect detector) | — | 0 | 100% | 0.4582 | **+0.0661** |

Three things worth stating plainly.

**Both principled operating points are worse than doing nothing.** The library's threshold loses
−0.1004; **the rule this preregistration itself derived loses −0.1152**, the worst of the three. It
refuses 38.9% of the questions it could have answered in order to catch 56.4% of the unanswerable
ones, and at a 6.6% base rate that trade is badly negative. The preregistered rule being the worst
performer is worth more than a rule that happened to work: the payoff arithmetic in §3 is correct,
and it is the *signal* underneath it that has nothing to give.

**No threshold rescues it, and this is now exact rather than sampled.** The optimum is taken over
every score breakpoint, not a 0.01 lattice, because the objective is a step function that changes
only at observed values. The best achievable gain is **+0.0002** against an oracle worth
**+0.0661** — the shipped signal recovers **0.3% of the available bound**. Same conclusion as dead
end #1 ("retune the threshold"), now on a fourth corpus.

**And that 0.3% is in-sample**, chosen and scored on the same 832 tasks, so it is an upper bound on
what a held-out threshold would achieve. The threshold-free immunity `separability` has does not
transfer to a cost table sitting next to it.

## The shipped guard fired zero times

`gap_warning` fired on **0 of 842 queries** — 0/709 answerable, 0/55 unanswerable, 0/68 partial,
0/10 conversational.

That is [[project-recall-threshold-embedder-fragile-2026-07-28]] reproduced on a new corpus: the
0.50 floor is **inert on `bge-small`**. Not "rarely fires". Never fires, on any class, in 842
queries. A guard that reads as protection and cannot fire.

## Diagnostics the primary contrast excludes

| class | n | mean cosine | AUC vs unanswerable |
|---|---|---|---|
| PARTIAL | 68 | 0.7240 | 0.4623 |
| CONVERSATIONAL | 10 | 0.6669 | **0.1873** |

PARTIAL is slightly *inverted* against unanswerable, which is consistent with relevance not being
answerability. CONVERSATIONAL is strongly inverted and low-cosine, which is unsurprising: a closing
pleasantry resembles no passage. It is also payoff-neutral (§ probe verdict: all 90 of its cells
score 1.0 regardless), so it is excluded rather than folded in.

**Per domain**, all four in the same weak band. Counts included because with 55 unanswerable tasks
split four ways each figure rests on ~a dozen negatives, and the same Hanley–McNeil width that
makes the primary interval wide makes these far wider. Read the ordering, not the values:

| domain | n answerable / unanswerable | AUC |
|---|---|---|
| clapnq | 192 / 16 | 0.6761 |
| fiqa | 163 / 13 | 0.6456 |
| cloud | 177 / 14 | 0.6364 |
| govt | 177 / 12 | 0.5330 |

`govt` is the lowest, but on n=12 negatives that is not a claim I would defend as a domain effect.

## A prediction I made before the run, which was wrong

On the 20-task smoke the unanswerable mean cosine came out *above* the answerable mean, and I
flagged that it might be the BEAM-style inversion. **It was not.** At n=55 the direction is normal
(0.7595 answerable against 0.7320 unanswerable). The smoke signal was noise on 2 samples, as
labelled at the time. Recording it because it was stated publicly beforehand.

## Consequence

**Arm L is warranted**, on the preregistration's own terms. The cheap signal is not weak here, it
is absent: +0.0002 of a +0.0661 bound, in-sample. If the abstention lever on MTRAGEval is reachable at all,
it is reachable only by the reading step, which is exactly what
[[project-recall-abstention-bounded-domain-2026-07-24]] named as the sole remaining remedy.

## Limits

- **n_unanswerable = 55.** The Hanley–McNeil interval is dominated by the smaller class and is
  correspondingly wide, [0.5504, 0.6920]. The verdict survives it only because the *whole* interval
  is below the 0.80 bound; no claim here rests on the point estimate's third decimal.
- **Measured on MTRAG. MTRAG-UN is held out and was not touched.** MTRAG-UN is purpose-built to be
  harder on exactly this axis, so 0.621 should be read as an **upper** bound on what the cheap
  signal would do there, not a forecast.
- The cost table applies per-task payoffs derived from per-cell (model × task) means. It is a
  first-order model of the decision, not a measurement of any single system's score.
- **Retrieval ran on the stack that built the index** (`3d3c905`), not master, because master's
  `recall/embeddings.py` differs by 370 lines (#200's profile registry, the change that invalidated
  every embedding cache) and a query-side/document-side mismatch would produce wrong cosines that
  still look like numbers. The analysis functions (`separability`, `separability_interval`,
  `from_samples`, `best_threshold`) and all four constants were verified **byte-identical** to
  `origin/master` before running.
- VPS2 was at load 19–25 on 12 cores throughout. **Timing figures are diagnostic only.**
- **Retrieval was approximate.** Every top-1 cosine came from an HNSW walk (`ORDER BY embedding
  <=> …`), and `recall/calibration.py` itself records that HNSW rebuild nondeterminism moved a
  comparable operating point across a 0.40–0.84 range on one host (issue #26). The AUC at n=709/55
  is robust to that. **The cost table's fine structure is not**: its winning point turns on 1
  caught unanswerable against 2 false abstentions, and a single ANN miss would move it.
- **The cost table's in-sample optimum is an upper bound**, not an estimate of held-out gain.

## Audit trail

A `bug-auditor` pass on the staged diff raised six findings, each checked before acceptance.
**BUG-001** was real and material: the cost table used the payoff `a = 0.4199` (defined over
ANSWERABLE + PARTIAL) while building its population from ANSWERABLE + UNANSWERABLE only, dropping
68 tasks — the constant and the population disagreed. Corrected above, which moved every cell and
made the conclusion **stronger**. **BUG-003** was real: the evaluated point was the library's
q05/q95 midpoint, not §3's break-even `p*`, which lives in probability space; both are now
reported, and the preregistered one turns out to be the worst. **BUG-002** (a 0.01 lattice cannot
find the optimum of a step function) was a valid method defect whose **empirical impact was nil** —
the exact breakpoint sweep returns the same +0.0002. **BUG-004** (in-sample selection),
**BUG-005** (per-domain counts) and **BUG-006** (HNSW approximation) are addressed above.

The auditor separately confirmed the load-bearing mechanical question: with `use_sparse=False`,
RRF is used **only for ordering** and every surviving score is overwritten with the raw dense
cosine, so `hits[0].score` genuinely is the top-1 cosine. It verified this against the revision
that actually ran, not master, because `recall/store.py` differs by 741 lines between them.
