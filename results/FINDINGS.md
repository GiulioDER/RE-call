# recall — evaluation findings

Interpreted results. The reproducible ablation numbers are in [`RESULTS.md`](RESULTS.md) (run
`make eval`); the per-embedder threshold numbers below come from `recall.eval.calibrate.calibrate()`.

## 1. Hybrid + rerank helps — where the embedder isn't already saturated

On the weak, non-semantic hashing embedder, quality climbs monotonically as we add the sparse leg
and then the cross-encoder reranker:

| fusion | MRR | nDCG@10 |
|---|---|---|
| dense only | 0.63 | 0.72 |
| + sparse (hybrid) | 0.74 | 0.80 |
| + cross-encoder rerank | 1.00 | 1.00 |

On the strong FastEmbed (bge-small) embedder, dense retrieval already scores nDCG@10 0.97 (MRR
0.96) and the hybrid arm saturates the corpus at 1.00, so the reranker has nothing left to gain. Honest reading: **hybrid + rerank buys
the most on weaker embedders or harder corpora; on an easy corpus with a strong embedder it is
redundant.** A rigorous eval has to be able to show that, not just a win.

## 2. The honest negative result: a fixed gap threshold does NOT transfer across embedders

The gap guard fires when the best dense cosine for a query falls below a threshold (default 0.50).
We measured the top-cosine distribution for answerable vs. unanswerable queries per embedder:

| embedder | answerable cos (min–max) | unanswerable cos (min–max) | separable? | good threshold | FCR @0.50 | FCR @calibrated |
|---|---|---|---|---|---|---|
| hashing-64 | 0.30 – 0.68 | 0.35 – 0.53 | no (overlap) | — | 0.20\* | — |
| bge-small (FastEmbed) | 0.70 – 0.90 | 0.51 – 0.64 | yes | ~0.70 | **1.00** | **0.00** |
| voyage-3\† | 0.53 – 0.70 | 0.09 – 0.32 | yes | ~0.50 | **0.00** | **0.00** |

Three embedders, three completely different cosine regimes. The fixed 0.50 threshold happens to sit
in Voyage's clean gap (unanswerable ≈ 0.1–0.3, answerable ≈ 0.5–0.7), sits *below the entire* bge
distribution (so the guard never fires — FCR 1.00), and lands inside hashing's overlap. **It works
for one strong model by luck, fails for another strong model, and cannot work for the weak one.**
(\* hashing's 0.20 at 0.50 is misleading: with overlapping distributions the guard also wrongly
flags answerable queries whose cosines sit below 0.50, and its error-minimizing threshold (~0.30)
simply stops firing at all — FCR 1.00. No threshold works.
\† voyage-3 was measured on the v0.1 corpus; the cloud row is not re-runnable key-free.)

Two lessons:

- **The default 0.50 is miscalibrated for a strong dense embedder.** bge-small's cosines live in
  roughly [0.50, 0.90]; 0.50 sits *below the entire distribution*, so the guard almost never fires
  and the false-confident rate on unanswerable queries is 1.00. Recalibrated to ~0.70 — the gap
  between the unanswerable ceiling (0.64) and the answerable floor (0.70) — the guard becomes
  perfect: FCR 0.00, with cleanly separable distributions.
- **Gap-detection quality is bounded by the embedder.** hashing-64's answerable and unanswerable
  distributions overlap, so no single threshold separates them: a weak, non-semantic embedder
  cannot support reliable gap detection at all, at any threshold. (0.50 scores FCR 0.20 while also
  wrongly flagging *answerable* queries whose cosines sit below 0.50.)

Takeaway for anyone building gap/abstention into a RAG system: **calibrate the threshold per
embedding model against a small labeled answerable/unanswerable set; do not ship a hard-coded
constant, and do not assume a strong embedder's cosines are centered where a weak one's are.**
`recall.eval.calibrate.calibrate()` reproduces these numbers.

## 3. Domain fine-tuning: an honest null result on this corpus

`finetune/train.py` fine-tunes `all-MiniLM-L6-v2` with OnlineContrastiveLoss on `(query, gold-chunk)`
positive / `(query, wrong-chunk)` negative pairs (recipe adapted from a proven production trainer),
then measures retrieval on a **held-out** set of differently-phrased queries:

| model | test MRR | test nDCG@10 |
|---|---|---|
| all-MiniLM-L6-v2 (base) | 1.00 | 1.00 |
| + fine-tuned | 1.00 | 1.00 |
| **Δ** | **+0.00** | **+0.00** |

**Zero lift — and that is the honest, expected outcome here.** The 14-document corpus is highly
separable; a modern small embedder already retrieves the correct chunk for every held-out query,
even when it is paraphrased with different vocabulary. There is no headroom to improve. Manufacturing
a win would have meant evaluating on the *training* queries (memorization) or crippling the base
model on purpose.

To demonstrate a *real* domain-adaptation lift you need a corpus the base model actually struggles
on. We built one — an **opaque-codename corpus** where the concept↔name mapping exists nowhere in
pretraining — and there the same pipeline **lifts held-out MRR by ~79% (0.31 → 0.55, Δ +0.24)**.
Same method, opposite outcome: **fine-tuning's payoff equals the vocabulary gap between the base
model and your corpus** — zero on this rich corpus, large on the jargon one.

**→ Full controlled study (both corpora, method, decision rule): [docs/RAG_TRAINING_STUDY.md](../docs/RAG_TRAINING_STUDY.md).**

The lesson from the production know-how the recipe came from: **embeddings only encode what they
encode — measure honestly, don't force a result.**

## 4. Validity beats similarity: the trust layer kills stale-memory false positives

v0.2 adds a trust layer: every hit returns **confidence + provenance + validity** (verdict:
`ok | superseded | expired | not_yet_valid | low_confidence | invalid_metadata | ambiguous_supersession`), and the result **abstains** when no
valid hit clears the calibrated threshold. The motivating failure: a memory that is *semantically
closest* to the query but **superseded** keeps winning plain vector search forever — the agent
confidently builds on a decision that was reversed. Six validity-sensitive queries (worded
deliberately closer to the *stale* version — the adversarial case) measure it. STR =
superseded-trust rate: how often a stale memory is presented as the answer (lower is better).

| embedder | STR plain search | STR trust layer | successor acc | abstain acc | MRR answerable (plain → trust) |
|---|---|---|---|---|---|
| hashing-64 | **1.00** | **0.00** | 0.25 | 0.00 | 0.737 → 0.737 |
| bge-small (FastEmbed) | **0.83** | **0.00** | 0.75 | 1.00 | 1.000 → 1.000 |

- **Plain search fails exactly as predicted**: on 83–100% of the trust queries the top answer is
  the superseded/expired memory — semantic similarity cannot see supersession. With the trust
  layer the stale memory is *never* presented as trustworthy (STR 0.00 on both embedders), and
  ordinary answerable retrieval is untouched (identical MRR).
- **Successor redirect**: an explicit `supersedes:` edge transfers relevance — when the stale hit
  scored above the threshold, its retrieved successor is promoted even if its own (different)
  wording scores lower. On bge the successor is the top trusted answer in 3/4 cases; the "miss"
  is honest ranking, not stale trust: the successor was verdict-ok but ranked behind *another
  valid, topically-related memory* (strict top-1 metric).
- **Abstention quality is bounded by the embedder — §2's lesson resurfaces.** bge + calibrated
  threshold abstains perfectly on expired/not-yet-valid-only queries (2/2). hashing-64 cannot
  abstain at all (0/2): its answerable/unanswerable cosine regimes overlap, so unrelated
  memories clear any workable threshold. A weak embedder cannot support calibrated abstention,
  at any threshold — same failure mode as gap detection.
- **Limits stated plainly**: the redirect requires the successor to be *retrieved* (it is not
  re-queried); validity metadata is declared by the memory author, not inferred; and the
  calibration comes from a small labeled query set (see §2 for why it must be per-embedder).

Reproduce: `make eval` → the trust table in `RESULTS.md` + `results/trust_effect.png`.

**Timestamps are not a substitute (steelman tested).** The trust table's `STR recency` column
scores the strongest reasonable timestamp heuristic — "among the confidently-relevant hits,
trust the newest", with the stale docs re-synced after their successors, as any living corpus
does constantly. It trusts the stale memory 83–100% of the time, and on bge-small it is *worse
than plain ranking* (1.00 vs 0.83): the tie-break promotes the freshly-touched stale memo
exactly where ranking had preferred the successor. A per-document timestamp cannot see a
two-document relation. Full discussion:
[docs/ENTAILMENT_SUPERSESSION_STUDY.md §3](../docs/ENTAILMENT_SUPERSESSION_STUDY.md).

## 5. Entailment abstention: the near-miss class needs a judge, and the judge needs the threshold

The calibrated threshold (§2) catches *far* gaps; it cannot catch the **near-miss** — a
high-similarity memory that does not answer the query — because the distractor's cosine clears
any threshold by construction. On a held-out 10-query near-miss set (excluded from
calibration), the threshold's false-confident rate is 0.40–1.00 per embedder.

v0.3 adds an optional entailment stage (`recall[entail]`, OFF by default): a QNLI cross-encoder
judges the verdict-ok hits and demotes non-answering ones to `not_entailed`. The decision is at
the judge's own trained boundary — no per-embedder constant to recalibrate, and none was tuned:

| embedder | near-miss FCR: threshold → +entail | gap FCR | false-abstain cost | judge ms (judged calls) |
|---|---|---|---|---|
| hashing-64 | 1.00 → **0.60** | 1.00 → 0.20 | 0.00 → 0.21 | 856 |
| bge-small | 0.80 → **0.50** | 0.00 → 0.00 | 0.00 → 0.07 | 149 |
| voyage-3 | 0.40 → 0.40 | 0.00 → 0.00 | 0.00 → 0.07 | 125 |

Honest reading: the same judge transfers across embedders with zero retuning (the property a
score threshold provably lacks, §2) — but the **judge-alone ablation degrades far-gap detection**
(gap FCR 0.00→0.40): threshold and judge guard *different failure classes* and must be stacked.
The residual near-miss FCR is the judge's own quality bound (a small QNLI model reads
"on-topic" as "answers" when the query asks for an absent detail), and the cost is real:
~0.1–1.0 s of judge time per query (~1.3× to >200× total latency depending on how fast the
embedder underneath is) and one answerable query (a *negation* answer: "we do **not** retry on 4xx")
wrongly rejected on both semantic embedders. §2's law, one layer up: **abstention quality is
bounded by the judge.** Full tables + arms:
[docs/ENTAILMENT_SUPERSESSION_STUDY.md](../docs/ENTAILMENT_SUPERSESSION_STUDY.md) and the
near-miss table in `RESULTS.md`.
