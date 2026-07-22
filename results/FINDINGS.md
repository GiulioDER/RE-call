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

### 2b. Correction: "FCR @calibrated" is an in-sample number, and the fit is one-sided

Two defects in the table above, found by re-deriving it under a held-out protocol.

**The calibrated column could not have been anything but 0.00.** `best_threshold` *minimises
misclassification on the samples it is handed*, and `fcr_at_suggested` then scored it on those
same 5 unanswerable samples. On separable data that is 0.00 by arithmetic, before any data is
collected — it measures the optimiser's objective, not the threshold's ability to generalise.
`calibrate()` now also reports `fcr_heldout` / `false_abstain_heldout`, cross-validated
leave-one-out, and those are the publishable numbers. **The FCR @0.50 column is unaffected** —
0.50 is a constant chosen before seeing the samples, so it was always a genuine measurement, and
it carries the actual finding of this section.

**The fit ignores the unanswerable class entirely.** A candidate below `min(answerable)` costs one
unanswerable error for every sample above it and saves nothing; a candidate above it costs
answerable errors. So the optimum always lands exactly on `min(answerable)`, wherever the
unanswerable samples fall. Two consequences:

- Holding out an unanswerable sample cannot move the threshold, so the cross-validated FCR equals
  the in-sample one. That column was never at risk of memorisation — the reason it reads 0.00 is
  the one above, not overfitting.
- The threshold has **zero margin on the answerable side**. Hold out the lowest answerable sample
  and the refit boundary rises above it, so it abstains: leave-one-out false-abstain is
  `1/n_answerable` (0.07 here) even on perfectly separable data. At runtime this means *any*
  genuine answer scoring below the weakest calibration sample is abstained on. A threshold placed
  in the middle of the gap rather than on its floor would carry margin on both sides.

> **Both defects are now fixed.** The threshold bisects the gap instead of sitting on the lowest
> answerable sample — see §6 for the measurements that drove the change and what it cost.

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
`ok | superseded | expired | not_yet_valid | low_confidence | invalid_metadata`), and the result **abstains** when no
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

## 5. At scale: the headline rate holds — and the coverage it costs becomes visible

§4's superseded-trust rate rests on **6** queries, so its 95% Wilson interval is **[0.00, 0.39]** —
consistent with a working trust layer and with a mediocre one. `recall.eval.synthetic` generates the
same *shape* of corpus at arbitrary size, so both axes scale: queries for interval width, documents
for index pressure. Two arms, because they answer different questions and have very different costs
(bge-small embeds at ~11 chunks/s on the reference CPU, so the large-corpus arm uses the offline
embedder).

**Arm A — interval width** (`bge-small`, 600 chunks, 550 queries, [SCALE.md](scale/SCALE.md)):

| metric | rate | 95% Wilson | n |
|---|---|---|---|
| STR trust | 0.00 | **[0.00, 0.02]** | 250 |
| trust coverage | 0.43 | [0.37, 0.49] | 250 |
| successor accuracy | 0.55 | [0.47, 0.63] | 150 |
| abstention accuracy | 0.92 | [0.85, 0.96] | 100 |

**The headline claim survives a 40× larger query set**: superseded-trust is 0.00 with the interval
tightened from [0.00, 0.39] to [0.00, 0.02], against an STR baseline of 1.00 (plain search returns
the stale memory *every time* on these adversarially-worded queries). That is the strongest
evidence in this document, and it is now bounded rather than asserted.

**But the coverage column changes the reading.** The trust layer returns no `ok` hit at all on
**57%** of validity-sensitive queries, and when it does answer a supersession query it names the
correct successor **55%** of the time. Demotion works; *promotion* is close to a coin flip. The
0.00 is real and is partly bought with silence — precisely the trade the STR column alone cannot
show, which is why coverage is now published beside it. Abstention accuracy (0.92) is genuinely
good: when nothing valid exists, it does say so.

**Arm B — index pressure** (`hashing-64`, 50,600 chunks, [SCALE.md](scale-pressure/SCALE.md)):

| measurement | value |
|---|---|
| recall@5, unfiltered | 1.00 [0.98, 1.00] (n=200) |
| recall@5, `source`-filtered | 1.00 [0.98, 1.00] (n=200) |
| search latency p50 / p95 / p99 | 10.7 / 13.5 / 16.5 ms |
| index throughput | 50,600 chunks in 126 s (~400 chunks/s) |

**A predicted failure that did not reproduce.** A `source`-filtered query pairs a `WHERE` clause
with an HNSW `ORDER BY`, and the graph walk cannot see the predicate — the textbook post-filtering
recall collapse. At 50k chunks, filtering to the single source holding the answer still returned it
every time. The likely reason is that the `source` btree index makes an exact scan cheap enough for
the planner to prefer it over the ANN path at this selectivity, which protects recall at the cost of
the ANN speed-up. Not disproven in general — one corpus size, one selectivity, one embedder — but
not observed here, and reported as such.

**What this still does not cover:** a real-language corpus (the generated text is templated, so
absolute retrieval quality is optimistic), the cloud embedder at scale, and any corpus large enough
to push HNSW past the point where an exact scan is competitive.

## 6. The abstention threshold: measured, and rebuilt

§2b reported that `best_threshold` sat exactly on `min(answerable)` — a one-sided fit with zero
margin. Measuring it on a real host turned that from a fragility into a defect with a number
attached, and the rule has been replaced.

**What the old rule did, measured** (`bge-small`, 5,450 chunks, 4 fresh HNSW builds, fitted on
half the queries and scored on the other half):

| rule | threshold (mean ± sd) | false-abstain | gap FCR | bal. err |
|---|---|---|---|---|
| `min(answerable)` *(old)* | 0.599 ± 0.008 | 0.003 | **0.205** | 0.104 |
| q05 | 0.678 ± 0.085 | 0.005 | 0.065 | 0.035 |
| **midgap q05/q95** *(new)* | 0.656 ± 0.044 | 0.010 | 0.045 | 0.028 |
| Youden J | 0.690 ± 0.061 | 0.013 | 0.015 | 0.014 |
| q20 | 0.899 ± 0.032 | **0.310** | 0.000 | 0.155 |

The old rule let **20.5%** of genuinely unanswerable queries through, because the answerable
distribution has a long lower tail (min 0.601 against p25 0.913) and the boundary sat at the
bottom of it. It also inherited ANN nondeterminism: HNSW builds are not reproducible, so the
identity of the worst sample — and with it the whole operating point — changed on every re-index
(coverage swung 0.40–0.84 on one host; issue #26).

**The shipped rule, verified end to end** on the same host after the change:
threshold **0.728 ± 0.042**, false-abstain **0.015**, gap FCR **0.000**, balanced error
**0.007** — against 0.104 for the rule it replaces.

**How conservative is enough:** only slightly. Moving off the minimum to the middle of the gap
buys the whole improvement for ~1.5% of answerable queries. Pushing further is a bad trade —
a q20 floor drives false-abstain to 0.31 to buy the last points of FCR.

⚠️ **Outlier robustness needs samples.** The floor is a 5th percentile, which cannot exclude
anything below ~20 answerable samples; on the 14-document corpus it still collapses onto the
minimum. Bisecting the gap adds margin at any size, but stability requires a real calibration set.

### What this evaluation still cannot measure

The synthetic corpus was fixed in one respect and remains broken in another.

- **Fixed:** its unanswerable queries were an answerable query plus a nonsense suffix
  (`"...retry budget for ivory-kiln-0000-absent"`), so every other word was shared. Measured with
  bge-small they were **not separable at all** — median top cosine 0.830 against answerable 0.923,
  with **0%** below the weakest answerable query. They are now genuinely off-topic questions
  (median 0.570, 78% below the answerable floor), matching how the 14-document corpus writes them.
- **Still broken:** every generated *document* is the same sentence with a different opaque token
  (`"The cache TTL for granite-harbor-0001 is 669 seconds"`). Hundreds of near-identical documents
  differ only by a string no embedder can interpret, so **successor accuracy and abstention
  accuracy on this corpus measure token discrimination, not the trust layer** — which is why they
  read 0.14 and 0.00 in the latest scale run regardless of threshold. STR, latency and index-scale
  figures are unaffected, because supersession is a declared relation rather than a similarity
  judgement.

Treat the successor/abstain columns from generated corpora as not-yet-measured.

## 7. The real number: paraphrased questions cut retrieval to a third

Every retrieval figure above this section was measured either on a corpus this repo ships, one it
generates, or — for the real corpus — with document **headings** as queries. That last one is
*known-item retrieval*: finding a document you can already name. It scored **recall@5 0.945** and
the README always flagged it as optimistic. This measures by how much.

**110 hand-labelled questions** against the reference corpus (794 memos → 6,491 chunks,
`bge-small`, hybrid dense + sparse, no reranker), phrased the way a person actually asks rather
than in the document's own words. Half fit the threshold, half scored:

| metric | value | 95% Wilson | n |
|---|---|---|---|
| **hit@5** | **0.33** | [0.21, 0.47] | 46 |
| MRR | 0.29 | — | 46 |
| abstention accuracy | 0.89 | [0.57, 0.98] | 9 |
| false-abstain | 0.04 | [0.01, 0.15] | 46 |
| search latency p50 / p95 | 78 / 124 ms | — | — |

**Headings 0.945 → real questions 0.33.** The proxy was hiding roughly two thirds of the
retrieval failures.

### The misses were inspected, not assumed

The runner reports what came back for every miss, because on a corpus of many closely-related
memos a miss can be a *labelling* failure rather than a retrieval one. Of eight sampled:

- **one was mislabelled** — for "what did the risk guard do when it could not read the market
  stress inputs?", the top hit was the market-stress fix and the second was the *fail-closed*
  counterpart of the labelled memo. Both answer it; the label named one file.
- **seven were genuine**, several landing in the right topic family but the wrong document
  (a database-outage question returning a different incident on the same host).

So 0.33 is a mild under-estimate — call it ~0.35–0.40 once labels are widened — and nowhere near
0.945.

### What the abstention layer did

It held up where retrieval did not: **89%** of genuinely unanswerable questions were abstained on,
while only **4%** of answerable ones were wrongly refused. The trust layer is not the bottleneck.

### The most likely lever, tested — and it is not the answer

Re-run with `--rerank`, scoring both arms from the **same index pass** so nothing but the ranking
stage differs (indexing this corpus costs ~800 s, and issue #26 showed an HNSW rebuild is not a
fixed quantity, so two separate runs would have compared two different indexes):

| arm | hit@5 | 95% Wilson | MRR | misses | p50 latency |
|---|---|---|---|---|---|
| hybrid | 0.326 | [0.21, 0.47] | 0.294 | 31 / 46 | **45 ms** |
| hybrid + cross-encoder | 0.391 | [0.26, 0.54] | 0.312 | 28 / 46 | **2,568 ms** |

**Three questions out of 46 changed from miss to hit, for a 57× latency increase.** The intervals
overlap almost entirely; this run cannot distinguish that gain from noise. §1's result — reranking
rescuing a weak embedder — did not transfer here.

And the shape of the failure is the useful part. **A reranker can only reorder what fusion already
retrieved.** If it converts 3 of 31 misses, then for roughly 28 of them the right document was never
in the candidate window at all. So the bottleneck is **candidate recall, not ranking**: the
retrieval stage is not returning the right memo to be re-ranked.

That redirects the next experiment away from ranking and toward the pool itself — a larger `k`
before fusion, a stronger embedder, or the chunking (800-character windows over dense, reference-
heavy memos may be splitting the answer away from the words the question uses). Each is testable
with this same harness; none has been run, and none should be claimed until it has.

### Candidate pool and chunking: also not the lever

With ranking ruled out, the two remaining candidate-recall knobs were swept — `candidate_k` (the
pool each leg contributes before fusion) and the chunk size — each chunk size costing its own
index pass. Same 46 held-out questions throughout, `candidate_k=100`:

| chunk chars | chunks | hit@5 | hit@10 | hit@20 | hit@50 | MRR |
|---|---|---|---|---|---|---|
| 400 | 13,262 | 0.326 | 0.370 | 0.413 | 0.478 | 0.245 |
| **800 (shipped)** | 6,491 | **0.348** | **0.435** | 0.457 | 0.500 | **0.311** |
| 1600 | 3,239 | 0.348 | 0.413 | 0.478 | 0.500 | 0.271 |

Three readings, and none of them is a lever:

1. **Chunk size does not move hit@5.** 0.326 / 0.348 / 0.348 sit inside each other's intervals.
   The shipped 800 is the best of the three on MRR, so the default was already right.
2. **A bigger candidate pool buys nothing at the top.** Raising `candidate_k` from 20 to 100 leaves
   hit@5 and hit@10 unchanged and adds only ~0.04 at hit@20 — it surfaces documents that then rank
   low anyway.
3. **hit@50 plateaus at ~0.48–0.50 in every configuration.** For half the questions the right
   document is nowhere in the top *fifty*, whatever the chunking or the pool.

That last line is the finding. It is a **hard recall ceiling**, and it explains why the two things
tried before it did so little: a reranker can only reorder what was retrieved, and a bigger pool
can only add what the index can match. Both were working on the half of the problem that was
already solvable.

### Confirmed: it was the embedder

The prediction was tested by swapping only the embedder — same corpus, same 46 held-out
questions, same pipeline:

| embedder | hit@5 | 95% Wilson | MRR | index | search p50 |
|---|---|---|---|---|---|
| bge-small (local, 384d) | 0.348 | [0.23, 0.49] | 0.311 | 696 s | 45 ms |
| **voyage-3 (cloud)** | **0.630** | **[0.49, 0.76]** | **0.503** | 224 s | 246 ms |

**hit@5 nearly doubles and MRR rises 62%.** The intervals barely touch — bge-small's upper bound
(0.49) is voyage-3's lower bound — so unlike every previous attempt this is a difference the
sample can actually resolve.

Set against the three eliminated levers, on the same questions:

| change | Δ hit@5 |
|---|---|
| cross-encoder rerank | +0.065 (within noise, 57× latency) |
| candidate pool 20 → 100 | +0.000 |
| chunk size 400 / 800 / 1600 | +0.000 |
| **embedder → voyage-3** | **+0.282** |

The pipeline was never the problem. Three knobs were turned first and none of them mattered,
which is the useful part of the result: it is evidence that this corpus was hitting a
representation ceiling, not a tuning one, and that no amount of retrieval engineering was going
to move it.

**Cost of the fix:** search latency 45 ms → 246 ms (a network round trip per query), an API
dependency, and the corpus leaving your infrastructure to be embedded. Indexing is *faster*
(224 s vs 696 s) because a batched API beats local CPU. Abstention is unaffected — accuracy 0.89
either way, false-abstain 0.065 vs 0.043.

### What is left, and why the repo's own §3 predicts it

Ranking, pool size and chunking are eliminated. What remains is the **representation**: `bge-small`
cannot connect a paraphrased question to these documents, because the vocabulary that identifies
them — project codenames, venue and bot names, internal shorthand — appears nowhere in its
pretraining.

That is precisely the condition §3 measured. On a rich corpus, fine-tuning bought **+0.00**; on an
opaque-codename corpus where the concept↔name mapping exists nowhere in pretraining, the same
pipeline lifted held-out MRR **0.31 → 0.55**. This corpus is the second kind, and its measured MRR
of **0.31** is, to the decimal, where that study started.

Two ways to act on that: a stronger embedder, or domain fine-tuning. **The stronger embedder was
run and settled it** (above). Fine-tuning `bge-small` on the same 46 training queries was started
and then **abandoned unfinished** — not because it failed, but on operational grounds: on the
reference host it consumed 629% CPU across 63 threads beside live systems, and was stopped at
44/96 steps. `nice` lowers scheduling priority but does not cap thread count.

So the honest status is: **fine-tuning remains untested here.** It would answer a narrower question
than the one already answered — whether a *local* model can close the gap that voyage-3 closes —
and that is worth knowing, but it is not what was blocking the result. The only datum recovered
from the attempt is the trainer's own baseline, `test MRR 0.292`, which independently corroborates
the 0.311 this harness measured file-level on the same embedder.

### The most likely lever, as first predicted (superseded by the runs above)

This run used dense + sparse fusion with **no cross-encoder**. §1 of this document measures
reranking lifting MRR from 0.63 to 1.00 on a weak embedder and finding it redundant on an easy
corpus — and this corpus is not easy: hundreds of memos share a vocabulary, which is precisely the
regime where fusion ranks the right document into the window but not to the top. Re-running this
harness with `CrossEncoderReranker`, and with a stronger embedder, is the obvious next experiment.
Until it is run, the honest statement is that **retrieval on a real jargon-dense corpus is the
weakest measured part of this system**, and that it was invisible until the questions stopped
being headings.

The labelled set is the corpus owner's private data and is not published; only these aggregates
and the runner (`python -m recall.eval.labelled`) are.


## 8. Replication on a second corpus: the cloud embedder's win is corpus-specific

§7 measured voyage-3 nearly doubling hit@5 over bge-small and concluded the ceiling was the
representation. That rested on **one** corpus, and a private one. This replicates it on an
independent, fully public corpus — the **732 Python PEPs**: dense technical jargon, many authors,
decades of drift, and heavy near-neighbour pressure (seven "Python X.Y Release Schedule"
documents, multiple steering-council elections, whole families of typing and packaging proposals).
110 hand-labelled questions ship in this repo, phrased away from every title.

| hit@5 | bge-small (local) | voyage-3 (cloud) | Δ |
|---|---|---|---|
| private memory corpus, 794 docs | 0.348 [0.23, 0.49] | **0.630** [0.49, 0.76] | **+0.282** |
| **PEPs, 746 docs (public)** | **0.705** [0.56, 0.82] | 0.727 [0.58, 0.84] | **+0.022** |

MRR: 0.311 → 0.503 on the memory corpus; 0.483 → 0.629 on the PEPs.

**The single-corpus conclusion was too broad.** Three things the replication shows that one corpus
could not:

1. **The pipeline is not the problem.** On ordinary technical prose the *free local* embedder
   reaches hit@5 0.705. Nothing about hybrid retrieval, chunking or the trust layer caps
   performance at 0.35 — §7's number was a property of that corpus, not of this software.
2. **The cloud embedder's advantage is corpus-dependent, not general.** It is worth **+0.28** on the
   idiosyncratic corpus and **+0.02** on the PEPs — the latter comfortably inside the noise, for
   ~5× the query latency, an API dependency and sending your documents to a third party.
3. **The right rule is therefore conditional**: pay for a cloud embedder when your corpus
   vocabulary is *unusual* — internal codenames, project shorthand, identifiers absent from any
   pretraining set. On ordinary technical English, don't; it buys nothing measurable here.

The trust layer holds on both: abstention accuracy **1.00** (11/11) on the PEPs for both embedders,
false-abstain 0.02–0.05. It was never the bottleneck on either corpus.

Reproduce this one end to end — corpus, questions and ground truth are all public:

```bash
git clone --depth 1 https://github.com/python/peps
python -m recall.eval.labelled --corpus peps/peps     --questions recall/eval/peps_questions.json --glob '**/*.rst'
```
