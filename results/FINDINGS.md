# RE-call — evaluation findings

**What each result means.** The numbers themselves — every table, with the command that reproduces
it and its evidence tier — live in [`RESULTS.md`](RESULTS.md). This file does not reprint them; it
says what they support and where each one stops. Section numbers correspond.

A note on what you will *not* find here: the running commentary of corrections this document used
to carry. Where a published claim did not survive re-measurement, the current statement is given
and the withdrawal noted in one line. The full history is in
[`CHANGELOG.md`](../CHANGELOG.md) and the git log.

## What this document establishes

Five things, in descending order of how well-evidenced they are. Everything else is support.

1. **Declared supersession beats similarity, and timestamps cannot substitute for it.**
   Superseded-trust **0.00 [0.00, 0.02]** at n=250 and full coverage, against a plain-search
   baseline of 1.00. The steelmanned timestamp heuristic still trusts the stale memory 83–100% of
   the time. → §4, §5b
2. **Abstention has a hard, measured boundary: far gaps yes, near-misses no.** Accuracy 1.00 (PEPs)
   and 0.89 (real corpus) where the unanswerable class is off-topic; on near-miss classes it fails,
   and **six** candidate signals were measured to establish that no cheap signal fixes it — the best
   one's interval tops out at 0.826 against a ~0.90 bar. Corroborated independently on two
   benchmarks. → §5, §9b, §10b, §10c
3. **Retrieval quality is a property of your corpus, not of this pipeline.** The same stack scores
   0.348 on an idiosyncratic private corpus and 0.705 on public technical prose. A cloud embedder is
   worth +0.282 on the first and +0.022 on the second — but **on 17 held-out corpora it wins 16
   times** (median +0.059, sign-test p = 0.00027), and what predicts the gap is **corpus size**, not
   unusual vocabulary. §8's "buys nothing on ordinary English" was drawn from a 746-document corpus
   and is **restated**. → §7, §8, [gap study](gap/FINDINGS-embedder-gap.md)
4. **On the standard benchmark, the substrate is competitive and the adversarial axis is unsolved.**
   LOCOMO hit@5 **0.671**, hit@20 **0.855** — and **0 of 446** adversarial questions abstained on out
   of the box, an axis no published LOCOMO result scores at all. → §9a, §9b
5. **Against Mem0, paired on identical questions, generator and judge: more accurate at $0.** Five
   configurations, all surviving Holm–Bonferroni — with the lead disclosed as reader-tier
   conditional, and reversing on a strong non-OpenAI reader. → §9d

The three genuinely negative results (2, and the eliminations inside 3) are the ones that took the
most work, and they are why 1 and 4 are worth anything.

## 1. Hybrid + rerank helps — where the embedder isn't already saturated

On the weak, non-semantic hashing embedder, quality climbs monotonically as the sparse leg and then
the cross-encoder are added: MRR **0.63 → 0.96 → 1.00** ([`RESULTS.md` §1](RESULTS.md)). On the
strong bge-small embedder, dense retrieval already scores nDCG@10 0.97 and the hybrid arm saturates
this corpus at 1.00, so the reranker has nothing left to gain.

The instructive detail: the hashing embedder's *hybrid* arm reaches the same 0.96/0.97 as
bge-small's *dense* arm. With the lexical leg working, the sparse signal substitutes for much of
what a real embedder buys on this corpus. **Hybrid + rerank buys the most on weaker embedders or
harder corpora; on an easy corpus with a strong embedder it is redundant** — and an eval worth
trusting has to be able to show that, not just a win.

_The hybrid row was published as 0.74 / 0.80 before
[#81](https://github.com/GiulioDER/RE-call/issues/81): the sparse leg ANDed every query term, so it
fired only when one chunk contained all of them. The direction of this finding was always right;
the magnitude was understated. `dense only` is unchanged, as it must be._

## 2. The honest negative result: a fixed gap threshold does NOT transfer across embedders

The gap guard fires when the best dense cosine for a query falls below a threshold (default 0.50).
We measured the top-cosine distribution for answerable vs. unanswerable queries per embedder:

| embedder | answerable cos (min–max) | unanswerable cos (min–max) | separable? | good threshold | FCR @0.50 |
|---|---|---|---|---|---|
| hashing-64 | 0.30 – 0.68 | 0.35 – 0.53 | no (overlap) | — | 0.20\* |
| bge-small (FastEmbed) | 0.70 – 0.90 | 0.51 – 0.64 | yes | ~0.70 | **1.00** |
| voyage-3\† | 0.53 – 0.70 | 0.09 – 0.32 | yes | ~0.50 | **0.00** |

_This table once carried an `FCR @calibrated` column reading 0.00 for both strong embedders. It was
fitted and scored on the same samples, so on separable data it is 0.00 by arithmetic before any data
is collected — **withdrawn**, not merely caveated (§2b). `FCR @0.50` is unaffected: 0.50 is a
constant chosen before seeing the samples, and it carries this section's finding on its own._

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
  and the false-confident rate on unanswerable queries is 1.00. The distributions **are** cleanly
  separable — unanswerable tops out at 0.64, answerable starts at 0.70 — so a threshold in that gap
  is the fix. How well it then generalises is measured out-of-sample in §6, not here.
- **Gap-detection quality is bounded by the embedder.** hashing-64's answerable and unanswerable
  distributions overlap, so no single threshold separates them: a weak, non-semantic embedder
  cannot support reliable gap detection at all, at any threshold. (0.50 scores FCR 0.20 while also
  wrongly flagging *answerable* queries whose cosines sit below 0.50.)

Takeaway for anyone building gap/abstention into a RAG system: **calibrate the threshold per
embedding model against a small labeled answerable/unanswerable set; do not ship a hard-coded
constant, and do not assume a strong embedder's cosines are centered where a weak one's are.**
`recall.eval.calibrate.calibrate()` reproduces these numbers.

### 2b. Why the old fitting rule was replaced

Re-deriving the table under a held-out protocol exposed two defects in `best_threshold`, both now
fixed:

- **It was scored on its own training samples.** `fcr_at_suggested` evaluated the fitted threshold
  on the same 5 unanswerable samples it was fitted to — the optimiser's objective, not
  generalisation. `calibrate()` now also reports cross-validated `fcr_heldout` /
  `false_abstain_heldout`, and those are the publishable numbers.
- **The fit had zero margin on the answerable side.** A candidate below `min(answerable)` costs an
  unanswerable error and saves nothing, so the optimum always landed exactly on `min(answerable)`
  wherever the unanswerable samples fell. At runtime that abstains on *any* genuine answer scoring
  below the weakest calibration sample; leave-one-out false-abstain is `1/n_answerable` even on
  perfectly separable data.

The threshold now bisects the gap instead of sitting on its floor. §6 has the measurements that
drove the change and what it cost.

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

The trust layer returns **confidence + provenance + validity** with every hit (verdict:
`ok | superseded | expired | not_yet_valid | low_confidence | invalid_metadata`) and **abstains**
when no valid hit clears the calibrated threshold. The motivating failure: a memory that is
*semantically closest* to the query but **superseded** wins plain vector search forever, and the
agent confidently builds on a decision that was reversed. Six validity-sensitive queries — worded
deliberately closer to the *stale* version, the adversarial case — measure it
([`RESULTS.md` §2](RESULTS.md)).

- **Plain search fails exactly as predicted.** On 83–100% of the trust queries the top answer is the
  superseded or expired memory: semantic similarity cannot see supersession. With the trust layer
  the stale memory is *never* served as trustworthy — **STR 0.00** on both embedders, at coverage
  0.67–0.83, so the zero is not bought by blanket abstention. Read STR and coverage together; §5b
  bounds the same claim at n=250.
- **Successor redirect works, and requires retrieval.** An explicit `supersedes:` edge transfers
  relevance: when the stale hit clears the threshold, its successor is promoted even where its own
  wording scores lower. On bge the successor is the top trusted answer in 3/4 cases; the miss is
  honest ranking, not stale trust — the successor was verdict-ok but ranked behind another valid,
  topically-related memory under a strict top-1 metric.
- **Ordinary retrieval is not free of cost, and that is embedder-dependent.** hashing-64 pays
  **0.964 → 0.804** answerable MRR under the trust layer — it now has genuinely-retrieved material
  to demote — while bge-small stays 1.000 → 1.000. An earlier revision claimed answerable retrieval
  was untouched; that held only while the sparse leg was broken, and is **withdrawn**.
- **The n=2 abstain column is not evidence in either direction.** Its ordering across the two
  embedders *flipped* between runs, which is what a two-sample column does. The
  embedder-bound-abstention claim stands, but on the measurements sized to carry it: §2's
  separability analysis, §5b's n=100 arm, and §9–§10.

**Timestamps are not a substitute, steelmanned.** `STR recency` scores the strongest reasonable
timestamp heuristic — "among the confidently-relevant hits, trust the newest" — with the stale docs
re-synced *after* their successors, as any living corpus does constantly. It trusts the stale memory
83–100% of the time, and on bge-small it is **worse than plain ranking** (1.00 vs 0.83): the
tie-break promotes the freshly-touched stale memo exactly where ranking had preferred the successor.
A per-document timestamp cannot see a two-document relation. Full discussion:
[docs/ENTAILMENT_SUPERSESSION_STUDY.md §3](../docs/ENTAILMENT_SUPERSESSION_STUDY.md).

Remaining limits: the redirect requires the successor to be *retrieved* (it is not re-queried),
validity metadata is declared by the author rather than inferred, and calibration comes from a small
labelled query set — see §2 for why it must be per-embedder.

## 5. Entailment abstention: the near-miss class needs a judge, and the judge needs the threshold

The calibrated threshold (§2) catches *far* gaps. It cannot catch the **near-miss** — a
high-similarity memory that does not answer the query — because the distractor's cosine clears any
threshold by construction. On a held-out 10-query near-miss set the threshold's false-confident rate
is 0.40–1.00 per embedder.

The optional entailment stage (`recall[entail]`, **off** by default) puts a QNLI cross-encoder over
the verdict-ok hits and demotes non-answering ones to `not_entailed`, at the judge's own trained
boundary — no per-embedder constant, and none was tuned. Three arms in
[`RESULTS.md` §3](RESULTS.md). What they establish:

- **The same judge transfers across embedders with zero retuning** — precisely the property a score
  threshold provably lacks (§2). It roughly halves near-miss false-confidence.
- **The judge alone degrades far-gap detection** (gap FCR 0.00 → 0.40 in the ablation). Threshold
  and judge guard *different failure classes*; they stack, and neither replaces the other.
- **The cost is real.** ~0.1–1.0 s of judge time per query — between ~1.3× and >200× total latency
  depending on how fast the embedder underneath is — and one answerable query wrongly rejected on
  both semantic embedders (a *negation* answer: "we do **not** retry on 4xx").

The residual near-miss FCR is the judge's own quality bound: a small QNLI model reads "on-topic" as
"answers" when the query asks for an absent detail. §2's law, one layer up — **abstention quality is
bounded by the judge**, and §10b is where that bound stops being tolerable. Full arms:
[docs/ENTAILMENT_SUPERSESSION_STUDY.md](../docs/ENTAILMENT_SUPERSESSION_STUDY.md).

_The one cloud measurement on this class (voyage-3, near-miss FCR 0.40 → 0.40) comes from the v0.3
run on an older query set and is not re-runnable key-free; it is reported in `RESULTS.md` §3 as
such. The local rows have since been regenerated post-#81 with an added `entail-only` arm, same
direction, per-run numbers differing within the noise a 10-query set carries._


## 5b. At scale: the headline rate holds at full coverage — and the abstain arm does not 📦

§4's superseded-trust rate rests on **6** queries, so its 95% Wilson interval is **[0.00, 0.39]** —
consistent with a working trust layer and with a mediocre one. `recall.eval.synthetic` generates the
same *shape* of corpus at arbitrary size, so both axes scale: queries for interval width, documents
for index pressure.

**Arm A — interval width** (`bge-small`, 600 chunks, 550 queries, [SCALE.md](scale/SCALE.md)):

| metric | rate | 95% Wilson | n |
|---|---|---|---|
| STR trust | 0.00 | **[0.00, 0.02]** | 250 |
| trust coverage | 1.00 | [0.98, 1.00] | 250 |
| successor accuracy | 0.14 | [0.09, 0.20] | 150 |
| abstention accuracy | 0.00 | [0.00, 0.04] | 100 |

**The headline claim survives a 40× larger query set**: superseded-trust 0.00, the interval
tightened from [0.00, 0.39] to [0.00, 0.02], against an STR baseline of 1.00 — plain search returns
the stale memory *every time* on these adversarially-worded queries. That is the strongest evidence
in this document, and it is bounded rather than asserted.

**Coverage 1.00 rules out the degenerate reading of STR** — a system that returns nothing scores a
perfect 0.00 — but it is not independent evidence, because `coverage` is the exact complement of
`abstained`: the harness sets the flag to `bool(ok_keys)` and `trusted_search` sets
`abstained = not ok` from the same list. Coverage 1.00 over n=250 *means* the layer abstained zero
times, which is the same fact the abstention row reports as 0.00 over the 100 queries where
abstaining was correct. One number read twice, once favourably. So the abstain arm did not go
unmeasured — **it was measured and it failed**, on all 100 queries.

**Both weak columns are generator artifacts, and neither should be cited from this arm.** Successor
accuracy (0.14) and abstention accuracy (0.00) are dominated by the corpus generator (§6): every
document is the same sentence with a different opaque token, so choosing between them asks an
embedder to discriminate meaningless strings. That explains the numbers; it does not convert a
measured failure into an absent measurement. STR, coverage, latency and index scale are unaffected,
because supersession is a **declared relation**, not a similarity judgement.

**Arm B — index pressure** (`hashing-64`, 50,600 chunks, [SCALE.md](scale-pressure/SCALE.md)). Run
at the *same seed* — same corpus, same 550 queries, same embedder — three times:

| run | STR baseline | STR trust | trust coverage | abstain acc | p50 / p95 / p99 (ms) | index |
|---|---|---|---|---|---|---|
| 1 | 0.92 | **0.00** | 0.01 | 0.99 | 67.2 / 196.6 / 353.9 | 221.5 s |
| 2 | 0.46 | **0.00** | 0.14 | 0.86 | 5.5 / 9.7 / 10.8 | 172.6 s |
| 3 | 0.82 | **0.00** | 0.14 | 0.86 | 18.3 / 25.0 / 29.7 | 130.7 s |

_Only run 3 has a retained artifact (`scale-pressure/SCALE.md`); runs 1 and 2 were overwritten by
the re-runs. They are kept because they weaken this table's own numbers rather than support them._

**STR trust is 0.00 in all three runs** — the claim the arm exists for. Everything else swings. The
seed fixes what this project controls; it does not fix the HNSW build, whose randomness pgvector
owns. On `hashing-64` that dominates: a 64-dimension hashing embedder puts almost no signal in the
vector, so the graph's shape decides what comes back and the STR *baseline* swings **0.46–0.92** on
identical inputs. Latency and throughput additionally share a contended developer machine, so they
bound nothing — order-of-magnitude only.

**The filtered-recall arm is pinned by construction and could not have fallen.** It scores the
filtered query through the *hybrid* retriever, whose sparse leg is an exact
`tsv @@ websearch_to_tsquery` scan — filter-aware and index-independent — and every generated
answerable document is a single chunk, so `source = ...` selects exactly one row. Degrade the ANN
path arbitrarily and the number stays 1.0000. It measures that the row is *findable*, not that HNSW
found it. A metric that cannot fail is not evidence, and an earlier version of this section read it
as evidence that post-filtering recall collapse "did not reproduce".

Measured properly — directly against `query_dense`, 20,000 rows, a filter matching 10% — pgvector's
defaults give **recall@10 ≈0.36–0.43 with 40/40 queries truncated** below the requested `k`.
`hnsw.ef_search=200` + `hnsw.iterative_scan=relaxed_order` on the filtered path take truncation to
**0/40** and recall to **0.88–0.94**, pinned by `tests/test_hnsw_filtered_recall.py`. Two
measurements were run in [#57](https://github.com/GiulioDER/RE-call/pull/57); they agree on
truncation and **disagree on recall**, so both are published:

| corpus | recall@10 untuned → tuned | truncated untuned → tuned |
|---|---|---|
| the test fixture (20,000 rows, 10% selectivity, analyzed) | 0.36–0.43 → 0.88–0.94 | 40/40 → 0/40 |
| built the way a real multi-file index run builds one (10 batched upserts, 30 queries) | **0.523 → 0.483** | 29/30 → 0/30 |

Untuned returns *few but accurate* hits; `relaxed_order` fills to `k` with approximate ones. **The
supportable claim is the narrow one: filtered dense search now returns `k` results when `k` exist.**
That is a correctness fix, not a quality one.

> **What actually drives this was itself misdiagnosed, and the correction is worth more than the
> numbers.** The collapse was attributed to pgvector building a less well-connected graph when rows
> arrive across several committed transactions. It is a **statistics race**: an *unanalyzed* table
> takes an exact `Seq Scan + Sort` plan, never consults the HNSW index, and therefore reports recall
> **1.0000 under any `ef_search` at all**. Measured both ways with an ANALYZE in place, a single
> 20,000-row single-transaction upsert reproduces the pathology just as hard. What batching did was
> commit rows mid-build, letting an autovacuum worker analyze the table first — winning the race,
> not shaping the graph. The fixture no longer retries until the pathology appears (that cap was
> compensating for exactly the variance ANALYZE removes) and the test now asserts the index is
> genuinely walked before measuring, so it cannot pass **vacuously** by reporting a fix that was
> never exercised. This also gives the second row above a candidate explanation — a batched build is
> the one most likely to have been analyzed — which is stated as a hypothesis, not a result.
> → [#98](https://github.com/GiulioDER/RE-call/pull/98)

**Not covered by either arm:** a real-language corpus (the generated text is templated, so absolute
retrieval quality is optimistic), the cloud embedder at scale, and a filtered-recall arm that
isolates the ANN path end to end.


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

**How conservative is enough:** only slightly. Within the sweep above, moving off the minimum to
the middle of the gap takes gap FCR **0.205 → 0.045** and costs an additional **0.7%** of
answerable queries (false-abstain 0.003 → 0.010). Pushing further is a bad trade — a q20 floor
drives false-abstain to 0.31 to buy the last points of FCR.

Two figures in this section are **not** comparable, and mixing them is the error this section
previously made. The 0.7% above is a within-sweep delta, fitted and scored on split halves. The
shipped rule's **0.015** false-abstain is an end-to-end measurement on a different protocol
(whole query set, fresh builds, after the change) — it is not the price of moving the threshold,
and subtracting it from anything here is meaningless. Likewise "the whole improvement" would be
wrong: midgap reaches 0.045 gap FCR, while Youden J reaches 0.015 and q20 reaches 0.000 in the
same table. Midgap buys most of the available FCR at a small fraction of the false-abstain cost,
which is the actual claim.

**Limitation, by design — outlier robustness needs samples.** The floor is a 5th percentile,
which cannot exclude anything below ~20 answerable samples; on the 14-document corpus it still
collapses onto the minimum. Bisecting the gap adds margin at any size, but stability requires a
real calibration set. This is a permanent property of a percentile rule, not a pending fix —
`Calibration.certified` (§10d) refuses to certify below 20 samples per class for exactly this
reason.

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

## 7. The real number: paraphrased questions collapse retrieval 🔒

The reference corpus is private, so this is the **weakest evidence tier** in this document —
aggregates only, no artifact, not independently checkable. §8 replicates the conclusion on a fully
public corpus; that is the one to trust.

**Current numbers** (2026-07-25, both retrieval fixes live — #81's sparse leg and the
`hnsw.ef_search` widening). 46 held-out questions, `bge-small`, `--candidate-k 20`:

| arm | hit@5 |
|---|---|
| dense | 0.326 |
| sparse | 0.348 |
| **hybrid (shipped)** | **0.457** |
| hybrid + cross-encoder | 0.435 |

A working lexical leg is worth **~+0.13 over dense alone** here. The historical figures below were
measured pre-fix on a smaller corpus (794 files / 6,491 chunks against today's 824 / 6,800), so the
two sets are **not** a clean before/after and are never differenced.

### The finding: headings were hiding two thirds of the failures

Every retrieval figure before this section was measured on a corpus this repo ships or generates —
or, for the real corpus, with document **headings** as queries. That is *known-item retrieval*:
finding a document you can already name. It scored **recall@5 0.945**.

Re-asked as **110 hand-labelled questions** phrased the way a person actually asks (half fit the
threshold, half score), the same corpus and pipeline scored:

| metric | value | 95% Wilson | n |
|---|---|---|---|
| **hit@5** | **0.33** | [0.21, 0.47] | 46 |
| MRR | 0.29 | — | 46 |
| abstention accuracy | 0.89 | [0.57, 0.98] | 9 |
| false-abstain | 0.04 | [0.01, 0.15] | 46 |

**0.945 → 0.33.** Eight sampled misses were inspected rather than assumed: one was a *labelling*
error (two memos answer the question; the label named one file) and seven were genuine, several
landing in the right topic family but the wrong document. So 0.33 is a mild under-estimate — call
it ~0.35–0.40 once labels are widened — and nowhere near 0.945. The abstention layer was never the
bottleneck: 89% of unanswerable questions correctly refused, 4% of answerable ones wrongly refused.

### Four levers tested; one moved it

Each on the **same** 46 held-out questions, scored from one index pass wherever two arms are
compared:

| lever | Δ hit@5 | verdict |
|---|---|---|
| cross-encoder rerank | +0.065 | n.s.; **57× latency** (45 ms → 2,568 ms) |
| chunk size 400 / 800 / 1600 | +0.000 | 0.326 / 0.348 / 0.348 — the shipped 800 was already best on MRR |
| candidate pool 20 → 100 | −0.065 fused, +0.022 reranked | n.s. at n=46 — and see below |
| **embedder → voyage-3** | **+0.282** | 0.348 → **0.630**; the one difference this sample can resolve |

The rerank result is the informative failure. **A reranker can only reorder what fusion already
retrieved** — it converted 3 of 31 misses, so for roughly 28 of them the right document was never
in the candidate window at all. The bottleneck is candidate recall, not ranking.

**The pool null is retracted and stays retracted.** That experiment could not have detected a pool
effect, for two independent reasons: `hnsw.ef_search` capped the dense leg at 40, so a pool of 100
never existed; *and* RRF scores `dense[r]` and `sparse[r]` identically, so a fused top-5 reads only
about three ranks into each leg whatever the pool is. Re-measured with both addressed (paired exact
McNemar, n=46), the two point estimates move in **opposite** directions exactly as that mechanism
predicts — fused −0.065, reranked +0.022 — and neither is significant. Where the same comparison
has the power to resolve itself, it does: on **FinanceBench** (n=150, 72,151 chunks,
`voyage-finance-2`) dense-only + reranker went **0.393 → 0.527** (p<0.001) when the pool grew
40 → 100 with `ef_search` corrected. So *"a bigger pool buys nothing"* is **not** a safe general
claim; on this corpus it remains unmeasured at usable power.

`hit@50` plateaus at ~0.48–0.50 in every local configuration — but with the dense leg capped at 40,
no run here ever offered a true top *fifty*. The ceiling's **shape** stands; its **level** was
measured through a truncated leg, and "the right document is nowhere in the top fifty" is stronger
than the data supports.

### What is left: the representation

Ranking and chunking are eliminated; **pool size is not**. What remains is the representation, and
it is evidenced directly by the +0.282 rather than by elimination: `bge-small` cannot connect a
paraphrased question to these documents, because the vocabulary that identifies them — project
codenames, venue and bot names, internal shorthand — appears nowhere in its pretraining.

That is precisely the condition §3 measured: **+0.00** on a rich corpus, **0.31 → 0.55** held-out
MRR on an opaque-codename one. This corpus's measured MRR of **0.31** is, to the decimal, where
that study started.

**Cost of the embedder fix:** search latency 45 ms → 246 ms (a network round trip per query), an
API dependency, and the corpus leaving your infrastructure to be embedded. Indexing is *faster*
(224 s vs 696 s — a batched API beats local CPU). Abstention is unaffected: accuracy 0.89 either
way.

**Fine-tuning the local model remains untested here.** It would answer the narrower question of
whether a *local* model can close the gap voyage-3 closes. The run was started and abandoned on
operational grounds — 629% CPU across 63 threads beside live systems, stopped at 44/96 steps
(`nice` lowers priority but does not cap thread count). The only datum recovered is the trainer's
own baseline, `test MRR 0.292`, which independently corroborates the 0.311 this harness measured
on the same embedder.

The labelled set is the corpus owner's private data and is not published; only these aggregates and
the runner (`python -m recall.eval.labelled`) are.


## 8. Replication on a second corpus: the cloud embedder's win is corpus-specific

§7 measured voyage-3 nearly doubling hit@5 over bge-small and concluded the ceiling was the
representation. That rested on **one** corpus, and a private one. This replicates it on an
independent, fully public corpus — the **public Python PEP corpus** (746 files matched by
`**/*.rst`, which is what the runner counts and what the table below reports): dense technical
jargon, many authors,
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

> ### ⚠️ RESTATED 2026-07-27 — point 3 came from one small corpus and does not hold
>
> Point 3 says a cloud embedder "buys nothing measurable" on ordinary technical English. That was
> measured on **one corpus of 746 documents**. Extended to **17 held-out BEIR / CQADupStack corpora**
> — none of which generated the hypothesis — it does not survive:
>
> | | hybrid | dense (embedder isolated) |
> |---|---|---|
> | corpora where voyage-3 beats bge-small | **16 / 17** | 15 / 17 |
> | median gap | **+0.059** | **+0.105** |
>
> Sign test **p = 0.00027**; bootstrap 95% CI on the mean gap **[+0.038, +0.068]**.
>
> **What moved was corpus size, not subject matter.** Median gap is **+0.013** below 10 000 documents
> and **+0.062** at 17 000+; gap against document count is Spearman **+0.509**, still **+0.436** with
> the local model's own score partialled out. The PEP corpus at 746 documents is smaller than
> anything in that study, and its +0.022 sits exactly where the small-corpus regime predicts —
> `nfcorpus` (3 633 docs) gives **+0.019**, `scifact` (5 183) gives **+0.013**.
>
> So the §8 *measurement* stands and its *scope* was wrong. The honest rule:
>
> > A cloud embedder buys little on a corpus of a few hundred documents and roughly **+0.06 hit@5**
> > at twenty thousand. **Size** is what moved here — not whether the vocabulary is unusual.
>
> **§7's proposed vocabulary mechanism also fails.** An out-of-vocabulary rate computed against
> bge-small's own tokenizer does *not* predict the gap across those 17 corpora (partial r **+0.265**,
> Holm-adjusted **p = 0.65**), and the null is clean — `oov_rate` correlates **−0.015** with corpus
> size, so nothing is hiding inside it. No corpus statistic tested beat simply measuring the local
> embedder, whose score alone correlates **−0.512** with the gap.
>
> Preregistered before any gap was measured; both corpora above are excluded there as the ones that
> generated the hypothesis. Per-corpus table, confounds and limits:
> [`results/gap/FINDINGS-embedder-gap.md`](gap/FINDINGS-embedder-gap.md).

The trust layer holds on both: abstention accuracy **1.00** (11/11) on the PEPs for both embedders,
false-abstain 0.02–0.05. It was never the bottleneck on either corpus.

Reproduce this one end to end — corpus, questions and ground truth are all public:

```bash
git clone --depth 1 https://github.com/python/peps
python -m recall.eval.labelled --corpus peps/peps     --questions recall/eval/peps_questions.json --glob '**/*.rst'
```

## 9. LOCOMO: the standard benchmark, and the one axis nobody else scores

Every number above is measured on this repo's own corpora, plus the public PEP replication. That was
the standing caveat the README carried: *nothing here is comparable to a published memory-benchmark
result.* This section closes that gap by running **LOCOMO** — the long-term-conversational-memory
benchmark Mem0 and Zep report — against this library.

**What is measured, and what is deliberately not.** LOCOMO's headline metric is LLM-as-a-Judge (J):
a model reads the retrieved context, writes an answer, a judge grades it (Mem0 J≈66.9, Zep J≈66.0;
arXiv:2504.19413 Table 2). **This repo does not produce a J score and none of the numbers below
belong in a column beside one** — RE-call has no generator in its path; it is the retrieval
substrate *under* a system like that. What it can measure exactly is the part it owns, and it
measures it with no judge and therefore no judge variance:

- **Retrieval (categories 1–4).** LOCOMO annotates every answerable question with the dialog turns
  that contain the answer (`evidence: ["D1:3"]`) and every turn carries that id, so "was the
  evidence turn retrieved" is string equality. `hit@k` here is a *ceiling* on any downstream J: a
  turn never retrieved cannot be answered from.
- **Abstention (category 5).** LOCOMO's adversarial split — 446 questions (22.5% of the set) that
  look answerable and are not, typically an event mis-attributed to the wrong speaker. An
  independent audit (github.com/dial481/locomo-audit) finds **no published LOCOMO result evaluates
  this category at all** — the original harness has a broken formatter for 444 of the 446, so
  vendors drop it. The one axis this library exists for is unmeasured by the entire field, inside
  the field's own benchmark.

### 9a. Retrieval: 0.671 at k=5 — and 0.855 at k=20

**Re-measured 2026-07-26 with both retrieval fixes live** — the
[#81](https://github.com/GiulioDER/RE-call/issues/81) sparse-leg fix and the
[#84](https://github.com/GiulioDER/RE-call/pull/84) dense-scan widening. Full curve, CIs and
per-category rates: [`RESULTS.md` §7a](RESULTS.md), backed by
`results/locomo/postfix_pool20.json`. Headline: **hit@5 0.671** [0.647, 0.694] rising to
**hit@20 0.855** [0.836, 0.872], n=1,536 answerable, bge-small, hybrid, no rerank, pool 20 per leg.

Every depth is scored from **one** retrieval per question — exact rather than approximate, because
the candidate pool does not depend on `k`, so top-k is a prefix of top-max(k) by construction. (The
harness's coherence assert pins the k-row to the pooled headline rate; that is a labelling guard,
not a check of the prefix itself.)

A comparable retrieval anchor — measured on the standard benchmark, not on this repo's own corpus.
Consistent with §8: on ordinary prose the local embedder is not the bottleneck.

**What the fixes bought.** The earlier runs of this harness predate #81: their sparse leg ANDed
every query term, and LOCOMO questions average ~8 content terms against single-turn documents, so
the leg was largely inert and those runs measured an effectively **dense-only** configuration. The
retained pre-fix run (`locomo/depth_curve_pool20.json`) scored **0.624** at k=5 and **0.798** at
k=20 — a correct measurement of that configuration, kept as its record. Against it the working
lexical leg is worth **+0.05 at k=5** and **+0.06 at k=20**; the post-fix 0.671 is a *different
configuration*, not a better sample of the same one. §10 (LongMemEval) remains pre-fix — see its
configuration note.

> An earlier revision of this paragraph also cited a **0.615** from a still-earlier build and read
> the spread against 0.624 as HNSW build noise. The figure stays withdrawn; the *reason* has
> changed. It was removed because its result artifact had never been retained — that artifact was
> committed in [#111](https://github.com/GiulioDER/RE-call/pull/111) and records **0.6152** at k=5
> (`locomo_fastembed_k5.json`), so this repo now holds five pre-fix artifacts, not two. Retaining
> it does not repair the claim: the runs whose spread was read as build noise differ by 0.0006 and
> differ in *candidate pool*, not in index build, so they still cannot support a build-noise claim.
> The figure is now checkable and still not evidence for what it was used for. HNSW build
> nondeterminism is real and is evidenced where it was actually measured — §5b and §6. Which
> artifact belongs to which configuration: [`ARTIFACTS.md`](ARTIFACTS.md).

#### Why quoting one depth was a mistake

An earlier version of this section reported a **single** depth while calling `hit@k` a **ceiling**
on any downstream J. Together those invite a reading the data never supported — that the k=5 figure
bounds what a system built on this library can reach. It bounds what it reaches *at k=5*. Hence the
full curve above. Two things it still does **not** license:

- **Depth is not free.** k=20 hands a generator four times the context of k=5, with four times the
  distractors, and this repo's whole thesis is that a confidently wrong retrieved memory is worse
  than a missing one. 0.855 is a ceiling at a larger context budget, not a better system.
- **cat3 is still the floor** — 0.620 at k=20 against 0.837 for cat1. Depth lifts it (0.228 →
  0.620, the steepest climb of any category) without closing the gap.

**The candidate pool, measured properly.** The natural objection to the curve is that a 20-per-leg
pool makes a k=20 row read the pool's edge rather than the retrieval's reach. The pre-fix attempt
at this control found the k ≤ 20 rows identical and concluded "the pool was not the constraint";
**that inference is retracted and its k=50 row (0.872) withdrawn** — with the sparse leg inert and
`hnsw.ef_search` capping the dense scan at 40, the pool of 100 never existed, so the control varied
a quantity that could not move the metric. Re-run with both defects fixed (2026-07-26, one run per
arm):

> ### ⚠️ RETRACTED 2026-07-28 — the pool-100 column was measured on a DOUBLED corpus
>
> **This is the second retraction of this control, and it invalidates the replacement produced by
> the first.** The pool-100 arm below was run against a corpus in which every document appeared
> twice, so a fixed-size candidate pool held roughly half as many *distinct* documents. Every
> pool-100 figure here — including the k=50 row published as the legitimate replacement for the
> withdrawn 0.872 — comes from that run and is withdrawn.
>
> **Evidence.** A deliberately doubled corpus reproduces the retracted column to **±0.0013 at every
> depth, and exactly at k=20**; a clean corpus does not, missing by up to +0.066. Control: doubling
> costs **−0.0625** at pool 20, so the defect is not depth-specific — and the pool-20 column
> reproduces clean to **0.0000**, so the two columns came from different corpus states. That was
> possible because the guard refusing to index over an existing corpus landed 2026-07-28, *after*
> these artifacts were produced on 07-26. Writeup and artifacts:
> [`results/wrrf/FINDINGS_pool100_contamination.md`](wrrf/FINDINGS_pool100_contamination.md).
>
> `results/locomo/postfix_pool100.json` is **deleted**, not annotated — an annotated wrong number in
> `results/` is still a number someone can read off a table.

| k | pool 20 | ~~pool 100 (retracted)~~ | pool 100 — clean re-run |
|---|---|---|---|
| 1 | 0.398 | ~~0.390~~ | 0.394 |
| 5 | **0.671** | ~~0.596~~ | **0.662** |
| 10 | 0.778 | ~~0.690~~ | 0.753 |
| 20 | 0.855 | ~~0.782~~ | 0.821 |
| 50 | — (beyond the pool) | ~~0.877 [0.860, 0.892]~~ | **not measured** |

- **A deeper pool dilutes a fused prefix, but by roughly one eighth of what this section claimed.**
  Corrected: **−0.009 at k=5** (0.671 → 0.662), not −0.075; **−0.034 at k=20**, not −0.073.
  n=1,536, clean corpus, `results/wrrf/arm_C_rrf_pool100.json`. The *direction* survives, and §7's
  mechanism — RRF scoring `dense[r]` and `sparse[r]` identically — remains a plausible cause; this
  measurement simply no longer evidences it at the published magnitude. Raising `--candidate-k` is
  still a **different fusion configuration**, not a deeper look at the published one.
- **There is currently no valid k=50 row.** The clean re-run used `--k-curve 1,5,10,20`, so depth
  past the 20-per-leg edge is unmeasured. Both the original 0.872 and its replacement 0.877 are
  withdrawn and nothing has replaced them.
- **A downstream consequence, recorded so it is not repeated:** the retracted −0.075 was the
  motivating premise for a weighted-fusion experiment. That work was built to counter a defect about
  eight times smaller than published. It failed on its own terms too, so its conclusion is
  unaffected — but the *motive* was not what this record said it was.

The headline table publishes pool-20 throughout, and its shape through k=20 is not an artifact of
the pool's edge — a 20-per-leg pool supplies more than 20 fused candidates.

```bash
python -m recall.eval.locomo --data locomo10.json --k-curve 1,3,5,10,20
python -m recall.eval.locomo --data locomo10.json --k-curve 1,5,10,20,50 --candidate-k 100
```


### 9b. Abstention: 0.00 out of the box, and why the shipped levers only half-fix it

The default configuration — bge-small, uncalibrated threshold, no judge — abstained on **0 of 446**
adversarial questions ([0.00, 0.009]). That is not a bug; it is §2/§4's lesson under maximum load.
The adversarial turn is *on-topic* — the right event, the wrong person — so it scores a high cosine
and sails past a gap-based threshold. The stale-memory geometry of §4 (the wrong hit outscoring the
right one on similarity) resurfaces here as wrong-attribution.

So the two levers this library ships to raise abstention were tested, each measured **against its
cost to answerable questions** — a mode that abstains on everything scores 1.00 on adversarials and
is useless. Calibration was fit **in-sample**, on the very answerable-vs-adversarial cosines it was
then scored on: not a realistic operating point but calibration's *upper bound*, so any failure to
separate is a property of the data, not the fit.

The four modes, with intervals: [`RESULTS.md` §7b](RESULTS.md), backed by
`results/locomo/postfix_abstention.json`. Discrimination (adversarial-abstain − false-abstain, the
only quantity that matters since either column alone is gameable) runs **0.000 → 0.154 → 0.057 →
0.193** across default, calibrated, judge and both.

Per-conversation calibrated thresholds came out 0.695–0.764 (mean 0.728) and **none of them
certified** — `from_samples` reported separability 0.53–0.69 against the 0.90 bar on every single
conversation, which is §10d's gate doing its job on the workload it was built for.

**The fixes did not move the conclusion, and the way they failed to move it is the finding.**
Better retrieval raises both columns together: calibration now catches 0.574 of adversarials
(was 0.527) but refuses 0.420 of answerable questions (was 0.370). Discrimination —
the only quantity that matters, since either column alone is gameable — is **0.154 against a
pre-fix 0.157**. A retrieval fix cannot improve an answerability judgement, because a
better-retrieved on-topic-but-wrong turn scores *higher*, not lower.

Three findings, one of them a correction of this repo's own first guess:

1. **Calibration moves it — the initial "wrong failure mode, won't help" call was wrong.** Fit on
   the actual distributions it lifts adversarial abstention from 0 to 0.574. The distributions
   overlap, as expected for an on-topic adversarial, but *not completely*, and calibration exploits
   the partial gap. Stated because the prediction was published internally before it was measured.
2. **No mode is a clean win, and the best absolute catch is the worst trade.** `both` refuses 79.6%
   of adversarials but also **60.3% of answerable questions** — unusable. Even the mildest useful
   mode gives up nearly a third of legitimate answers.
3. **Entailment is the better-founded lever, even though its absolute catch is lower.** Its
   0.347 / 0.290 is not in-sample-biased — the judge is pretrained, nothing is fit to the test set —
   whereas calibration's 0.574 is an optimistic ceiling a held-out fit would not reach. Best honest
   *conclusion*: neither closes the gap, and post-fix neither even improves the trade (finding
   above the table).

**The residual is architectural, not a tuning failure.** Separating "what did Caroline realize" from
"what did Melanie realize" when both turns are in the corpus and both score high is entity-level
reasoning — and this library states plainly that it does none: no LLM, no graph, no entity linking
in the retrieval path (README, "Prior art"). LOCOMO cat5 stresses exactly the axis the architecture
excludes by design.

### 9c. A stronger entailment judge: it shifts the curve, it does not cross the line

§9b used one judge at one threshold. The obvious next question — does a *stronger* judge close the
gap? — separates two things a better judge could do: move along the same tradeoff curve (a threshold
change, no new capability) or lift the whole curve (catch more adversarials at the *same*
false-abstain). The experiment scores each judge once and sweeps the threshold analytically, so a
full ROC costs one model pass. **Separation** below is adversarial-abstain minus answerable
false-abstain — how well the judge tells the two classes apart.

The ROC table is in [`RESULTS.md` §7b](RESULTS.md).

> **Configuration: this sweep is the pre-#81/#84 run and has not been re-measured.** §9b's table
> above *has* been, and its `entailment judge` row now reads 0.347 / 0.290 rather than the
> 0.374 / 0.263 this sweep recorded — so the two harnesses are no longer measuring the same
> configuration, and the cross-check between them that this section used to publish has been
> withdrawn. The finding below does not rest on it: it is a within-sweep comparison of two judges
> scored in the same pass, which the fixes do not disturb.

n=446 adversarial, n=400 answerable, seed 0. Three results:

1. **A stronger same-task judge does help — modestly.** electra-base dominates distilroberta across
   nearly the whole curve; at a matched ~0.30 false-abstain it catches ~0.50 of adversarials against
   distilroberta's ~0.40, and best separation rises 0.197 → 0.240. So model quality is *a* lever,
   which refines §9b's lean on "the judge isn't the bottleneck".
2. **The threshold is as large a lever as the model.** Sweeping the default judge from 0.5 to 0.95
   lifts separation 0.112 → 0.197 on its own — the shipped 0.5 is not this task's best operating
   point.
3. **No configuration crosses into usable territory.** The best point measured — electra at
   threshold 0.99 — still refuses **43.8% of legitimate questions** to catch 67.7% of adversarials.
   Separation tops out at 0.24: the judge is only weakly telling "Caroline realized X" from "Melanie
   realized X", because both topically answer the question. A stronger QNLI judge nudges the curve;
   it does not change the §9b conclusion.

The strongest available entailment model, `nli-deberta-v3-large`, was tried and **excluded**: it is
3-way NLI, so it must be applied as (premise=passage, hypothesis=question), and an NLI model trained
on declarative hypotheses scores the entailment class ≈0 for a question-hypothesis whether the
passage answers it or not (measured: 0.000 for both the right and the wrong-speaker passage). Making
it usable would need a question→statement rewrite — a generation step this library does not have.

Reproduce all of §9 — the dataset is public and the harnesses ship here:

```bash
curl -sLO https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python -m recall.eval.locomo                  --data locomo10.json                       # 9a + 9b default
python -m recall.eval.locomo_abstention       --data locomo10.json --answerable-sample 40 # 9b ablation
python -m recall.eval.locomo_entailment_sweep --data locomo10.json --answerable-sample 40 # 9c ROC sweep
```

### 9d. Answer accuracy, cost and speed: the paired head-to-head vs Mem0

§9a measured *retrieval* and was explicit that a top-5 hit is **not** the LLM-judged answer accuracy
the incumbents report. This subsection measures that metric directly — but as a **paired** contest,
not a leaderboard number. RE-call and **Mem0** (the most-adopted open-source memory layer) answer
the **identical** LOCOMO question set through an **identical generator and judge**; the only variable
is the memory. Paired **McNemar** over per-question outcomes, full **n=1,540** answerable questions.

| generator | budget | judge | RE-call | Mem0 | paired p |
|---|---|---|---|---|---|
| gpt-4o-mini | item (k=10/10) | gpt-4o-mini | **0.416** | 0.378 | 0.0059 |
| gpt-4o-mini | item | gpt-4o | **0.466** | 0.412 | 0.00018 |
| gpt-4o-mini | token (k=10/20) | gpt-4o-mini | **0.416** | 0.370 | 0.00077 |
| gpt-4o-mini | token | gpt-4o | **0.466** | 0.411 | 0.00018 |
| gpt-4o | token | gpt-4o | **0.484** | 0.444 | 0.0065 |

RE-call is the more accurate of the two on every row, and the margin survives **Holm–Bonferroni**
across all five (largest adjusted p = **0.012**). It also holds on the 1,369 questions where the two
judges agree (0.440 vs 0.399, p=0.006), so it is not a judge-noise artifact.

**Reader-tier caveat — the lead is a property of the reader, not a universal fact.** At the gpt-4o
judge the margin runs +0.055 (gpt-4o-mini generator) → +0.041 (gpt-4o) and **reverses on Claude
Sonnet** (RE-call 0.565 vs Mem0 0.608, n=584) — a generator run *after* pre-registration. The
mechanism is measurable: Mem0 returns LLM-compressed facts a stronger reader can exploit; RE-call
returns raw turns. The claim is for the OpenAI reader tier the field benchmarks with, not beyond.

**Mem0 as-shipped.** On Mem0's own documented default embedder (`text-embedding-3-small`, both arms,
full n=1,540): RE-call **0.42** vs Mem0 **0.366** (+0.046 to +0.057, p ≤ 0.0014). Its shipped
embedder did not close the gap — it widened it slightly (Mem0 scored *below* its own bge-small).

**Cost.** Metered memory-layer LLM usage building the full benchmark's memory: RE-call **0 calls /
$0**; Mem0 **272 calls / 2.6M tokens / $7.29** (gpt-4o extraction). RE-call's write path calls no
LLM, so this stays $0 at any scale and any model.

**Speed** (isolated memory-layer timing, both arms on the same local embedder, idle machine): ingest
**~4.3× faster** — 67 s vs 288 s for two conversations, the extraction gap as wall-clock; retrieve
median **77 ms vs 104 ms** (~26%, non-overlapping intervals) *despite* RE-call opening a Postgres
connection per call. Ingest and the $0/0-calls cost are the robust claims; retrieve speed is
reported as **directional** — the repeated-query bootstrap CI is optimistic (effective n ≈ 80) and
the backends differ (Postgres vs in-process Qdrant).

> **Reproduce, and what is actually published.** The harness, the pre-registration, the blind
> human labels, the corrupt-key list and an independent adversarial recompute of every cell are in
> **`benchmarks/`** on master (`benchmarks/REVIEW.md`, `benchmarks/PREREGISTRATION.md`). The
> **per-question dumps are not published** — `benchmarks/results/` is gitignored, so each run
> writes them locally only.
> Regenerate with `python -m benchmarks.run --arm {recall,mem0} --embedder router:openai/text-embedding-3-small --conversations 10`
> (as-shipped arm) and `python -m benchmarks.latency` (timings).

### 9e. BEAM: what Mem0's 2026-07 headline numbers actually are, and what it takes to sit beside them

Mem0's "token-efficient memory algorithm" post reports LOCOMO 92.5, LongMemEval 94.4 and BEAM
64.1 (1M) / 48.6 (10M). Before spending anything to answer them, we read the harness and the
published artifacts (`mem0ai/memory-benchmarks`, Apache-2.0). Four facts change what a comparable
reply has to look like — each one verifiable from files in that repo, and none of them stated in
the blog post:

1. **The algorithm is in the OSS package; the published numbers are not.** Single-pass ADD-only
   extraction, multi-signal retrieval (semantic + BM25 + entity), and entity linking all landed in
   the open-source SDK at **v2.0.0** — they are in `mem0/memory/main.py` today, and our arm has
   been running them since. But the published cells come from `results/platform/`, i.e. Mem0
   Cloud, which their own post says "includes proprietary optimizations" absent from the SDK. No
   OSS BEAM results are published at all. So *no* `pip install mem0ai` at *any* version reproduces
   64.1, and a number that claims to is measuring something else.
2. **The answerer and judge are `gpt-5`, not gpt-4o.** The BEAM runner's CLI defaults to `gpt-5`
   for both, and the published metadata records `gpt-5` on Azure. (The README's "default: GPT-4o"
   describes the generic pipeline, not this benchmark.) A BEAM score is a property of the
   (retrieval, answerer, judge) triple; changing the reader changes the number, as §3 of the
   head-to-head shows in miniature.
3. **64.1 is the mean rubric-nugget score, not a pass rate.** `avg_score` = 0.6409 at the `top_200`
   cutoff; the pass rate (`accuracy`, nugget mean ≥ 0.5) for the same run is **70.14 %**. Quoting
   the pass rate as the headline would inflate every system by ~6 points while looking like the
   same metric.
4. **The published artifact is per-question.** `beam_1m_results.json` carries all 700 question ids,
   rubrics, generated answers and per-nugget scores. That is what makes a genuinely *paired*
   comparison possible without paying for a Mem0 arm at all.

**The design that follows from those four.** `benchmarks/beam/` runs RE-call over the same 700
question ids with Mem0's answerer prompt and judge prompt vendored byte-for-byte, then scores
**both** systems' answers with the *same judge instance in the same session* — ours on our answer,
ours on their stored answer. Comparing our judge's verdict on our answer against their judge's
verdict on their answer would fold judge drift into the headline and call it retrieval. The residual
confound is stated rather than resolved: their *answers* are historical (Azure gpt-5, May 2026), so
if their platform has moved since, this measures the platform as published, not as it is today.

Two harness details exist because getting them wrong would have handicapped our own arm invisibly:

- **Question ids are reconstructed, not invented.** `dataset.py` reproduces upstream's
  `{size}_{conv}_q{qi}_{type}` numbering from the same `BEAM_QUESTION_TYPES` ordering, and the join
  is asserted: for conversation 0 our 20 ids are set-identical to the published 20.
- **BEAM dates one turn per batch, not every turn.** In the 1M bucket that is 10 dated turns out of
  1,710. Upstream applies each batch's anchor to everything ingested from it, so every Mem0 memory
  carries a date; reading the field per-turn would have left 99.4 % of *our* memories undated while
  the vendored answerer prompt — which prints each memory's date and is told to reason about
  ordering — graded us on `temporal_reasoning` and `event_ordering`. The anchor is propagated.

BEAM's ten ability types include **abstention**, so this arm reports what the LOCOMO arm reports:
the abstention rate on the questions where withholding is correct, *and* the false-abstain rate on
the other nine categories, as two independent numbers. A system that refuses everything scores
1.00 on the first; only the second catches it.

```bash
# $0 — index + retrieve only, no LLM call. Validate the arm before committing money.
python -m benchmarks.beam.run --dry-run --data <beam_1M.parquet> --conversations 0
# paid — RE-call over the 1M bucket, Mem0's answerer + judge
python -m benchmarks.beam.run --data <beam_1M.parquet> --model openai/gpt-5
# paid — the same judge, re-scoring Mem0's published answers
python -m benchmarks.beam.run --rejudge-mem0 <beam_1m_results.json> --model openai/gpt-5
# free — pair them: McNemar on accuracy, abstention and false-abstain, Holm across the three
python -m benchmarks.beam.pair --a <recall.json> --b <mem0-rejudged.json>
```

Cells land here when the paid arms run. Provenance for every constant above:
`benchmarks/beam/UPSTREAM.md`.

### 9f. Why 92.5 and our 0.444 are not the same measurement

The same harness read for §9e also explains the LOCOMO headline, and the explanation is entirely in
the protocol rather than in either memory layer. Verified by reading `benchmarks/locomo/prompts.py`
and the published `results/platform/locomo_results.json` in that repo:

| Their published LOCOMO run | Our head-to-head |
|---|---|
| `CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]` — **category 5 excluded entirely** | cat 5 is the axis §9b exists to measure |
| Answerer told: *"COMMIT AND ANSWER … NEVER say 'not specified' … NEVER return an empty answer"* — abstention is **forbidden** | abstention is the measured behaviour |
| `ANSWERER_MEMORY_LIMIT = 200` memories, chronological (~7k tokens — their "mean tokens per query") | k = 10/20, token-matched |
| Judge: dates within **14 days** correct, durations within **50 %** correct, 1-of-N partial credit, and evidence used *"only to ACCEPT answers, never to reject them more strictly"* | strict single verdict |
| Answerer and judge: **`gpt-5`** (Azure), n = 1,540 | gpt-4o-mini / gpt-4o |

Every one of those moves the number the same way, and none of them is cheating — they are choices,
documented in code. But they mean **92.5 and our 0.444 are not two values of one quantity**, and any
table that stacks them is comparing a lenient judge over four categories with abstention forbidden
against a strict judge over five with abstention scored.

Note the model row in particular: their answerer and judge default to `gpt-5` on *both* benchmarks,
not gpt-4o. A "reproduction" run configured on gpt-4o would differ from the published cells for that
reason alone, before retrieval is considered.

The genuinely comparable experiment is the reverse of what we ran: put RE-call through *their*
unmodified harness. The seam is one call — `mem0.search(question, user_id, top_k)` returning
`[{memory, created_at}]` — which is exactly the interface `benchmarks/beam/systems.py` already
implements for BEAM. That is the natural next arm, and the adapter for it already exists.

### 9g. The BEAM arm embeds through the hosted embedder — stated, not buried

The LOCOMO arms in §9a-c run RE-call on local `bge-small` via fastembed. The BEAM arm does not:
it uses `openai/text-embedding-3-small` through OpenRouter. That is a real methodological change
and it belongs in the open, so:

**Why.** BEAM's 1M bucket is **37.2 M tokens** of dialogue — 1.06 M per conversation, mean turn
620 tokens, an order of magnitude denser than LOCOMO's chat lines. Through a 33 M-parameter model
on CPU that is ~70 TFLOPs per conversation: measured at 12-20 minutes each and ~10 hours for the
bucket, at 4.5 GB resident. The same tokens cost ~$0.75 hosted, at ~42 s per conversation.

**Why it does not weaken the comparison — it strengthens it.** Mem0's published BEAM run embeds
with `text-embedding-3-small` too. Running RE-call on `bge-small` against a Mem0 cell built on
`text-embedding-3-small` would have handicapped our own arm on a dimension nobody asked about,
and any deficit would have been reported as a retrieval result. Matching their embedder isolates
what the arm is actually for: retrieval and answer construction, judged by their judge.

**What it costs in comparability elsewhere.** A BEAM cell and a LOCOMO cell in this repo are now
not embedder-matched to each other. They were never meant to be pooled — different corpora,
different judges, different protocols (§9f) — but no table should imply otherwise.

Vector width changes 384 -> 1536 with the embedder, so `bench_beam_chunks` is rebuilt for the run;
nothing is carried over from a 384-wide index.

### 9h. BEAM's abstention category is a hallucination test, and it is where the margin is

Three questions were asked of this data: is the benchmark measuring something real, is it
backfitted to Mem0, and is there room to improve. The answers are yes, no, and yes — with a
concrete mechanism for the third.

**The category is sharp.** Scoring Mem0's own published answers (n=70 abstention questions):

| Mem0 did | n | mean score |
|---|---|---|
| abstained | 38 | **0.974** |
| answered | 32 | **0.016** |

Near-perfectly binary. The rubric nuggets read "there is no information related to X", so the
judge is testing exactly one thing: does the system invent an answer when the evidence is absent.
Mem0 invents one **46 % of the time** — e.g. answering "User testing showed a positive response:
the dynamic language switching feature achieved a 90 % satisfaction rate" to a question whose
gold answer is that no such feedback was ever recorded. The corpus does contain "achieving a 90 %
satisfaction rate is a strong start" — the ASSISTANT speculating, which the retriever surfaced and
the answerer read as fact.

**It is not backfitted to Mem0.** A benchmark tuned to flatter them would not expose a 46 %
fabrication rate on its own vendor, and would not score them 64.1 here against 92.5 on LOCOMO.
The dataset is third-party (ICLR 2026), and the harness's leniency (§9f) runs the OTHER way — it
is generous to whoever is being scored, including us.

**The margin is the largest in the benchmark.** Mem0's abstention cell is 0.536. A system that
withheld correctly every time would score ~0.97. No other category has a 0.43 gap sitting in it,
and it is the one category whose claim is RE-call's own.

**Why RE-call does not capture it yet, precisely.** `apply_entailment` abstains only when NO hit
entails an answer — an `any()` over the trusted pool. With ~200 candidates, two false-positive
entailments are enough to suppress the abstention, and two is what we measured. The judge itself
is not the problem: it correctly rejected 8 of the top 10 on the question above. The AGGREGATION
is.

Measured on conversation 0, entailed-count in the top 10 by class:

| class | counts | mean |
|---|---|---|
| unanswerable | `[0, 2]` | **1.00** |
| answerable | `[0,0,0,2,3,3,3,4,5,6,7,7,8,8,9,10,10,10]` | **5.28** |

| rule | correct-abstain | false-abstain |
|---|---|---|
| abstain if entailed < 1 (**current**) | 50 % | 17 % |
| abstain if entailed < 3 | **100 %** | 22 % |

Note the direction: this is the separation the COSINE threshold could not provide at any value,
because on BEAM the two classes are ordered **the wrong way round**:

| BEAM 1M, top-1 cosine | median |
|---|---|
| unanswerable (the `abstention` category) | **0.676** |
| answerable (the other nine categories) | 0.641 |

That is the raw top-1 cosine per class under BEAM's own labels, collected by
`benchmarks/beam/calibrate.py` — embeddings and cosines only, no answerer and no judge. The two
medians above come from the `*_report.json` sidecar, which records the per-question cosines; the
calibration JSON itself is not committed, so this is a reproducible figure rather than an
artifact-backed one:

```bash
python -m benchmarks.beam.calibrate --data <beam_1M.parquet> \
    --out beam_calibration.json --save-uncertified
```

`--save-uncertified` is required here because the fit this produces on BEAM is **0.617 and does
not certify**, and `calibrate` now refuses to write an uncertified threshold by default — it used
to write one straight to the process-global `calibration.json` that `trusted_search` autoloads.
The sidecar report, which is what carries the cosines quoted above, is written either way.

A threshold is a monotone rule on that column, so no value of it can separate two classes that are
already in the wrong order — the argument §9n generalises to every rule of that shape. Counting
entailments recovers the ordering that similarity inverts.

**⚠ n = 2 unanswerable questions in this sample.** This is a direction with a mechanism, not a
result. It needs the full 30 abstention questions across conversations 0-14 before any claim is
made — that validation is free (dry runs make no LLM call) and is the next thing to run.

**The research item, stated as a change:** `recall.entailment.apply_entailment` currently exposes
one abstention policy, "none entailed". The data says the policy should be a threshold on the
count (or fraction) of entailed hits, and that the right value is not 1. That is a library change
validated on adversarial held-out data, not a benchmark tweak — the abstention claim is what
RE-call sells, and on the one public benchmark that tests it directly, the default policy is too
permissive to collect.

### 9i. RETRACTION of §9h's proposed fix — the count rule does not pay

§9h reported, on ONE conversation, that counting entailed hits separates BEAM's unanswerable
questions from its answerable ones, with "abstain if entailed < 3" giving 100 % correct-abstain
at 22 % false-abstain. It flagged n = 2 unanswerable and called itself "a direction with a
mechanism, not a result". The full probe (30 unanswerable, 270 answerable, conversations 0-14,
$0) came back weaker and the conclusion does not survive:

| entailed in top-10 | §9h pilot (n=2) | full probe (n=30) |
|---|---|---|
| mean, unanswerable | 1.00 | **3.57** |
| mean, answerable | 5.28 | **5.93** |
| "< 3" correct-abstain | 100 % | **43.3 %** |
| "< 3" false-abstain | 22 % | 19.6 % |

The separation is real but modest, and on BEAM's 9:1 answerable:unanswerable mix it never pays:

| policy | correct-abstain | false-abstain | net vs shipped |
|---|---|---|---|
| answer if ≥1 entails (**ships today**) | 23.3 % | 9.3 % | — |
| ≥2 | 26.7 % | 11.8 % | **−0.011** |
| ≥3 | 43.3 % | 19.6 % | **−0.036** |
| ≥5 | 66.7 % | 34.4 % | **−0.092** |

Every stricter policy gains on 30 questions and loses on 270. **The shipped `any()` policy is
already the best of the five on this benchmark**, which is the opposite of §9h's recommendation.

Two things worth keeping from this:

**The mechanism in §9h was right; its extrapolation was not.** `any()` over ~200 candidates IS
maximally permissive, and entailment count DOES order the classes correctly where cosine inverts
them (§9h). Neither of those claims is retracted. What is retracted is that changing the policy
improves the cell — it does not, and the reason is arithmetic that was available before the probe
ran: abstention is 10 % of BEAM, false-abstention risk is 90 %.

**A benchmark whose payoff is 9:1 against withholding cannot be the venue for an
abstention claim.** That is not a complaint about BEAM — its abstention category is well built
(§9h's hallucination finding stands: Mem0 fabricates on 46 % of unanswerable questions). It means
the abstention argument needs a metric that prices a false answer against a withheld one, and
BEAM's aggregate does not. Reporting our abstention story through this aggregate would understate
it no matter how good the policy got.

### 9j. Newest-wins dedup: rejected on principle, and it does not fire anyway

Proposed after §9's TTL diagnosis: collapse near-identical chunks, keep the newest, so a fact
restated 24 times cannot outvote its own correction. Two independent findings kill it.

**On principle.** Recency is not reliability, and this corpus contains the counter-example: the
top-ranked chunk for the user-feedback question was *"Achieving a 90 % satisfaction rate is a
strong start"* — the ASSISTANT speculating, not a recorded fact. Mem0 built a fabricated answer
on it. A newest-wins rule promotes exactly that: recent, on-topic, and without authority. A
memory layer that confuses the two is worse than one that does not order at all.

**Empirically it barely fires** (58 questions, k=200, conversations 0-14, $0):

| cosine threshold | survivors of 200 | gold value still present |
|---|---|---|
| 0.98 | 196 | 0.466 |
| 0.95 | 193 | 0.466 |
| 0.92 | 187 | 0.466 |
| 0.90 | 183 | 0.466 |

Four chunks collapse at the strict threshold, seventeen at the loose one, and the presence of the
gold value does not move at any of them. The reason matters more than the numbers: the 24 stale
restatements are **not textual near-duplicates**. They are paraphrases — "I set the TTL to 15
minutes", "cache expiry is 900 seconds", "using a 15-minute TTL". The repetition is SEMANTIC and
cosine does not group it, so no similarity threshold reaches it.

That also retires the idea in its general form: any dedup keyed on embedding similarity will miss
the repetition that actually causes the failure.

**What survives.** The diagnosis in §9 stands and is causal — at k=5, where the stale copies fall
outside the window, the same system answers the TTL question correctly and scores 1.00 against
0.00 at k=200. Cutting context defeats stale repetition; deduplicating it does not. That is why
the k sweep, not the dedup rule, is the result worth carrying forward.

### 9k. The "wiped tables" were RLS working correctly — a false alarm, and a passed isolation test

Reported mid-session: all five BEAM chunk tables found empty, no process running, no error, cause
unknown. A guard was added and the next run was held pending investigation.

**There was no deletion.** The tables carry row-level security (`relrowsecurity` and
`relforcerowsecurity` both true) with the policy `tenant_id = current_setting('recall.tenant_id')`.
Raw `psql` sessions do not set that variable, so RLS correctly returned nothing. With the context
set, the data is untouched:

| table | live tuples | rows deleted, all time |
|---|---|---|
| bench_beam_k45 | 107,902 | **0** |
| bench_beam_fix | 108,015 | **0** |
| bench_beam_probe | 107,880 | **0** |

All 15 tenants present; `beam-1m-0` alone holds 6,998 chunks. The measuring instrument was wrong,
not the system — the third time in one session that a conclusion was drawn from a bad probe
(the 4-question abstention sample, the n=2 entailment pilot, and now this).

**What it accidentally demonstrated.** A session holding a *valid login role* on the database saw
**zero rows** because it lacked the tenant context. That is precisely the property Track E of the
suite is meant to test, verified unintentionally against a live index. It is weak evidence — one
observation, not an adversarial suite — but it is evidence, and it is the first time the isolation
claim has been exercised outside its own unit tests.

**What is retracted:** the wipe, its unknown cause, and the concern that a mid-run deletion could
have corrupted the k=45 result. **What stands:** the empty-index guard, which is correct on its
own terms — an ingest that consumes turns and stores no chunk is a broken run, not a low score.

**What must be re-measured:** the five questions that returned cosine 0.0 in the threshold probe
now have no explanation, since every tenant is populated. That probe is void; the thirteen
questions measured between 0.418 and 0.498 remain plausible but should be confirmed.

### 9l. `temporal_reasoning` — diagnosed, no cheap fix, documented as a limit

Second-worst category (0.408 vs Mem0's 0.567). Unlike `instruction_following`, it is NOT a
threshold artefact: of 7 badly-lost questions only 1 had empty retrieval, and 5 were answered
confidently and wrongly.

Every one is an interval question — "how many days between A and B" — and in every one the wrong
INSTANCE of a date was used:

| gold | our answer |
|---|---|
| 25 Mar → 1 Apr = 7 days | 14 days, using the *updated* deadline of 15 Apr |
| 25 Mar → 10 Apr = 16 days | 26 days, using a *different* viewing on 15 Mar |
| 15 Feb → 20 Feb = 5 days | 0 days, using 10 Jan — the date the deadline was *set* |

The decisive detail: sometimes the correct instance is the OLDER one (the original deadline,
not the revision). That rules out every recency heuristic, and is the third independent line of
evidence against newest-wins (§9j).

Mem0 answers these because its stored memory is one distilled line — "Sprint 1 deadline: February
15, 2024" — while ours is the same date scattered across many raw turns in different roles
(when set, when revised, when discussed). This is the one category where LLM distillation at
ingest is genuinely the better architecture, and no retrieval-side change we can afford replicates
it. Recorded as a known limit rather than an open task.

### 9m. The absolute threshold is embedder-fragile — indicative, NOT yet established

The shipped abstention gate is an absolute cosine, `DEFAULT_GAP_THRESHOLD = 0.50`. Measured
top-1 distributions on the SAME three BEAM conversations, varying only the embedder:

| embedder | min | q05 | median | max |
|---|---|---|---|---|
| bge-small (fastembed, the default) | 0.629 | 0.728 | **0.825** | 0.939 |
| text-embedding-3-small | 0.403 | 0.498 | **0.635** | 0.852 |

The medians differ by 0.19, and bge-small's MINIMUM (0.629) sits above the cloud embedder's
median. What the one constant does in each regime:

| threshold 0.50 | starves |
|---|---|
| bge-small | **0 of 54** (0 %) |
| text-embedding-3-small | **19 of 270** (7 %) |

On the default embedder 0.50 is not a threshold at all — it sits below the observed minimum and
never fires. That is why the defect was invisible until an embedder swap: the gate only starts
discarding answers on a model nobody had run it against. A corpus-quantile floor derived from the
data produced 0.728 and 0.498 respectively — two very different numbers, comparable behaviour
(4 % and 6 %) — which is the thing a constant cannot do.

**Why this is filed as indicative rather than established.** The pattern of this session is that
small samples reverse at scale: an n=2 entailment pilot promoted a policy that was net NEGATIVE at
n=30 (§9i), and a k-sweep advantage of +0.029 at n=60 became +0.0007 at n=300. This measurement
has the same profile — 54 answerable questions in one arm, and **one corpus**. Varying the embedder
while holding the corpus fixed cannot separate an embedder effect from a property of those three
conversations, because the top-1 distribution depends on both.

**What would settle it:** 3 embedders x 3 corpora (BEAM, LOCOMO, and the curated memory corpus,
which is the documental case RE-call actually targets), with the statistic being the VARIANCE of
the starve rate across the nine conditions — large for the constant and small for the quantile, or
the claim fails. ~6-10 h of VPS CPU, ~$1 of cloud embedding, no LLM spend at all.

**It can fail.** If the three embedders turn out to live in similar cosine regimes, embedder
independence is not a real problem, 0.50 is fine, and this line of work closes as unnecessary.

**What is NOT claimed:** that a rate-based threshold improves the BEAM score. It does not. On BEAM
`absolute@0.40` remains the best cell of everything tested (268 of 270 answerable served), because
here the unanswerable questions score HIGHER than the answerable ones (§9h) and no function of the
score separates them. The quantile's argument is robustness across deployments, not accuracy here.

### 9n. The regime sweep settles the problem; four candidate fixes are now measured and dead

**Established** (2 corpora x 3 embedders, n=100 and n=775 per cell, no LLM):

| | median (memory) | median (BEAM) | range | starve @0.50 |
|---|---|---|---|---|
| bge-small | 0.852 | 0.819 | 0.284 / 0.300 | 0 % / 0 % |
| bge-large | 0.827 | 0.782 | 0.316 / 0.344 | 0 % / 0 % |
| text-embedding-3-small | 0.710 | 0.608 | 0.422 / 0.376 | 0.3 % / **16 %** |

`DEFAULT_GAP_THRESHOLD = 0.50` sits at the **0th percentile of five distributions and the 16th of
the sixth**; `absolute@0.40` starves nothing anywhere (spread 0.0000). The constant is not
mis-tuned, it is inert everywhere except one cell. Model spread (0.142-0.211) exceeds corpus shift
(+0.033 / +0.044 / +0.102), and the two INTERACT — the corpus effect is three times larger for one
model than another — so no stored per-model constant can work.

Predictions were committed before the run (`PREDICTIONS-regime-sweep.md`). The discriminating one
held: bge-large landed at 0.782, inside the predicted 0.78-0.86 and far from the cloud model's
0.608, so the split is a property of the model FAMILY and not of local-vs-cloud plumbing. Two were
wrong: the memory-corpus levels were over-estimated (0.85 actual vs 0.88-0.93 predicted — the
near-duplicate bias from `description:` queries is real but much smaller than assumed), and the
spread of `absolute@0.50` was under-estimated at 0.05-0.09 against an actual **0.1600**.

**Four candidate replacements, all measured, all dead:**

| rule | why it fails |
|---|---|
| per-query percentile | vacuous — the top score clears its own distribution's percentile by construction; 268 vs 267 served from p=0.0 to p=0.5 |
| gap (top vs median) | starves 64 of 270 answerable to gain 2 correct abstentions at 0.10; flattens exactly where a corpus restates one fact many ways |
| corpus quantile on real queries | works, but **tautological** — the floor is computed from the scores it is applied to. Describes; not shown to generalise |
| corpus quantile from self-queries (H4) | **fails on all three conditions**: derived floors 0.792 / 0.759 / 0.621 against real-query floors 0.766 / 0.744 / 0.590, starving ~10 % / ~9 % / >10 % against a 5 % target |

H4's failure mode is the one written down before running it: a chunk is phrased in the corpus's own
register, a user's question is not, so self-queries sit high and the derived floor is too strict.
The gap is systematic (+0.026, +0.015, +0.031, always the same direction) and WORST on
text-embedding-3-small — the model where the constant does damage and a replacement was most
needed.

**Why all four failed, in one sentence.** They are all monotone functions of the same score, and
§9h established that on BEAM the unanswerable questions score HIGHER than the answerable ones. A
monotone transform preserves order; the order is what is wrong. This decomposes the work into
**Problem A** (cross-model comparability — solvable by rescaling) and **Problem B** (the score does
not separate the classes — NOT solvable by any rescaling), and every negative result of the last
two days was an attempt to solve B with an instrument that can only touch A. It also explains why
the entailment guard is the only mechanism that moved the abstention number at all: it is the only
one that introduces evidence of a different KIND.

**Design defect to fix next time:** the sweep stored summary statistics rather than raw scores, so
H4's starve rates had to be INTERPOLATED between the stored q05 and q10 rather than computed. The
direction and order of magnitude hold; the second digit does not. Discarding the data the next
question needs is the same error as the sampling mistakes earlier in the session.

**Where this leaves the threshold.** The problem is established and no replacement is proven.
`absolute@0.40` is the best measured configuration for this embedder and starves nothing in any of
the six conditions, but it is still a constant and will be wrong for the next model. Shipping it as
a new default would repeat the original mistake with a different number.

### 9o. BEAM is the worst case, not the typical one — and the entailment guard does not discriminate

Five signals had failed to separate BEAM's unanswerable questions from its answerable ones, the
last one (lexical coverage, §9n) failing INVERTED — coverage 0.741 for unanswerable against 0.717
for answerable, the same direction cosine shows. Two signals sharing no mathematics and no model,
inverted identically, pointed at the questions rather than the retriever.

So the ordinary case was built mechanically, with no labelling: hold out 120 memos from the
787-memo curated corpus, index the other 657, and use each memo's own `description:` as its query.
Descriptions of indexed memos are answerable; descriptions of held-out memos are unanswerable
because the document is genuinely absent. Predictions were committed first (`e273c99`).

| signal | answerable | unanswerable | mean separation | **AUC** |
|---|---|---|---|---|
| cosine | 0.7055 | 0.6250 | +0.0805 | **0.7802** |
| entailment | 0.7715 | 0.6917 | +0.0798 | **0.5945** |

**P1 held (predicted +0.05-0.15, actual +0.0805).** On a corpus whose unanswerable questions are
about genuinely absent subjects, cosine separates in the RIGHT direction. **BEAM's inversion is a
property of its adversarial construction, not of retrieval.** Every BEAM number in this repo is
therefore an upper bound on difficulty, not an estimate of deployed behaviour, and must be
presented that way.

**P2 held.** The separation is real but narrow — the corpus is topically dense, so a held-out memo
still finds plausible neighbours.

**P3 failed, and failed BACKWARDS.** The prediction was that entailment would separate BETTER than
cosine. It separates far WORSE: AUC 0.59 against 0.78, barely above the 0.50 of no signal at all.

**A reading error of mine, worth recording.** The two mean separations are nearly identical
(+0.0805 vs +0.0798) and I first read that as "equivalent". It is not: a difference of means says
nothing about discrimination unless normalised by spread. Entailment shifts BOTH classes up by
~0.07 while overlapping far more, which a mean cannot show and AUC does. The lesson is the same one
as the n=2 and n=4 samples earlier — the statistic has to match the question being asked.

**The tails are worse still.** To abstain on half the unanswerable questions:

| signal | threshold | false-abstain cost |
|---|---|---|
| cosine | 0.6283 | **13.7 %** |
| entailment | 0.7000 | **26.0 %** |

Entailment costs twice as much for the same benefit — it does not help where a gate is most needed.

**Consequence for the abstention lane.** The small movement the entailment guard produced on BEAM
is not reproduced where the test is fair to both signals. At AUC 0.59 it is close to no
discriminator at all, so that movement was plausibly noise or an artefact of the adversarial
construction. The guard costs a cross-encoder — 4 hours of CPU for 777 queries here — and buys
nothing measurable. **It should not be promoted toward a default, and the abstention lane has no
remaining candidate with an empirical basis.**

**The positive result, which is the one to carry forward.** On an ordinary corpus the plain cosine
works: AUC 0.78, abstaining on half the unanswerable questions for a 13.7 % false-abstain cost.
The retrieval signal was never the problem. The THRESHOLD was — an absolute constant that is not
comparable across models (§9n: 0th percentile of five distributions, 16th of a sixth). That is the
one thing two days of measurement established with certainty, and it is already shipping as a
warning in PR #105.

### 9p. Provenance note — which code produced the BEAM cells

`/opt/recall-beam` on VPS2 is not a git checkout: it was unpacked from a tarball and then patched
file-by-file over the session, so "which commit produced this number" cannot be read off it. At
session close every file was md5-compared against `bench/beam-1m`:

- `benchmarks/beam/{run,systems,dataset}.py` — **identical** to the committed branch.
- `recall/{trust,calibration}.py` — **diverged**, and this is stated rather than quietly synced:
  the VPS carried the FIRST version of the calibration auto-load (`load_for(embedder.name)` read
  directly), while the branch carries the hardened one (defensive `getattr`, mtime-keyed cache).

**The divergence does not affect the cells.** Both versions load the same calibration for a real
embedder; the hardening guards against a stub without `.name` and removes a file read from the hot
path — correctness and latency, not retrieval behaviour. The 0.594 stands.

Both files were re-synced and re-verified at close, so a re-run from `/opt/recall-beam` now matches
the branch exactly.

**The process defect worth keeping:** a results directory that is not a checkout cannot answer
"what code made this", and the answer had to be reconstructed by md5 at the end rather than being
knowable throughout. The BEAM harness should be deployed as a `git clone` at a named commit, the
way `scripts/deploy.sh` treats the sentiment project — the CLAUDE.md deploy recipe already says to
verify md5 against master for exactly this reason, and that rule was applied here only at closing
time instead of at each patch.


## 10. LongMemEval: the retrieval result, and the abstention failure underneath it

§9 measured LOCOMO — the benchmark the vendors report. This section measures the *other* public one,
**LongMemEval** (MIT, `xiaowu0162/longmemeval-cleaned`), which was worth running separately for a
specific reason: it is the only benchmark in this field whose question taxonomy **names** the two
things this library is about — **knowledge-update** (78 instances) and **abstention** (30). The
retrieval protocol published alongside it *skips every abstention instance*, on the reasonable
grounds that they have no answer location. Between §9 and this section, that class is now measured
on two independent benchmarks by two independent harnesses.

**The headline is that they agree.** §9 found LOCOMO's adversarial split unusable at every setting —
no threshold, no judge, and in §9c not even a *stronger* judge crosses into usable territory. This
section hits the same wall from the other side: a different corpus, a different question taxonomy,
and a comparison across six candidate signals rather than a judge sweep. Convergent negative results
from two harnesses are worth more than either alone, so the agreement is stated here rather than
left for a reader to assemble.

`bge-small` (the free local embedder), hybrid dense+sparse, no reranker. 500 questions, calibrated
on half and scored on the other half.

> **🔒 Weakest evidence here, for two reasons.** (1) These runs **predate
> [#81](https://github.com/GiulioDER/RE-call/issues/81)**: the sparse leg ANDed every query term, so
> on questions of this length it rarely fired, and every retrieval row is a **lower bound on the
> fixed hybrid configuration**. (2) **No result artifact was retained** — the indexes and run output
> are gone (the merged `lme_s` index alone cost 6h39m to build), so these tables cannot be diffed
> against anything in this repo. A post-fix re-score is tracked as follow-up, not silently pending.
> The *abstention* conclusion — §10's actual finding — is unaffected either way: it rests on signal
> separability (AUC ≤ 0.753 across six candidates), and a better candidate pool does not turn a
> relevance signal into an answerability signal.

Three candidate-set sizes, because the benchmark's own protocol gives each question its own
~49-session haystack while a single merged index is what a real memory store looks like. Full table:
[`RESULTS.md` §8](RESULTS.md). Headline: **hit@5 0.970** [0.94, 0.99] on the per-question arm,
falling to 0.719 on Oracle (940 sessions) and **0.366** on a merged 19,195-session index. hit@5 is
monotone in candidate-set size across a 390× range — a coherence check the harness could have
failed and did not.

**Knowledge-update — the benchmark's own name for the class this library exists for — scores
hit@5 1.000** [0.90, 1.00] (36/36) on the comparable arm, alongside single-session-assistant and
-preference at 1.000, multi-session 0.983, single-session-user 0.969 and temporal-reasoning 0.922.

It is also **the most robust category under haystack pressure**: from Oracle to the merged corpus
it retains 74% of its hit@5 while the overall figure retains 51% and single-session-user retains
30%. The plausible mechanism — a knowledge-update session contains an explicit revision, which is
lexically distinctive and survives the sparse leg, whereas a preference mentioned in passing is not
— is a hypothesis. The retention numbers are not.

**Four things 0.970 is not.** (1) A *retrieval* figure — whether the evidence session came back in
the top 5 — **not** the benchmark's LLM-judged answer accuracy, and it does not belong in a column
with one. (2) The merged arms are *harder* than the published protocol, so 0.366 is a lower bound
and is not comparable in that direction either. (3) **Temporal-reasoning is not quotable from any
arm**: 3,942 of 19,195 sessions carry more than one date across haystacks and a merged corpus holds
one copy per session; the converter counts and prints this. (4) Ground truth is session-level, so a
multi-session question scores a hit on *any one* of its evidence sessions.


### 10b. The abstention layer failed here, and no available signal fixes it

False-abstain **0.481** on the comparable arm: retrieval returned the right session 97% of the time
and the trust layer then refused to answer nearly half of those. It also moves the wrong way —
false-abstain *rises* as the haystack narrows (0.328 -> 0.409 -> 0.481) while the fitted threshold
*falls* (0.752 -> 0.723 -> 0.713).

The obvious diagnosis — a misplaced threshold — was tested first and is wrong. Top-1 cosine over
all 500 questions:

| | answerable (n=470) | unanswerable (n=30) |
|---|---|---|
| q05 / q25 | 0.612 / 0.671 | — / 0.620 |
| median | 0.723 | 0.647 |
| q75 / max | 0.774 / 0.938 | 0.689 / 0.811 |

**AUC 0.753.** The unanswerable range sits almost entirely inside the answerable range. The best
threshold obtainable on these samples scores balanced error **0.285** against the shipped rule's
**0.305** — and that ceiling is *in-sample*, the very defect §2b retracted a number for, so
held-out the gap is smaller still. Driving false-abstain to 0.05 costs false-confidence of ~0.78.
**Recalibration was ruled out by measurement, not by argument.**

Six candidate signals were then measured on the same 500 questions and the same haystacks, differing
only in the signal — three relevance (`dense_top1` **0.753**, `rerank_top1` 0.742, `hybrid_top1`
0.739), one answerability (`entail_max` 0.648) and two distributional (0.579, 0.545). Full table
with intervals: [`RESULTS.md` §8](RESULTS.md).

**Nothing beat the signal already shipping.** Three structurally different relevance signals —
including a cross-encoder that reads query and document *jointly* — cluster at 0.74–0.75. The
cross-encoder's failure is the informative one: it is trained for **relevance**, it ranks relevance
well enough to reach hit@5 0.970, and it scores a topically related session that does *not* contain
the answer just as highly as one that does. **Relevance is not answerability.**

The QNLI judge — the one built-in trained on answerability rather than relevance — came in **below
plain cosine** (0.648). At its own untuned boundary it scores false-confidence 0.533, and §5 of this
document measured that same judge's residual near-miss false-confidence at **0.50** on a corpus it
had never seen. The bound transferred exactly; it was simply never good enough for this workload.
Stacked behind a lowered gate it does not beat the threshold alone either:

| configuration | false-abstain | false-confident | balanced |
|---|---|---|---|
| shipped cosine @0.713 | 0.443 | 0.167 | **0.305** |
| judge alone @0.5 | 0.321 | 0.533 | 0.427 |
| gate 0.600 + judge | 0.332 | 0.433 | 0.383 |
| gate 0.650 + judge | 0.381 | 0.233 | 0.307 |

**Sample-size limitation — n=30 unanswerable, and what that does and does not permit.** This is
the benchmark's own abstention-class size, not a sampling choice, so it cannot be re-run larger.
The three signals at 0.74–0.75 are **not** distinguishable from one another; their intervals
overlap almost entirely, so the ordering among them is noise and no claim here rests on it.

What the sample *does* support is the only conclusion drawn from it: **none of them reaches the
~0.90 a usable abstention gate needs.** The best signal's interval tops out at **0.826**, and the
bar sits outside it — a measured *exclusion*, not a small-sample shrug.

_That depends on the interval being computed correctly, and an earlier revision got it wrong in the
conservative direction: it used `sqrt(A(1-A)/n_min)` (~0.08), which ignores the 470-sample
answerable class and is not the standard error of an AUC. The Hanley & McNeil (1982) estimator over
both classes gives **0.037** — the intervals tabulated above. The wrong figure was 2.1× too wide,
enough to put 0.90 back inside the interval and downgrade this finding to "unproven" when the data
in fact excludes the bar. The estimator now ships as `recall.calibration.separability_interval`, so
this table and the library's own certification read from one function, pinned by a test._

The interval is also why the certification rule tests the **lower bound** rather than the point
estimate. At the 20-samples-per-class minimum this module accepts, a measured AUC of 0.95 carries a
lower bound of 0.879: it clears the bar on the point and has not established it. Certifying on the
point would readmit, through small-sample noise, exactly the silent failure §10b exists to expose —
the same defect as the in-sample fit §2b retracted a number for, wearing a different hat.

**§9c closes the obvious escape hatch.** The natural objection to the row above is that only the
*shipped* QNLI judge was tried, and a better one might separate what it cannot. That was tested
independently on LOCOMO: `qnli-electra-base` dominates the shipped `qnli-distilroberta` across
nearly the whole curve and lifts best separation 0.197 → 0.240, and the best point measured still
refuses **43.8%** of legitimate questions to catch 67.7% of adversarials. A stronger same-task judge
moves along the curve; it does not lift it over the line. Two harnesses, two benchmarks, and the
"just use a bigger judge" answer fails on both.

### 10c. Why abstention works elsewhere — the bounded domain

This is not a contradiction of §2, §4 or §8; it is their boundary, located. §9b reached the same
boundary on LOCOMO and named the residual *architectural* — separating "what did Caroline realize"
from "what did Melanie realize" is entity-level reasoning the retrieval path excludes by design.
This section is the same finding stated in terms of the signal rather than the architecture.

Where abstention was measured to work — PEPs accuracy **1.00**, the private memory corpus **0.89**,
the 14-document corpus 2/2 — the unanswerable queries are **genuinely off-topic**, and the two
cosine distributions are disjoint (bge-small: answerable 0.70–0.90 against unanswerable 0.51–0.64).
That is the **far-gap** class, and the calibrated threshold handles it.

LongMemEval's abstention questions are **near-miss by construction**: the haystack is the user's own
conversation history and the question asks about something never mentioned but topically adjacent.
That is §5's class, at 30 instances instead of 10, with the same outcome.

**So the honest scope of the abstention guard is: far gaps, yes; near-misses, no.** This document
already said "abstention quality is bounded by the embedder" (§2) and "the near-miss class needs a
judge" (§5). This section adds the part that was missing: *the judge this repo ships is not good
enough for it either*, and no cheaper signal is.

### 10d. What was changed as a result — a diagnosis, not a retune

Nothing in the abstention path was tuned, because every alternative measured worse. The defect
worth fixing was not that abstention fails on this workload; it is that it failed **silently**.
`best_threshold` bisects overlapping distributions — it always did, and its docstring always said
so — and then returns a number indistinguishable from a working threshold.

`from_samples` now reports **`separability`** (the Mann-Whitney AUC of the two calibration classes,
deliberately threshold-free so it cannot be inflated by fitting and scoring on the same samples)
along with the class counts, and `Calibration.certified` is tri-state: `False` when the classes
overlap (AUC < 0.90) or a class has fewer than 20 samples (§6), and `None` — never `True` — when
there is nothing to judge or the artifact predates the check. It warns from the library, records the
verdict in the saved artifact, and makes `recall calibrate` exit non-zero.

**It changes nothing at runtime**, by design and by test: the threshold, the scale and the
confidence mapping are identical whether or not it certifies. A gate that also moved the boundary
would replace one invisible failure with another.

Reproduce:

```bash
python -m recall.eval.longmemeval --dataset longmemeval_s_cleaned.json --out ./s_out
python -m recall.eval.labelled --corpus ./s_out/corpus --questions ./s_out/questions.json
python -m recall.eval.longmemeval_perq --questions ./s_out/questions.json --master <indexed-table>
```

## 11. Reranking: the largest single retrieval gain measured here — and it was already shipped

§9a reported hit@5 **0.671** against hit@20 **0.855** and left the implication unexamined: for
**85.5%** of questions the correct turn was *already retrieved* and merely ranked below position 5.
That is not a retrieval failure, it is a ranking failure, and it is the failure a cross-encoder
exists to fix. This library has shipped one since 0.2 and no LOCOMO figure had ever been measured
with it.

**Turning it on moves hit@5 from 0.671 to 0.777** ([`RESULTS.md` §11](RESULTS.md)) — intervals
disjoint from the baseline through k=10 at n=1 536. That closes **57%** of the distance to the
pool's own ceiling, and it is roughly **twice** the largest embedder effect this project has
measured (the cloud-vs-local median of +0.059 across 17 corpora, §8's restatement).

Three things make it credible rather than merely large:

1. **hit@20 barely moves** (0.855 → 0.870). Reranking reorders a fixed pool, so the pool's own
   coverage must stay put — and it does. A gain that also lifted k=20 would have meant something
   other than reranking had changed.
2. **The gain decays with depth exactly as the mechanism predicts**: +0.155 at k=1, +0.106 at k=5,
   +0.016 at k=20. Reordering can only act where the truncation bites.
3. **A second, unrelated cross-encoder reproduces it.** `bge-reranker-base` — 12× the parameters,
   four years newer, multilingual — lands *within noise* of the shipped model at every depth
   (0.7734 vs 0.7767 at k=5). Two models that different agreeing that closely says the effect
   belongs to **reranking**, not to a model choice.

### Every category gains, including the one that usually does not

cat3, the multi-hop floor, goes **0.478 → 0.533**. That is worth stating because the prediction
registered before the run said the opposite: a pointwise cross-encoder scores one document against
the query and cannot combine evidence across turns, so extra ranking quality "should not" help
multi-hop. It helps. §10c's boundary — that the retrieval path cannot *represent* multi-hop
reasoning — remains true and was over-applied: **ranking the single most relevant turn correctly
still helps a question whose full answer needs several.** Those are two different claims.

### Why it stays optional

**It costs about 1 050 ms per question** on CPU, roughly 4.2× the wall clock of the whole benchmark
run. That is a real trade, not a rounding error, and it falls entirely on query latency — indexing
is untouched.

So reranking is **off by default and one flag away**, which is the honest arrangement when a feature
buys a large quality gain at a large latency cost. A library that silently made every query four
times slower to improve a benchmark would be optimising for the benchmark.

The decision it implies, though, is not symmetric:

- **If you are answering a human's question**, ~1 s is usually invisible next to the LLM call that
  follows, and +0.106 hit@5 is a different quality of answer. Turn it on.
- **If you are serving high-volume automated retrieval**, or running on constrained hardware, the
  4× is the dominant cost and the baseline is already competitive. Leave it off.
- **`ms-marco-MiniLM-L-6-v2` is the right model**, and that is now measured rather than assumed:
  `bge-reranker-base` is statistically indistinguishable from it at **6.3× the per-query cost**.
  Reranker choice here is about task match — short query against short passage — not model size.

**Abstention is unaffected** (0.00 on all three arms, n=446). Reranking reorders what retrieval
returned; the trust layer's verdicts sit downstream and were measured to confirm it.

_An earlier version of these numbers was produced against a corpus that had been indexed twice by
two concurrent runs (11 764 rows against a correct 5 882). Every depth came in ~0.05 low and cat3
appeared unmoved at +0.011, which would have published a false limitation. `run_conversation` now
refuses to index over an existing tenant, and the runner asserts the row count before any result is
read. See `docs/RESEARCH_PROTOCOL.md`._
