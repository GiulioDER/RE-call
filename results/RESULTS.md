# RE-call — measured results, complete

Every evaluation this project has run, in one file. Interpretation and history live in
[`FINDINGS.md`](FINDINGS.md); this file is the numbers, each section with the command that
reproduces it.

**Local vs cloud, up front.** Rows marked **local** run entirely on your hardware — the corpus,
the queries and the index never leave your machine, and the marginal API cost is $0
(`hashing-64` needs no model at all; `bge-small`/`bge-large` are downloaded once and run offline).
Rows marked **cloud** use the Voyage API: better on some corpora (measured below, §4), but every
document and query is sent to a third party, and the row only reproduces with `VOYAGE_API_KEY`
set. RE-call itself never requires the cloud path — it is an option, not a dependency.

## 1. Core retrieval ablation — embedder × fusion (14-doc corpus)

Reproduce the local (key-free) rows with `make eval` — needs Docker + the local embedder only. The
Voyage cloud row appears when `VOYAGE_API_KEY` is set.

> **Provenance.** Regenerated 2026-07-25 on PostgreSQL 16.14 / pgvector 0.8.5 / Python 3.12.3, in
> one run, after the [#81](https://github.com/GiulioDER/RE-call/issues/81) sparse-leg fix. Two things
> follow, and they have different causes:
>
> - **Quality metrics moved because of the fix.** The `hybrid` rows rose (hashing-64 MRR 0.737 →
>   0.964); `dense` is unchanged, as it must be. The sparse leg previously ANDed every query term,
>   so it fired only when one chunk contained all of them.
> - **Latency moved because of the host.** The previous table was measured on a different machine
>   (PostgreSQL 17 / pgvector 0.8.2). Rerank ms/query going 691.7 → 2383.0 is a slower shared CPU,
>   **not** a regression. Latency figures here are only comparable within this table.
>
> Mixing the two runs — keeping the old latency beside the new quality — would have been the
> dishonest option, so the whole file is one run.

| embedder | | fusion | P@5 | R@5 | MRR | nDCG@10 | FCR no-guard | FCR guard |
|---|---|---|---|---|---|---|---|---|
| hashing-64 | local | dense | 0.186 | 0.929 | 0.626 | 0.715 | 1.00† | 0.20 |
| hashing-64 | local | hybrid | 0.200 | 1.000 | 0.964 | 0.974 | 1.00† | 0.20 |
| hashing-64 | local | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00† | 0.20 |
| BAAI/bge-small-en-v1.5 | local | dense | 0.200 | 1.000 | 0.964 | 0.974 | 1.00† | 1.00 |
| BAAI/bge-small-en-v1.5 | local | hybrid | 0.200 | 1.000 | 1.000 | 1.000 | 1.00† | 1.00 |
| BAAI/bge-small-en-v1.5 | local | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00† | 1.00 |

_† FCR no-guard is ANALYTIC, not measured: with no gap guard the system never abstains, so every unanswerable query is answered confidently and the rate is 1.00 by definition. It is the reference point for FCR guard, not an observation._

_P@5 is mechanically capped at 0.20: each query has exactly one relevant doc, so the best possible precision@5 is 1/5. Read it as "answer found in the top 5" (binary), not as classical precision — R@5 / MRR / nDCG@10 are the informative ranking metrics._

_bge-small's FCR guard 1.00 is the **shipped-default-threshold** failure §2 of FINDINGS documents:
0.50 sits below that embedder's entire cosine distribution, so the guard never fires. Calibrated
per embedder it reaches 0.00 — the reason `recall calibrate` exists._

Cost/latency (mean wall time per call):

| embedder | fusion | embed ms/query | rerank ms/query |
|---|---|---|---|
| hashing-64 | dense | 0.1 | 0.0 |
| hashing-64 | hybrid | 0.1 | 0.0 |
| hashing-64 | hybrid+rerank | 0.1 | 1357.1 |
| BAAI/bge-small-en-v1.5 | dense | 13.3 | 0.0 |
| BAAI/bge-small-en-v1.5 | hybrid | 16.1 | 0.0 |
| BAAI/bge-small-en-v1.5 | hybrid+rerank | 39.4 | 1922.1 |

## 2. Trust layer — superseded/expired memories vs plain search

STR = superseded-trust rate: how often a stale memory was presented as the answer on the
validity-sensitive queries (lower is better). The final two columns verify the trust layer does
not change ordinary answerable retrieval.

| embedder | STR baseline | STR recency | STR trust | trust coverage | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|---|---|
| hashing-64 | 1.00 | 1.00 | 0.00 | 0.67 | 0.50 | 1.00 | 0.964 | 0.804 |
| BAAI/bge-small-en-v1.5 | 0.83 | 1.00 | 0.00 | 0.83 | 0.75 | 0.50 | 1.000 | 1.000 |

**Read STR trust together with trust coverage.** STR counts queries where a stale memory was served
with verdict `ok`, so a system that returns nothing scores a perfect 0.00. The claim is 0.00 STR
*at high coverage*; 0.00 STR at low coverage is a system that abstained its way to a good number.
`STR recency` is the steelman timestamp heuristic — "among confident hits, trust the newest" — and
it trusts the stale memory 100% of the time here: a per-document timestamp cannot see a
two-document relation.

95% Wilson score intervals for the headline rates (n in parentheses):

| embedder | STR trust | trust coverage | successor acc | abstain acc |
|---|---|---|---|---|
| hashing-64 | [0.00, 0.39] (n=6) | [0.30, 0.90] (n=6) | [0.15, 0.85] (n=4) | [0.34, 1.00] (n=2) |
| BAAI/bge-small-en-v1.5 | [0.00, 0.39] (n=6) | [0.44, 0.97] (n=6) | [0.30, 0.95] (n=4) | [0.09, 0.91] (n=2) |

At scale (synthetic corpus, §5): STR trust **0.00 [0.00, 0.02]** at coverage **1.00** over n=250 —
the headline claim, bounded.

## 3. Entailment abstention — near-miss queries (arms A/B/C)

Near-miss = a high-similarity memory that does NOT answer the query — the class a cosine threshold
passes by construction. Arms: `threshold` = calibrated cosine threshold (status quo),
`threshold+entail` = threshold plus the QNLI judge, `entail-only` = judge alone (ablation). The
judge is identical across embedders — no per-embedder recalibration. The judge-ms column averages
only over the queries the judge actually ran on (threshold-abstained queries never reach it), so in
the stacked arm it can exceed the all-queries total mean.

| embedder | arm | near-miss FCR | gap FCR | false-abstain | MRR ans | judge ms (judged calls) | total ms/query |
|---|---|---|---|---|---|---|---|
| hashing-64 | threshold | 0.70 | 0.60 | 0.29 | 0.429 | 0 | 9 |
| hashing-64 | threshold+entail | 0.30 | 0.20 | 0.64 | 0.357 | 1675 | 1167 |
| hashing-64 | entail-only | 0.70 | 0.20 | 0.07 | 0.893 | 2753 | 2772 |
| BAAI/bge-small-en-v1.5 | threshold | 1.00 | 0.00 | 0.07 | 0.929 | 0 | 41 |
| BAAI/bge-small-en-v1.5 | threshold+entail | 0.50 | 0.00 | 0.14 | 0.857 | 1791 | 1486 |
| BAAI/bge-small-en-v1.5 | entail-only | 0.80 | 0.20 | 0.07 | 0.929 | 2225 | 2287 |

The one cloud measurement on this class (v0.3 run, older query set, not re-runnable key-free):
voyage-3 near-miss FCR **0.40 → 0.40** with the judge, gap FCR 0.00 → 0.00, false-abstain
0.00 → 0.07, judge 125 ms — the strongest embedder had the least for the judge to fix.

## 4. Real corpora — where local vs cloud actually separates

Two corpora, 110 hand-labelled paraphrased questions each (half calibrate the threshold, half
score), hybrid dense+sparse, no reranker unless stated. This is the section that prices the cloud:
same pipeline, only the embedder swapped.

| corpus | embedder | | hit@5 | 95% Wilson | MRR | search p50 | index |
|---|---|---|---|---|---|---|---|
| private memory corpus (794 memos / 6,491 chunks) | bge-small | local | 0.348 | [0.23, 0.49] | 0.311 | 45 ms | 696 s |
| private memory corpus | **voyage-3** | cloud | **0.630** | [0.49, 0.76] | **0.503** | 246 ms | 224 s |
| public Python PEPs (746 docs) | bge-small | local | 0.705 | [0.56, 0.82] | 0.483 | — | — |
| public Python PEPs | voyage-3 | cloud | 0.727 | [0.58, 0.84] | 0.629 | — | — |

- **Idiosyncratic corpus (internal codenames, project shorthand): cloud is worth +0.282** — the
  intervals barely touch, the one lever of five tested that the sample resolves (vs rerank +0.065
  n.s., pool ±0.065 n.s., chunk size +0.000; FINDINGS §7).
- **Ordinary technical English: cloud is worth +0.022** — comfortably inside the noise, for ~5×
  the query latency, an API dependency, and your documents leaving your infrastructure.
- **Abstention holds on both corpora with both embedders**: accuracy 0.89 (memory corpus) and
  1.00 (11/11, PEPs); false-abstain 0.02–0.065. The trust layer was never the bottleneck.
- Post-fix arms on the memory corpus (2026-07-25, corpus grown to 824 files / 6,800 chunks,
  `--candidate-k 20`, bge-small): dense **0.326**, sparse **0.348**, hybrid **0.457**,
  hybrid+rerank **0.435** — the working sparse leg is worth ~+0.13 over dense alone.

The PEP replication is fully public — corpus, questions and ground truth:

```bash
git clone --depth 1 https://github.com/python/peps
python -m recall.eval.labelled --corpus peps/peps --questions recall/eval/peps_questions.json --glob '**/*.rst'
```

## 5. Domain fine-tuning — free lift where the vocabulary gap is

`finetune/train.py`, OnlineContrastiveLoss on (query, gold-chunk) pairs, scored on held-out
paraphrased queries. All local. Full study: [docs/RAG_TRAINING_STUDY.md](../docs/RAG_TRAINING_STUDY.md).

| corpus | base MRR | fine-tuned MRR | Δ |
|---|---|---|---|
| rich 14-doc corpus | 1.00 | 1.00 | +0.00 |
| opaque-codename corpus | 0.31 | **0.55** | **+0.24 (~+79%)** |

Fine-tuning's payoff equals the vocabulary gap between the base model and your corpus: zero on a
corpus the embedder already reads, large on jargon. It is the local-only counterpart to §4's cloud
swap — same condition (unusual vocabulary) predicts both.

## 6. Scale — synthetic corpus, two arms

`recall.eval.synthetic` generates the trust-corpus shape at arbitrary size. Details:
[scale/SCALE.md](scale/SCALE.md), [scale-pressure/SCALE.md](scale-pressure/SCALE.md).

**Arm A — interval width** (bge-small, 600 chunks, 550 queries):

| metric | rate | 95% Wilson | n |
|---|---|---|---|
| STR trust | 0.00 | **[0.00, 0.02]** | 250 |
| trust coverage | 1.00 | [0.98, 1.00] | 250 |
| successor accuracy | 0.14* | [0.09, 0.20] | 150 |
| abstention accuracy | 0.00* | [0.00, 0.04] | 100 |

_* Generator artifact, not a trust-layer measurement: every generated document is the same sentence
with a different opaque token, so these two columns measure token discrimination by the embedder.
Treat them as not-yet-measured at scale (FINDINGS §6). STR and coverage are unaffected —
supersession is a declared relation, not a similarity judgement._

**Arm B — index pressure** (hashing-64, 50,600 chunks; three runs at the same seed, spread shown
because pgvector's HNSW build is not deterministic):

| run | STR baseline | STR trust | trust coverage | p50 / p95 / p99 (ms) | index time |
|---|---|---|---|---|---|
| 1 | 0.92 | **0.00** | 0.01 | 67.2 / 196.6 / 353.9 | 221.5 s |
| 2 | 0.46 | **0.00** | 0.14 | 5.5 / 9.7 / 10.8 | 172.6 s |
| 3 | 0.82 | **0.00** | 0.14 | 18.3 / 25.0 / 29.7 | 130.7 s |

**STR trust is 0.00 in all three runs** — the claim the arm exists for. The baseline and latency
swing across identical inputs (shared dev machine + HNSW build randomness), so read them as
order-of-magnitude only. Filtered dense search returns `k` results when `k` exist
(`hnsw.ef_search=200` + `iterative_scan=relaxed_order`, pinned by
`tests/test_hnsw_filtered_recall.py`) — a correctness fix, not a quality one (FINDINGS §5b).

## 7. LOCOMO — the standard benchmark (retrieval + the axis nobody scores)

Public benchmark (10 conversations, 1,536 answerable + 446 adversarial questions). RE-call
produces **no LLM-judge (J) score** — it has no generator; these are exact retrieval and
abstention measurements of the substrate under such a system. bge-small (local), hybrid, no rerank.

### 7a. Retrieval depth curve — post-fix (2026-07-26, #81 + #84 live)

One retrieval per question, every depth scored from the same pass (top-k is a prefix of
top-max(k) by construction). n=1,536 answerable; candidate pool 20 per leg; 625 s:

| k | overall hit@k | 95% CI | cat1 | cat2 (temporal) | cat3 | cat4 |
|---|---|---|---|---|---|---|
| 1 | 0.398 | [0.374, 0.423] | 0.316 | 0.480 | 0.228 | 0.413 |
| 3 | 0.577 | [0.552, 0.601] | 0.514 | 0.651 | 0.380 | 0.591 |
| **5** | **0.671** | [0.647, 0.694] | 0.628 | 0.720 | 0.478 | 0.687 |
| 10 | 0.778 | [0.757, 0.798] | 0.731 | 0.807 | 0.554 | 0.807 |
| **20** | **0.855** | [0.836, 0.872] | 0.837 | 0.872 | 0.620 | 0.880 |

The working sparse leg is worth about **+0.05 at k=5** over the effectively dense-only
configuration the earlier runs measured (0.615/0.624 across two pre-fix builds), and +0.06 at
k=20 (0.798 → 0.855). cat3 is still the floor at every depth. Default-configuration adversarial
abstention on the same run: 0.000 [0.00, 0.01], n=446 (§7b).

Pool control (same code path, `--candidate-k 100`; 1,038 s): a deeper pool **dilutes** the fused
prefix — RRF interleaves five times as many low-rank candidates — while enabling depth past the
20-per-leg edge:

| k | pool 20 | pool 100 |
|---|---|---|
| 5 | **0.671** | 0.596 |
| 20 | 0.855 | 0.782 |
| 50 | — (beyond the pool) | **0.877** [0.860, 0.892] |

Raising `--candidate-k` is a different fusion configuration, not a deeper look at the published
one. Full analysis and the history of the earlier (invalid, retracted) pre-fix control:
FINDINGS §9a.

Historical anchor: the pre-#81 runs (sparse leg effectively inert → dense-only) measured overall
hit@5 **0.615/0.624** across two index builds — the ±0.01 build noise is documented in FINDINGS §9a.

```bash
python -m recall.eval.locomo --data locomo10.json --k-curve 1,3,5,10,20
python -m recall.eval.locomo --data locomo10.json --k-curve 1,5,10,20,50 --candidate-k 100
```

### 7b. Abstention on the adversarial 446 — the axis no published result reports

LOCOMO cat5 questions are on-topic but unanswerable (right event, wrong speaker). Default
configuration abstains on **0 of 446** — a gap threshold cannot catch an on-topic near-miss. The
shipped levers, each priced against answerable false-abstain (n=446 adversarial / 400 answerable):

| Mode | Adversarial abstain ↑ | Answerable false-abstain ↓ | discrimination |
|---|---|---|---|
| default | 0.000 [0.00, 0.01] | 0.000 [0.00, 0.01] | 0.000 |
| calibrated (in-sample = upper bound) | 0.574 [0.53, 0.62] | 0.420 [0.37, 0.47] | 0.154 |
| entailment judge | 0.347 [0.30, 0.39] | 0.290 [0.25, 0.34] | 0.057 |
| both | 0.796 [0.76, 0.83] | 0.603 [0.55, 0.65] | 0.193 |

Re-measured 2026-07-26 post-fix. Better retrieval raises **both** columns — discrimination is
0.154 against a pre-fix 0.157, i.e. unchanged. Every per-conversation calibration came back
**uncertified** (separability 0.53–0.69 vs the 0.90 bar), which is the shipped gate refusing to
bless a threshold the data cannot support.

Judge sweep (full ROC from one scored pass; **separation** = adversarial-abstain − false-abstain):

| Judge | at threshold 0.5 (adv / ans) | best operating point | best separation |
|---|---|---|---|
| qnli-distilroberta (shipped) | 0.374 / 0.263 | thr 0.95 → 0.697 / 0.500 | 0.197 |
| qnli-electra-base (stronger) | 0.511 / 0.328 | thr 0.99 → 0.677 / 0.438 | **0.240** |

No configuration is usable — the best point still refuses 43.8% of legitimate questions. The
residual is architectural (entity-level reasoning the retrieval path excludes by design), not a
tuning failure: FINDINGS §9b–§9c.

```bash
python -m recall.eval.locomo_abstention       --data locomo10.json --answerable-sample 40
python -m recall.eval.locomo_entailment_sweep --data locomo10.json --answerable-sample 40
```

## 8. LongMemEval — knowledge-update and abstention, named by the benchmark itself

`bge-small` local, 500 questions, calibrated on half, scored on half. Configuration note: measured
pre-#81, so read retrieval rows as effectively **dense-only lower bounds** (a post-fix re-score
requires rebuilding the 6h39m merged index and is tracked as follow-up; FINDINGS §10).

| | per-question (~49 sessions) | Oracle (940) | merged S (19,195) |
|---|---|---|---|
| chunks searched | 796 | 21,251 | 321,569 |
| **hit@5** | **0.970** [0.94, 0.99] | 0.719 [0.66, 0.77] | 0.366 [0.31, 0.43] |
| MRR | 0.921 | 0.577 | 0.242 |
| abstention accuracy | 0.733 | 0.733 | 0.800 |
| **false-abstain** | **0.481** | 0.409 | 0.328 |
| search p50 | 66 ms | 68 ms | 90 ms |

**knowledge-update: hit@5 1.000** [0.90, 1.00] on the comparable arm — the benchmark's own name for
the supersession class this library exists for — and the most robust category under haystack
pressure (retains 74% of its hit@5 from Oracle to merged-S vs 51% overall).

The abstention failure underneath (false-abstain 0.481) was diagnosed, not tuned away: six
candidate signals measured on the same questions —

| signal | kind | AUC | 95% CI |
|---|---|---|---|
| dense_top1 *(ships)* | relevance, bi-encoder | **0.753** | [0.680, 0.826] |
| rerank_top1 | relevance, cross-encoder | 0.742 | [0.666, 0.818] |
| hybrid_top1 | relevance, RRF | 0.739 | [0.663, 0.815] |
| entail_max | answerability, QNLI | 0.648 | [0.557, 0.739] |
| margin_1_5 | distributional | 0.579 | [0.479, 0.679] |
| ratio_1_5 | distributional | 0.545 | [0.442, 0.648] |

Nothing reaches the ~0.90 a usable gate needs (best upper bound 0.826): **relevance is not
answerability**, on this benchmark's near-miss class. Where the unanswerable class is genuinely
off-topic (PEPs, memory corpus, §4) abstention works; the boundary is the class, not the corpus
size. Since then the library *certifies* rather than pretends: `Calibration.certified` refuses
when the calibration classes overlap (AUC < 0.90) or are under 20 samples per class — it warns,
records the verdict, and changes nothing at runtime (FINDINGS §10d).

## 9. Head-to-head vs Mem0 — same generator, same judge, only the memory differs

The comparison people actually ask for. Full LOCOMO, LLM-as-judge, **paired** on identical
question sets; pre-registered before any number was seen. The harness, pre-registration,
per-question raw dumps, blind human labels and the corrupt-key list live on the
`bench/head-to-head` branch (being finalized for publication alongside the write-up).
RE-call ran **entirely local** (bge embedder, no LLM in the memory layer);
Mem0 as shipped (LLM extraction per memory written).

Answerable accuracy (n=1,540, all 10 conversations; every row survives Holm–Bonferroni, largest
adjusted p = 0.012):

| generator | budget | judge | RE-call | Mem0 | paired McNemar p |
|---|---|---|---|---|---|
| gpt-4o-mini | item-matched (k=10/10) | gpt-4o-mini | **0.416** | 0.378 | 0.0059 |
| gpt-4o-mini | item-matched | gpt-4o | **0.466** | 0.412 | 0.00018 |
| gpt-4o-mini | token-matched (k=10/20) | gpt-4o-mini | **0.416** | 0.370 | 0.00077 |
| gpt-4o-mini | token-matched | gpt-4o | **0.466** | 0.411 | 0.00018 |
| **gpt-4o** (strong) | token-matched | gpt-4o | **0.484** | 0.444 | 0.0065 |

Abstention, both columns (gpt-4o-mini generator; n=446 adversarial / 1,540 answerable):

| | RE-call | Mem0 |
|---|---|---|
| adversarial abstention (want high) | 0.883 | **0.948** |
| answerable false-abstain (want low) | **0.291** | 0.340 |
| discrimination (difference) | 0.593 | 0.608 |

Cost of the memory layer, **metered, not modelled** (per-call token accounting in the harness):

| memory-layer LLM cost | RE-call | Mem0 |
|---|---|---|
| LLM calls made by the memory layer (4 conversations) | **0** | 99 |
| tokens sent to an LLM | **0** | 985,687 |
| $ to build, gpt-4o-mini extraction | **$0.00** | $0.166 |
| $ to build, gpt-4o extraction | **$0.00** | $2.65 |
| $ full benchmark, gpt-4o tier | **$0.00** | **$7.29** (272 calls, 2.62M tokens) |

At that $7.29 tier Mem0 scored 0.444 against RE-call's 0.484 at $0 — the cost structure is
zero-marginal vs linear-per-memory, not "N× cheaper".

**Honest boundaries, from the same study:**

- **The accuracy lead is reader-conditional.** +0.054 (gpt-4o-mini reader) → +0.040 (gpt-4o) →
  **−0.043 on Claude Sonnet 4.5** (post-hoc, not pre-registered, n=584): the stronger the reader,
  the smaller the lead, until it reverses. Claimed for the OpenAI reader tier the field
  benchmarks with; disclosed where it doesn't hold.
- **The standard judge is unreliable and it doesn't rescue anyone.** On the 199
  judge-disagreement questions, blind hand-labelling found gpt-4o right 85.6% of the time and
  gpt-4o-mini 14.4% — and the error is not system-asymmetric, so the ranking holds under either.
  On the 1,369 questions where both judges agree the margin is the same: RE-call 0.440 vs
  Mem0 0.399 (+0.041, p=0.0056) — the win does not live in the contested tail.
- **Same-embedder control**: both systems on bge-large — RE-call 0.478 vs Mem0 0.370, paired
  p=0.000022 (n=584) — the gap is in how each memory's output reads, not retrieval quality.
- **LOCOMO's answer key is 6.4% wrong** (99/1,540, independently audited); excluding those moves
  both systems ~equally (+2.2 / +2.1).

### Vendor-published numbers, and why they do not slot into the table above

Mem0's 2026-07 announcement of its new algorithm ("single-pass hierarchical extraction,
multi-signal retrieval") publishes LOCOMO **92.5**, LongMemEval **94.4**, and — on its new
**BEAM** benchmark (1M/10M-token corpora) — **64.1 / 48.6**, at ~6.7–7.0k mean tokens per answer.
Three reasons those figures cannot be pasted beside this section's, none of them a criticism this
repo hasn't already applied to itself:

1. **A J score is a property of the whole (memory, generator, judge) triple.** Our own paired runs
   moved Mem0 from 0.370 to 0.444 by changing nothing but reader and judge — and the hand-labelled
   audit found the standard judge wrong on 85.6% of contested calls. A number from an unmatched
   pipeline is on a different scale, whoever publishes it.
2. **LOCOMO's corrupted answer key caps the honest ceiling near 93.6.** A 92.5 sits essentially at
   that ceiling; the independent audit (dial481/locomo-audit) also measured the common judge
   over-accepting 62.8% of deliberately wrong answers. High absolute scores on this benchmark
   measure judge leniency as much as memory quality. The published harness
   (mem0ai/memory-benchmarks, read directly from its code) makes the leniency concrete: the judge
   marks CORRECT on *one* matching item out of an N-item gold list, tolerates ±14 days on dates
   and ±50% on durations, and is instructed to use evidence "only to ACCEPT answers, never to
   reject them". The same harness **excludes LOCOMO's 446 adversarial questions from scoring
   entirely** (`CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]`) and its answer prompt forbids abstaining
   ("NEVER say 'not specified' … COMMIT AND ANSWER") — the axis this library exists for is removed
   from the metric, which is the field-wide omission §7b documents. The reader is also handed up
   to **200 memories per question** (~7k tokens) — the retrieval-budget control our paired
   protocol exists to impose.
3. **The comparison in this file is paired**: identical generator, identical judge, identical
   questions, both retrieval budgets — the only construction in which a difference is attributable
   to the memory layer. And it already covers the announced algorithm as shipped to open source:
   the paired runs used **mem0ai 2.0.13 (released 2026-07-22)**, twelve days *after* the
   announcement, which states the new algorithm was "available today on both the Mem0 platform and
   the open-source SDK". The version is recorded in every result artifact precisely so this
   question is answerable. What the pairing has *not* covered: the **Mem0 platform** arm (if the
   hosted service diverges from the SDK, that difference is unmeasured here), and **BEAM** — whose
   published harness pins gpt-4o as answerer and judge, and which includes an **abstention**
   category, the axis this library is built for. A paired BEAM run and a platform-parity check are
   queued as follow-up work rather than asserted in advance.

The honest status: **RE-call's measured edge is against Mem0's OSS SDK as shipped after the new
algorithm's announcement** — vendor-published scores from unmatched pipelines remain on a
different scale regardless.

## 10. Local vs Voyage — the decision, priced

| axis | free / local (bge-small) | cloud (voyage-3) |
|---|---|---|
| ordinary technical prose (PEPs) | 0.705 | 0.727 (+0.022, n.s.) |
| idiosyncratic jargon corpus | 0.348 | **0.630 (+0.282)** |
| LOCOMO conversational (§7a) | measured | not run at scale |
| search latency p50 | **45 ms** | 246 ms (network RTT) |
| index 6.5k chunks | 696 s (CPU) | 224 s (batched API) |
| marginal API cost | **$0, at any scale** | per-token, per query and per re-index |
| data leaves your infra | **never** | every document and every query |
| abstention accuracy | 0.89–1.00 | 0.89–1.00 (unaffected) |

The measured rule: **pay for the cloud embedder only when your corpus vocabulary is unusual**
(internal codenames, shorthand absent from pretraining) — that condition, and only that condition,
produced a resolvable gap; domain fine-tuning (§5) attacks the same gap without the API. On
ordinary English the free local stack is within noise of the paid one, and the Mem0 head-to-head
(§9) was won with the local stack end to end.
