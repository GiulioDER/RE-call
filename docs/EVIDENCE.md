# Evidence

What RE-call claims, what was measured to support it, where each claim stops, and what was
withdrawn. Companion to [results/RESULTS.md](../results/RESULTS.md) (every number and its command)
and [results/FINDINGS.md](../results/FINDINGS.md) (what each one means).

## The problem, precisely

**Most RAG hands back the closest vector match. That's the wrong answer more often than you'd think.**

A long-running agent piles up memory — decisions, closed experiments, incident notes — and then it
**re-litigates settled decisions**, **hallucinates over gaps** the memory can't fill, and **builds on
facts that are no longer true**. The catch: when you've reversed a decision, the *stale* memory of it
is often the **highest-cosine hit in the whole result**. Similarity search serves it, confidently.

RE-call scores that ordering differently. A hit is ranked on whether it is still *true* — declared
supersession, expiry, and a calibrated confidence — not on distance alone, so the corrected memory
outranks the one it replaced no matter which is closer. When nothing clears the bar, the result is
an abstention rather than the nearest match.

## See it in one command

<details>
<summary>same run, as text</summary>

```text
$ python -m recall.cli demo

[ok] query='how many requests per second can a client make?'
  ok           conf=1.00  cos=0.784  rate_limits_v2.md                       '# API rate limits (revised)'
  superseded   conf=1.00  cos=0.806  rate_limits_v1.md → use rate_limits_v2  '# API rate limits … limited to 100'

[ABSTAIN · gap] query='how do we handle penguins on mars?'
  reason: no hit above the calibrated confidence threshold (probable corpus gap)
```
</details>

Look at the cosines. The **stale** memory scores **higher (0.806)** than the current one — plain vector
search returns it, and the agent builds on a limit that no longer exists. RE-call flags it
`superseded`, points at its successor, and puts the *current* memory on top. When the memory genuinely
has no answer, it says so. **That ordering decision is the whole thesis.**

## What is actually verified

Every headline number below was measured, and every one carries its limit. Where a claim could not be
supported, it was withdrawn rather than softened — the withdrawals are listed too, because a claims
table without them is marketing.

| Claim | Measurement | Limit |
|---|---|---|
| **Supersession beats similarity — where the edge was authored** | Superseded-trust rate **0.00**, 95% Wilson **[0.00, 0.02]**, n=250, against a baseline of **1.00** — plain search returns the stale memory *every time* on adversarially-worded queries | Generated corpus; the successor/abstain columns on it are **not** meaningful ([why](PRODUCTION.md#what-this-does-not-do)). **And the mechanism is only as good as its coverage: 2 of 792 real memos declared `supersedes:` while 60 closed a decision in prose** — the enforcement is exact, the corpus is sparse, and both halves are load-bearing ([prior art](PRIOR_ART.md)) |
| **Abstention is calibrated, not guessed** | On the real corpus: threshold **0.728 ± 0.042** over 4 index rebuilds, false-abstain **0.015**, gap false-confidence **0.000** | Needs ≥ ~20 labelled samples; below that the rule loses its outlier robustness |
| **Timestamps cannot replace declared supersession** | "Trust the newest relevant hit", steelmanned, still trusts the stale memory **83–100%** of the time | — |
| **Reranking rescues a weak embedder** | Hybrid + cross-encoder lifts MRR **0.63 → 1.00** offline | Situational: a strong embedder already saturates this corpus |
| **Fine-tuning pays only for a vocabulary gap** | **+0.00** on a rich corpus; **0.31 → 0.55** held-out MRR on opaque jargon → [study](RAG_TRAINING_STUDY.md) | Measure your gap first |
| **Near-misses need a judge, not a threshold** | QNLI stage cuts near-miss false-confidence **0.70 → 0.30** (hashing) and **1.00 → 0.50** (bge-small), same judge across embedders, no per-embedder retuning → [RESULTS §3](../results/RESULTS.md) | Judge-alone *degrades* far-gap detection — the two stack, neither replaces the other. Costs ~0.1–1.0 s per query |
| **Retrieval, on a second public benchmark** | **knowledge-update 1.000** (36/36) — the category this library exists for, and the most robust one under haystack pressure (retains 74% of hit@5 across a 20× larger corpus where the overall figure retains 51%). Overall **hit@5 0.970** [0.94, 0.99] on LongMemEval's own per-question haystacks with the *free local* embedder → [FINDINGS §10](../results/FINDINGS.md) | A *retrieval* figure — evidence session in the top 5 — **not** the benchmark's LLM-judged answer accuracy. It does not belong in a column with one. **And 0.970 is the benchmark's ~49-session haystack, not a memory store: on one merged 19,195-session index the same questions score 0.366.** Both arms are published because the second is the one that looks like production |
| **Abstention has a bounded domain** | Far gaps: accuracy **1.00** (PEPs), **0.89** (real corpus). Near-misses: **it fails** — false-abstain **0.481** on LongMemEval, and **six** candidate signals all score AUC ≤ 0.753 — the best one's 95% interval tops out at **0.826**, below the ~0.90 a usable gate needs, so the bar is *excluded* rather than merely unproven. Independently corroborated on LOCOMO, where no judge configuration crosses into usable territory either → [FINDINGS §9–§10](../results/FINDINGS.md) | Nothing was retuned, because every alternative measured *worse*. `recall calibrate` reports separability **with its interval**, certifies on the interval's lower bound, and exits non-zero rather than certify a threshold the data cannot support |
| **Abstention holds up under an external judge — and retrieval was not the cap** | On **MTRAG**, IBM's 842-task multi-turn RAG benchmark scored by *its own* `gpt-4o-mini` judge: **16/55 = 29.1%** of unanswerable tasks correctly refused, **second of ten**, tied with `llama-3.1-70b`, where the two that outscore us end to end refuse 12.7% and 5.5%. RE-call's contexts beat the benchmark's own retrieval **+0.0011**, consistently signed across two prompts → [MTRAG_BENCHMARK.md](MTRAG_BENCHMARK.md) | ⛔ **We do not top this benchmark**: end-to-end **−0.0064** against the published `gpt-4o`. The harness is not the excuse — on gold contexts we score 0.6195 against their 0.6208, gap **0.0013**. Baselines are **recomputed**, so every comparison is an anchored lift and none may be quoted against the public leaderboard. **And the abstention rate is a property of the generator prompt, not of retrieval**: one prompt change moved correct abstentions 89% → 64% while still lifting the aggregate |
| **Free to write — and faster** | No LLM at ingest (Mem0 runs one extraction call per session): **0 LLM calls / $0** to build memory, measured **~4.3× faster to build** and **~26% faster per query** vs Mem0 on LOCOMO, same local embedder → [benchmarks/REVIEW.md](../benchmarks/REVIEW.md) | One head-to-head (2 conversations, single run); the retrieve CI is optimistic (repeated queries) and the backends differ (Postgres vs in-process Qdrant), so treat retrieval speed as directional — ingest speed and the $0/0-calls cost are the robust part |

Full methodology, per-embedder tables and the negative results → **[results/FINDINGS.md](../results/FINDINGS.md)**.
Design rationale and the reasoning behind each guard → **[docs/WRITEUP.md](WRITEUP.md)**.

## Claims that were withdrawn

A previous version of the README published each of these. They did not survive re-measurement:

- **"FCR @calibrated 0.00"** — the threshold was fitted and scored on the same samples. On separable
  data that is 0.00 by arithmetic. Now cross-validated, and the fitting rule was
  [replaced outright](../results/FINDINGS.md) after it proved to let **20.5%** of unanswerable queries through.
- **Coverage and abstention accuracy on generated corpora** — the "unanswerable" queries were an
  answerable query plus a nonsense suffix, so nothing could separate them. Rebuilt as genuinely
  off-topic questions; the *document*-level degeneracy remains and is stated as unmeasured.
- **"6× faster incremental re-index"** — understated. Measured on a Linux server it is **33×**.
- **Real-corpus recall@5 of 0.945**<!--@ withdrawn: docs/EVIDENCE.md 'Claims that were withdrawn' --> — that used document *headings* as queries, which is known-item
  retrieval. Against 110 hand-labelled questions phrased the way a person actually asks, hit@5 is
  **0.33** on that corpus. → [FINDINGS §7](../results/FINDINGS.md)
- **"Retrieval is the weakest part of this system"** — the sentence this project carried after that
  measurement. A replication on a public corpus scored **0.705** with the same local embedder, so
  0.33 was a property of *that corpus*, not of this software. Corrected rather than quietly
  deleted, because the claim was published. → [FINDINGS §8](../results/FINDINGS.md)
- **"ANN recall is tuned on the filtered path"** — the heading this project gave the HNSW fix, which
  reads as a recall improvement. Two measurements were taken and only the flattering one reached
  the docs: a fixture corpus moved 0.36–0.43 → 0.88–0.94, while an independent A/B on a
  normally-built corpus moved recall the *other* way (0.523 → 0.483). What was actually fixed is
  **truncation** — filtered search returning fewer results than requested. Reworded above, and
  corrected in FINDINGS §5b, rather than deleted.
  → [#57](https://github.com/GiulioDER/RE-call/pull/57)
- **"The collapse needs rows committed across several transactions"** — the mechanism this repo
  published for that pathology, attributing it to pgvector building a less well-connected graph.
  It is a **statistics race**: an unanalyzed table takes an exact `Seq Scan` plan, never consults
  the HNSW index, and reports recall 1.0000 under any `ef_search`. A single-transaction 20,000-row
  upsert reproduces the collapse just as hard once the table is analyzed. The batching was winning
  the race, not shaping the graph. → [#98](https://github.com/GiulioDER/RE-call/pull/98)
- **LOCOMO "hit@5 0.615"**<!--@ withdrawn: docs/EVIDENCE.md 'Claims that were withdrawn'; results/FINDINGS.md 9a --> — published as the pre-fix retrieval anchor, and as one of two runs whose
  spread was read as HNSW build noise. **Still withdrawn, but no longer for the original reason.**
  It was removed because its result artifact had never been retained; that artifact was committed in
  [#111](https://github.com/GiulioDER/RE-call/pull/111) and records **0.6152** at k=5
  (`results/locomo_fastembed_k5.json`). What the artifact does not repair is the claim it was used
  for: reading its spread against 0.624 as HNSW build noise, when the two runs differ in *candidate
  pool*, not in index build. So it is now checkable and still not evidence for that claim. The
  pre-fix anchor remains **0.624** at k=5, 0.798 at k=20.
  → [FINDINGS §9a](../results/FINDINGS.md),
  [`results/ARTIFACTS.md`](../results/ARTIFACTS.md)
- **"Per-question raw dumps are published"** — said of the Mem0 head-to-head. They are not:
  `benchmarks/results/` is gitignored, so each run writes them locally only. The harness, the
  pre-registration, the blind human labels, the corrupt-key list and an independent adversarial
  recompute of every cell *are* published, in `benchmarks/` on master. Corrected in
  [RESULTS §9](../results/RESULTS.md) and FINDINGS §9d.

## Retrieval quality: it depends on your corpus, and here is the rule

110 hand-labelled questions per corpus, phrased the way a person asks rather than as document
headings, on the **same** held-out split throughout. Two corpora, one embedder swap:

| hit@5 | bge-small (local, free) | voyage-3 (cloud) | Δ |
|---|---|---|---|
| private memory corpus — internal codenames, project shorthand | 0.348 [0.23, 0.49] | **0.630** [0.49, 0.76] | **+0.282** |
| **public Python PEPs** — ordinary technical prose | **0.705** [0.56, 0.82] | 0.727 [0.58, 0.84] | +0.022 *(n.s.)* |

> ### ⚠️ This rule was restated on 2026-07-27. Read this before the two rows above.
>
> The original rule read: *"pay for a cloud embedder only when your corpus vocabulary is unusual."*
> It was drawn from those two corpora, and **it does not hold**. On **17 held-out** BEIR /
> CQADupStack corpora — none of which produced the hypothesis — the cloud embedder wins **16 out of
> 17**, median **+0.059** hit@5 (dense-only **+0.105**), sign test **p = 0.00027**, 95 % CI
> **[+0.038, +0.068]**.
>
> **What predicts the gap is corpus SIZE, not unusual vocabulary**: median **+0.013** below 10 000
> documents against **+0.062** at 17 000+. The PEP corpus above is 746 documents — smaller than
> anything in that study — so its +0.022 is the small-corpus regime, not a property of "ordinary
> English". An out-of-vocabulary rate, the mechanism originally proposed, predicts **nothing**
> (Holm-adjusted p = 0.65).
>
> **The rule as it now stands:** a cloud embedder buys little on a few hundred documents and about
> **+0.06 hit@5** at twenty thousand — worth weighing against ~5× query latency, an API dependency
> and your documents leaving your infrastructure. And the cheapest way to predict your own case is
> not a corpus statistic: **measure your local embedder on ~30 labelled questions.**
>
> Full study, per-corpus table, confounds and limits →
> [`results/gap/FINDINGS-embedder-gap.md`](../results/gap/FINDINGS-embedder-gap.md)

### So which configuration should you actually run?

RE-call is meant to be configured, not merely installed. Legal constraints, hardware, latency
budgets, retrieval quality, and per-query cost are first-class deployment choices. The default is
the conservative one: local, offline, and free at the memory layer. Hosted embedders, rerankers,
learned sparse retrieval, and stricter calibration gates are opt-in policy choices, not hidden
defaults.

Two starting points cover most deployments, and the honest answer is that most people should start
with the first.

**Default — free, local, offline, nothing leaves your machine.** No flags. `bge-small` via
fastembed, hybrid dense+sparse, no reranker. $0 per memory at any scale, and the configuration
every number in this file was measured on unless stated otherwise.

**Best measured quality** — when retrieval accuracy is worth an API dependency and ~1 s per query:

```bash
python -m recall.eval.locomo --data locomo10.json \
    --embedder voyage:voyage-4-large --candidate-k 250 --rerank
```

| knob | why | measured |
|---|---|---|
| `voyage-4-large` | best embedder we have measured | wins **16/17** corpora, median **+0.059** hit@5; **+0.282** on a jargon-heavy corpus; gap **grows with corpus size** (+0.013 under 10k docs → **+0.062** at 17k+) |
| `--rerank` | largest single retrieval gain in this project | hit@5 **0.671 → 0.777** (n=1,536). Costs ~1,050 ms/query, which is why it is off by default |
| `--candidate-k` **above** `--k` | **without this the reranker does nothing** | at `candidate_k == k` the pool, the returned set and your context are the same memories, so reranking reorders a list you were going to get anyway. Widening it changed the returned set on **100%** of questions (mean Jaccard 0.372) |

Two things worth knowing before you copy that line:

- **`voyage-3` is legacy.** It is still `VoyageEmbedder`'s default, has no free tier, and is
  superseded by `voyage-4-large` — which carries a **200M tokens/month free tier**, so for most
  corpora the cloud embedder costs nothing.
- **`--candidate-k` tops out near 250.** The HNSW scan is widened to `candidate_k × 4` and pgvector
  caps `hnsw.ef_search` at 1000. Past that the pool is still honoured, a `RuntimeWarning` says the
  over-fetch margin was reduced, and retrieval still covers your `k`.

**Calibrate every immutable generation before trusting its abstention gate.** A threshold is valid
only for the exact tenant, generation, pipeline, corpus, and labelled query set on which its scores
were measured. Reusing labels on a new generation reruns every retrieval score. A legacy
`calibration.json` has none of those bindings, so search never selects it automatically. See the
[calibration operations guide](CALIBRATION.md).

**On this corpus the pipeline was not the cap.** Three other levers were tested one at a time on the
same questions and none moved it: cross-encoder rerank +0.065 *(n.s., 57× latency)*, chunk size
400/800/1600 **+0.000**, candidate pool 20 → 100 *(n.s.)*. Abstention accuracy held at 0.89–1.00
throughout — the trust layer was never the bottleneck on either corpus.

> ### ⚠️ The rerank null did NOT generalise — corrected 2026-07-27
>
> That +0.065 came from **110 questions** and was not significant at that size. On **LOCOMO,
> n = 1 536**, reranking is the **largest single retrieval gain this project has measured**:
> **hit@5 0.671 → 0.777**, intervals disjoint from the baseline through k=10 — roughly **twice** the
> best embedder effect (§8's +0.059 median across 17 corpora). It lifts every question category,
> including the multi-hop floor (0.478 → 0.533).
>
> It stays **off by default** because it costs ~**1 050 ms/query** on CPU (≈4× wall clock), and it
> is one flag to turn on. Worth it when a human is waiting for the answer; leave it off for
> high-volume automated retrieval or constrained hardware. `ms-marco-MiniLM-L-6-v2` is the right
> model — `bge-reranker-base`, 12× larger and four years newer, is statistically
> **indistinguishable** at 6.3× the per-query cost.
>
> Numbers → [`RESULTS.md` §11](../results/RESULTS.md) · meaning →
> [`FINDINGS.md` §11](../results/FINDINGS.md)

> **One claim here was withdrawn.** The pool null was read as "bigger pools cannot help". It could
> not have detected a pool effect at all — `hnsw.ef_search` capped the dense leg at 40, and RRF
> fuses round-robin so a top-5 reads ~3 ranks into each leg whatever the pool is. Where the
> comparison has power (FinanceBench, n=150) a bigger pool **did** help: 0.393 → 0.527. The recall
> ceiling is real and the embedder is what moved it — but the pipeline was never actually ruled out.
> → [FINDINGS §7](../results/FINDINGS.md)

### Against a baseline — because 0.705 means nothing on its own

A hit@5 is only a result next to what a boring baseline scores on the *same* corpus, chunks and
questions. On the PEPs, bge-small, 44 held-out answerable questions:

| arm | hit@5 | MRR | p50 | reading |
|---|---|---|---|---|
| **BM25** (Okapi, untuned) | 0.455 [0.32, 0.60] | 0.313 | 150 ms | the thirty-year-old anchor |
| sparse only (Postgres FTS) | 0.023 [0.00, 0.12] | 0.023 | 24 ms | near-useless alone on this corpus |
| dense only (pgvector) | 0.682 [0.53, 0.80] | 0.483 | 31 ms | carries almost all of the result |
| **hybrid** (dense + sparse + RRF) | **0.705** [0.56, 0.82] | 0.494 | 26 ms | the published number |

<sub>The `p50` column is pre-fix: measured with a percentile index one rank too high (see the
`latency_ms` entry under Fixed in `archive/CHANGELOG_FULL.md`). A re-run returns the next sample
down whenever the scored sample size is even, which it is here. The hit@5, interval and MRR columns
are unaffected: the fix is to the latency percentile alone.</sub>

**The pipeline beats BM25 by +0.25**, so the embedding stack earns its keep — and **dense is doing
the work**: hybrid's +0.023 over dense-alone is inside the interval. On ordinary prose the fusion
barely moves the top-5; its value is on the rare identifiers a memory corpus has and this one does
not. (BM25's tokeniser has no stemming while the FTS leg does, so it is mildly handicapped on
morphology — noted in `recall/eval/bm25.py`; it does not move the +0.25.)

Reproduce the public half end to end — corpus, questions and ground truth are all public:

```bash
git clone --depth 1 https://github.com/python/peps
python -m recall.eval.labelled --corpus peps/peps --questions recall/eval/peps_questions.json --glob '**/*.rst'
```

## Reproduce everything else

```bash
make eval                                        # ablations + trust + near-miss → results/
python -m recall.eval.scale --embedder hashing --filler 50000    # scale + latency
```

→ Every number, its command and its evidence tier: **[results/RESULTS.md](../results/RESULTS.md)**.
What each one means and where it stops: **[results/FINDINGS.md](../results/FINDINGS.md)**.
