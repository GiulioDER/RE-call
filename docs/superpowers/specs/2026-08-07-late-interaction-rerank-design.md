# Late-interaction reranking on MTRAG-human dev: preregistration

Status: **PREREGISTERED**. Written 2026-08-07, before any score from this experiment was observed.

Everything below the "Predicted outcome" heading is a commitment. Results go in a separate
`RESULTS` section appended after the run, and the arms, contrasts and decision rule in this file
are not edited afterwards.

## The question

Three retrieval levers have now been measured on MTRAG-human dev and all three returned the same
shape: coverage rises, ranking does not follow.

| lever | coverage | ranking |
|---|---|---|
| SPLADE learned sparse | R@100 +0.0331, Holm-significant | nDCG@5 +0.0020 |
| Multi-query diversity (`mq_nested3`) | R@100 +0.1236, Holm-significant | nDCG@5 +0.0054, p=0.057, **does not convert** |
| Bigger cross-encoder (MiniLM 22M → BGE 568M) | n/a | buries the same documents |

The third is the load-bearing one. Of 123 gold documents reachable only via SPLADE,
`ms-marco-MiniLM-L-6-v2` (22M) buries **90** below rank 10 and `BAAI/bge-reranker-v2-m3` (568M,
25x larger) buries **91**. Median rank 29 → 31. Two cross-encoders 25 times apart in capacity bury
the same documents, which is why the conclusion drawn was not "we need a bigger reranker" but:

> those documents are hard to rank from a `(query, passage)` pair, which is all a cross-encoder
> sees.

**That conclusion names an architecture, not a model.** Every reranker measured so far pools the
pair into a single representation before scoring it. Late interaction does not: ColBERT encodes
query and document independently to per-token vectors and scores by MaxSim, so per-token evidence
survives to the scoring step instead of being pooled away.

This experiment asks one question:

> **Does late interaction rank the documents that pooled-pair cross-encoders bury?**

It is a measurement, not a shipping proposal. Nothing here touches a migration, a sidecar table, or
the erasure paths. Serving late interaction is a separate project gated on this verdict.

## Why this is not another coverage lever

The distinction matters because the three prior nulls make a fourth null the default expectation,
and it would be easy to read this as more of the same.

It is not. The prior levers all **widened the candidate pool** and asked the reranker to convert
the extra coverage. This one holds the retrieved set **byte-identical** and changes only the
scoring function, so no arm here can add a document the others did not have.

⚠️ **That does not make R@100 invariant, and it would be easy to write as though it did.** The
whole pool (200 for `hybrid_splade`) is reranked and then truncated to the metric cutoff, so
reordering changes *which* 100 survive. The 2026-08-07 run measured this directly: whole-pool
reranking **destroys** coverage in proportion to depth (`mq_last` at 200 gained +0.0226 R@100,
`nested3` at 547 lost −0.0513). R@100 is therefore a **veto** here, not a target: retrieval
coverage is held fixed, and any R@100 movement is the scorer discarding coverage it was handed.

A null here consequently means something different from the previous nulls. It means the ranking
failure is not attributable to pair-pooling either, and the lever class closes rather than one more
instance of it.

## Scope, and what is deliberately not in it

**In scope.** A reranker that implements the existing `Reranker` protocol, an offloaded scorer that
reuses the existing dump, and a preregistered verdict.

**Out of scope, explicitly.** A `recall_late_v1` sidecar, any migration, any retrieval-side use of
late interaction as a first-stage retriever, and any change to `search()` or `search_fused()`
defaults. `recall/retriever.py:118` refuses the learned sparse leg under `RECALL_ENV=production`
because `recall_sparse_v1` has no foreign key to the chunk table and the erasure paths
(`generations.forget`, `control_plane.erase_sources_from_pending`) predate it, so a forgotten chunk
would leave reconstructable weights behind. **A late-interaction sidecar is the same shape and
strictly worse on that axis**: per-token vectors carry more reconstructable content than pooled
term weights. That problem is real, it is unsolved, and this experiment does not go near it.

## Licence containment

`li_jina` is `cc-by-nc-4.0`. The precedent for handling this is
`naver/splade-cocondenser-ensembledistil`, which was encoded, staged on VPS2, and **deliberately
not loaded** over its `cc-by-nc-sa-4.0` licence. Here the arm is wanted, so the risk is contained
mechanically rather than by intention.

Four gates, all in `benchmarks/mtrag/late_interaction.py`:

1. **Registry.** Every checkpoint carries its licence, mirroring `recall/sparse.py:131`. An
   unrecorded checkpoint raises, with `sparse.py:190`'s wording: an unrecorded licence is exactly
   what the check exists to prevent.
2. **The Holm family builder raises when handed a non-deployable arm.** This is the load-bearing
   gate. The verdict that gates the follow-on project is computed from a family `li_jina` cannot
   mechanically enter. Enforced by refusal, not by a docstring, for the reason `search_fused` gives
   for its own refusals.
3. **Every emitted record carries `checkpoint`, `licence`, `deployable`.** Numbers do get lifted
   out of these archives into later documents. A lifted number arrives with its taint attached
   rather than as a bare float.
4. **Runtime opt-in.** `li_jina` requires `--accept-noncommercial` or it refuses to run, mirroring
   `accept_noncommercial_license=True` in `sparse.py:195`.

Plus one interpretive rule, fixed here before any score exists:

> **`li_jina`'s effect is monotone.** It answers exactly one question: is a shared null a capacity
> result? It can strengthen a null or weaken a positive claim. It can never support a decision to
> build the follow-on project.

That monotonicity is what actually sorts the risk. Even if a jina number leaks into a later
document, the only claim it can support is a negative one.

### A correction to the precedent being copied

`recall/sparse.py:195` gates on `license_id != "apache-2.0"`, which would refuse an **MIT**
checkpoint. MIT is compatible with RE-call's own MIT distribution, so that check is wrong in
principle. It is **latent, not live**: `KNOWN_MODELS` contains no MIT entry, so it cannot fire
today, and fixing it is out of scope here.

The late-interaction registry gates on a permissive **set** (`{mit, apache-2.0}`) rather than
string equality. Copying `sparse.py` verbatim would have refused `colbert-ir/colbertv2.0`, the
primary arm, under its own guard.

## Architecture

Four pieces, three of them new.

**`recall/rerank.py` gains `LateInteractionReranker`.** Implements the existing `Reranker`
protocol. Encodes query and passages separately, scores by MaxSim, reorders, and ends with the same
line `CrossEncoderReranker.rerank` ends with:

```python
return [hits[i] for i in order]
```

**Reorder only.** Every hit keeps its dense cosine `score`. This is not a style choice:
`recall/trust.py:292` thresholds on `hit.score` and `recall/trust.py:536` passes it to
`cal.confidence()`. A MaxSim score is an unbounded sum over query tokens in entirely different
units, so leaking it into `score` would corrupt the calibrated confidence of every hit that reaches
the trust layer. `rerank.py:84` already documents this hazard for the cross-encoder; the same
constraint binds here and is pinned by a test.

**`benchmarks/mtrag/late_interaction.py`** holds the arm registry, the licence gates, and the
MaxSim scorer.

**A scorer that reuses the existing dump verbatim.** `rerank_offload.cmd_dump` already emits
`queries.jsonl`, `docs.jsonl`, `pairs.jsonl` and `pools/<arm>.jsonl`. The late-interaction scorer
reads those same files and writes the same `{qid, doc_id, score}` lines `_load_scores` already
parses. `cmd_apply`, `rerank_order` and `analyse_contrasts.py` are reused untouched.

The consequence is the cleanest available isolation of the claim under test: **identical pools,
identical tie rule, identical metrics, and the only thing that varies is the score source.**

**A `validate` command**, mirroring `rerank_offload.cmd_validate`. It runs the real
`LateInteractionReranker` locally on a sample and requires the offloaded ordering to match.

### The design decision that matters most: stream the documents

A cross-encoder runs one forward pass per pair. The 2026-08-07 run scored 241,270 of them and
rented an RTX 5090 to do it.

Late interaction encodes queries and documents **independently**, which changes the cost structure:

- Encode 777 queries once. Their token embeddings are a few MB. Hold in RAM.
- Stream `docs.jsonl`. Encode each document once, MaxSim it against only the queries whose pairs
  reference it, emit those scores, **discard the token matrix.**

Materialising document token embeddings would cost roughly 7 GB at 128 dims (unique documents times
~180 tokens times 128 floats). There is no reason to hold them, and streaming makes peak memory
independent of corpus size.

**Consequence: this probably does not need a GPU rental.** It is one encode pass over the unique
documents rather than 241,270 joint forward passes. That is an expectation about the arithmetic,
not a measured runtime, and it is recorded here as a prediction rather than a plan assumption.

## Dependency route

`fastembed` is already a declared extra (`pyproject.toml:55`, `fastembed>=0.3`), already in
`uv.lock` at 0.8.0, and already exposes late interaction. Verified in the project venv:

| checkpoint | params | dim | licence |
|---|---|---|---|
| `colbert-ir/colbertv2.0` | ~110M | 128 | mit |
| `answerdotai/answerai-colbert-small-v1` | ~33M | 96 | apache-2.0 |
| `jinaai/jina-colbert-v2` | ~560M | 128 | cc-by-nc-4.0 |

`pylate` is rejected on the reasoning `pyproject.toml:84` already records for SPLADE: it is built
on `sentence-transformers`, and routing a retrieval experiment through it would raise the
**reranker's** ST floor as a side effect. That note says to bump ST for rerank, entail and finetune
together, so this experiment must not force it unilaterally.

## Arms

All arms score the **same frozen `hybrid_splade` pools** at identical width. Pool width alone moves
reranker results here (`closed-hypothesis-recall-rerank-pool-interaction-2026-08-05`: on PEPs the
same MiniLM got worse as the pool widened, entire 95% CI below threshold). Holding the dump fixed
removes that confound by construction rather than by care.

| arm | checkpoint | role | deployable |
|---|---|---|---|
| `rr_none` | n/a | raw fused order, floor | n/a |
| `rr_minilm` | `ms-marco-MiniLM-L-6-v2` | **control**, the shipped reranker | yes |
| `rr_bge` | `BAAI/bge-reranker-v2-m3` | strongest available cross-encoder | yes |
| `li_colbertv2` | `colbert-ir/colbertv2.0` | **primary** | yes |
| `li_answerai` | `answerai-colbert-small-v1` | secondary permissive | yes |
| `li_jina` | `jina-colbert-v2` | capacity diagnostic | **no** |

`rr_minilm` is the control because the 2026-08-07 preregistration established the principle: **the
shipped model decides.** When MiniLM and BGE disagreed there, the shipped model's verdict stood and
the disagreement itself was recorded as the finding.

`rr_minilm` and `rr_bge` are already measured (0.7603 / 0.3769 and 0.7599 / 0.3931 whole-pool at
pool 200) and are recomputed here only as a reproduction gate.

## Contrasts, decision rule, and the Holm families

### Family A: ranking, 777 paired queries, metric nDCG@5

| id | contrast |
|---|---|
| **C1 (primary)** | `li_colbertv2 − rr_minilm` |
| C2 | `li_answerai − rr_minilm` |
| C3 | `li_colbertv2 − rr_bge` |

### Family B: the mechanism, the 123 buried gold documents

| id | contrast |
|---|---|
| **D1 (primary)** | rescue rate, `li_colbertv2` against `rr_minilm`: how many of the 123 reach the top 10, where MiniLM buries 90 and BGE buries 91 |
| D2 | same for `li_answerai` |

Two families rather than one, declared in advance and reported separately. They answer different
questions on different units (queries against documents), and merging them would let a strong
mechanism result mask a weak ranking result. **The cost is that the overall error rate across both
families exceeds 0.05**, and that is stated rather than hidden inside one incoherent family.

### Family C: the serving-path guard

`search_fused` shipped in PR #235 and **refuses to run without a reranker**
(`recall/retriever.py:359`), because `mq_nested2_nogold` is −0.0447 nDCG@5 raw and +0.0084
reranked. That refusal is a claim about **MiniLM specifically**, since MiniLM is what the arm was
measured with.

| id | contrast |
|---|---|
| **V1 (veto)** | `li_colbertv2` against `rr_minilm` on `mq_nested2_nogold` pools, nDCG@5 |

⚠️ **V1's pool width is read from the archive, never assumed.** The 2026-08-07 run used an
equal-width protocol precisely because width alone moves reranker results, and `mq_nested2_nogold`
was reranked at a different natural depth from `mq_last`. V1 must reuse that run's recorded width
so it is comparable to the +0.0084 figure it is checking. If the width cannot be recovered from the
archive, V1 is not run rather than run at a guessed width.

V1 is a **veto on the follow-on project, not a contributor to this one's verdict.** If a
late-interaction reranker were to replace MiniLM in serving while V1 is negative, `search_fused`'s
refusal message would assert a gain that no longer holds. A positive C1 with a negative V1 means
the lever is real and the serving path needs remeasuring before it can adopt it.

### Decision rule, fixed before any score

- **POSITIVE**, follow-on project justified: C1 point estimate **≥ +0.010 nDCG@5** and
  Holm-significant within Family A, with no veto tripped.
  The 0.010 bar is **copied from the 2026-08-06 multi-query rerank preregistration**, not chosen
  now, so it carries no information from this experiment.
- **Vetoes**, tripped on "CI excludes zero" rather than on Holm, because a veto should be easy to
  trip while a ship bar should not: a regression in R@100 (the whole pool is reranked and then
  truncated, so coverage can be lost) or in nDCG@10.
- **NULL**: C1 below the bar, or its CI includes zero.
- **The capacity reading**, which is `li_jina`'s only permitted use: if C1 and C2 are null **and**
  jina is also null, the pooled-pair conclusion is confirmed on a second architecture and late
  interaction closes as a lever. If jina alone is positive, capacity matters, no permissive
  checkpoint of that size exists, and that is a research finding rather than a ship path.

  ⚠️ **The capacity spread here is weaker than the one that closed the reranker lever, and the
  conclusion must be scaled to it.** That lever was closed on two cross-encoders **25x** apart
  (22M against 568M). These three span 33M / 110M / 560M, so the widest ratio is **17x** (jina
  against answerai) and the ratio against the **primary** arm is only **5x** (560M against 110M).
  A shared null across all three is therefore weaker evidence than the MiniLM/BGE null was, and it
  licenses "capacity does not appear to be the binding constraint over 33M to 560M", not "capacity
  is ruled out". No stronger phrasing is permitted in the write-up.

## Power: does the deciding cell have any?

**Family A has power.** The 2026-08-07 harness resolved +0.0054 at p=0.057 on these same 777 paired
queries, so it discriminates effects around the 0.010 bar.

**Family B is NOT certified, and is not preregistered until it is.** n=123 with 90 buried under the
control is a paired binary design, and its minimum detectable shift must be computed **before the
arms freeze**, not after. The standard is explicit about this failure mode:
`feedback-check-the-deciding-cell-has-power-2026-08-06` records a prior session that built three
guards which could not fire, could not pass, or rested on n=8.

**Precondition task, blocking the freeze:** re-derive the 123-document set from the archived
`/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/` run on VPS2, compute the minimum
detectable rescue shift under a paired binary test, and record it in this file before any arm runs.

**If Family B is underpowered it is demoted to a descriptive diagnostic with no p-value attached**,
and the verdict rests on Family A alone. That consequence is fixed now so it cannot be argued after
seeing the counts.

## Validation gates, run before any contrast is computed

**G1. Reproduction.** `rr_minilm` recomputed through the new code path must reproduce the archived
figures to four significant figures (0.7603 R@100 / 0.3769 nDCG@5). The 2026-08-07 run already
demonstrated this is achievable across different hardware and a different code path, so a failure
means the new path is wrong rather than that reproduction is hard. **If G1 fails, nothing else is
read.**

**G2. Offload validation.** The offloaded ordering must match a live `LateInteractionReranker` on a
sample. Non-optional, for the reason `rerank_offload` gives: numbers from an unvalidated substitute
are not results.

**G3. Score completeness.** Every pair in `pairs.jsonl` must receive a score and every document in
every pool must be scored. This generalises the `ef_search` lesson: `_query_learned_sparse`
silently returned **6 of 100** candidates until `hnsw.ef_search` was widened 10x, no test caught
it, and a timing anomaly did. The failure mode here is quieter still, because a missing score does
not raise, it sinks that document to the bottom of the ranking. The gate asserts counts rather than
assuming them.

**G4. Encoder identity.** Query-side and document-side encodings must come from the same checkpoint
and the same `fastembed` version, both recorded in the artifact header. Mirrors the SPLADE run's
cosine 1.00000000 verification and its standing warning to re-verify after any venv change.

**G5. Mutation check.** Replace MaxSim's `max` with `mean`, confirm G2 goes red, revert. The
`search_fused` plan carried this same check, and it is warranted: this line of work has already
produced two guards that could not fire.

One trap that does **not** apply, recorded so nobody re-solves it: `|user|: ` prefixes every
MTRAG-human turn and depresses encoders silently. Because this reuses the existing dump rather than
re-running retrieval, `strip_speaker` has already been applied upstream.

## Tests

The load-bearing one:

- **Score preservation.** Build hits with distinct known scores, rerank, assert every output hit
  carries the exact score it entered with and that the multiset of scores is unchanged. This is
  what fails if a MaxSim value ever leaks into `hit.score` and reaches `cal.confidence()`.

The rest:

- Reorder-only: output is a permutation of the input, same length, same chunk ids.
- Empty hits returns empty, matching `CrossEncoderReranker`.
- Tie stability: equal scores preserve fused order, since `rerank_order` relies on a stable sort.
- MaxSim correctness against a hand-computed tiny example, pinning the metric independently of any
  model download.
- Registry refusals: unknown checkpoint raises; `cc-by-nc-4.0` raises without
  `--accept-noncommercial`; **MIT is accepted**, which is the corrected gate and would fail under
  `sparse.py`'s equality check.
- The Holm-family builder raises when handed a non-deployable arm.

## Predicted outcome

Written before running, per the standard that today's nulls are only worth their compute because
the interpretation was fixed in advance.

**P1. C1 is positive but small: +0.005 to +0.015, straddling the 0.010 bar.**

**P2. D1 is clearly positive**, conditional on Family B having the power to say so.

**P3. V1 is smaller than C1**, because the fused pool already contains history-driven candidates
that a token-level matcher handles no better than a pooled one.

**P4. No GPU rental is needed.** Streaming encode over unique documents completes on CPU within a
working day.

Reasoning for P1 and P2, including the part I distrust: the 123 are documents SPLADE found and both
cross-encoders buried, so they are lexically anchored but rank badly from a pooled representation.
MaxSim keeps per-token evidence instead of pooling it, which is the specific deficiency involved.

**The distrust:** those documents are *already* lexically matched, and ColBERT's token maxima may
reproduce the same lexical judgment SPLADE made, in which case it ranks them no better and both P1
and P2 fail together. If that happens, the correlated failure is itself the finding, and it is a
stronger result than either prediction landing.

These are directional predictions rather than hedges, because the 2026-08-06 deployable-fusion run
recorded exactly that lesson: naive arithmetic predicted −0.024, the result came in at −0.0231, and
the hedge was the error.

## Method requirements

- **Freeze arms before observing any score.** Declared above as a frozen tuple in code, as
  `SPARSE_ARMS` is.
- **Paired bootstrap CI (n≥2000) + sign-flip permutation (n=5000) + Holm-Bonferroni at 0.05.**
  Reuse `benchmarks/mtrag/analyse_contrasts.py` unchanged rather than rewriting it.
- **Do NOT score on MTRAG-UN.** It is the sealed held-out set. Dev is MTRAG-human and the harness
  defaults to `--split dev`.
- **No paid API work**, standing user decision. Every checkpoint here is a local model.
- **A point estimate is not a result.** Every figure reported with its CI.

## Deliverable

A verdict on C1 with paired CIs and Holm correction, plus an archived run under
`/var/lib/recall-benchmarks/YYYY-MM-DD-mtrag-late-interaction/`, named for the date the run
executes, with a SHA256 manifest and a `NOTE.md` carrying the caveats, matching the shape of the
2026-08-06 and 2026-08-07 archives.
