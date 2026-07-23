# What the comments taught me: entailment vs. threshold, and write-time supersession

> Phase-0 study of the two strongest objections raised against the RE-call design — run on the
> same harness, reported whether the numbers flattered the objections or not.

Two expert comments on the self-recall writeup made claims sharp enough to test:

1. **"A similarity score is not a confidence score."** The near-misses that hurt are
   *high-similarity-and-wrong* — memories semantically adjacent to the query that do not answer
   it — and a threshold-based gap guard waves them straight through, because their similarity
   clears any calibrated threshold *by construction*. The abstention signal cannot be the
   retriever's own score; it needs a separate judgment that the memory actually **entails** an
   answer. Proximity is a candidate; entailment is the evidence.
2. **"Supersession is a relation, not a property."** Timestamps fail because you are inferring a
   relation between two memories at read time, when both look valid in isolation. Capture the
   relation at **write time** (the new memo names what it replaces) and retrieval returns the
   head of the chain. The residual failure mode — an author forgetting the link — is a much
   better place to fail, because it is *lintable*.

Both claims turn out to be substantially right. Neither turns out to be the whole story.

---

## 1. The near-miss class exists, and the threshold cannot see it (as predicted)

We added a held-out challenge set of 10 **near-miss queries**: each names a distractor document
that is strongly on-topic but does not contain the asked-for fact ("how much did the
read-through cache reduce **memory usage**" against a memo that measures *latency*; "what
**constant k** do we use in RRF" against a memo that defines the formula but never states the
constant). The set is deliberately excluded from threshold calibration — a challenge set must
not tune the guard it challenges.

Baseline (calibrated cosine threshold, the v0.2 status quo), near-miss false-confident rate:

| embedder | near-miss FCR @ calibrated threshold |
|---|---|
| hashing-64 | 1.00 |
| bge-small | 0.80 |
| voyage-3 | 0.40 |

The commenter's mechanism is confirmed: these queries score *above* the answerable floor —
the same threshold that scores FCR 0.00 on far-gap queries (FINDINGS §2) passes 40–100% of
near-misses. There is no threshold to fix; the distractor's cosine is genuinely high.

## 2. An entailment stage helps — as a *layer*, not a replacement

We added an optional entailment stage (`recall/entailment.py`, `pip install recall-rag[entail]`,
OFF by default): a QNLI cross-encoder — "does this sentence answer this question?" — judges the
verdict-ok hits, and an ok hit that does not entail the query is demoted to a new verdict
`not_entailed`. The decision happens at the judge's own trained boundary; **there is no
per-embedder constant to recalibrate**, which is the property Finding 2 said a score threshold
cannot have. Three arms, same judge everywhere, zero per-embedder tuning:

| embedder | arm | near-miss FCR | gap FCR | false-abstain | MRR ans | judge ms (judged calls) |
|---|---|---|---|---|---|---|
| hashing-64 | threshold | 1.00 | 1.00 | 0.00 | 0.696 | 0 |
| hashing-64 | threshold+entail | **0.60** | 0.20 | 0.21 | 0.714 | 856 |
| hashing-64 | entail-only | 0.60 | 0.40 | 0.07 | 0.881 | 889 |
| bge-small | threshold | 0.80 | 0.00 | 0.00 | 1.000 | 0 |
| bge-small | threshold+entail | **0.50** | 0.00 | 0.07 | 0.929 | 149 |
| bge-small | entail-only | 0.80 | 0.40 | 0.07 | 0.929 | 827 |
| voyage-3 | threshold | 0.40 | 0.00 | 0.00 | 1.000 | 0 |
| voyage-3 | threshold+entail | 0.40 | 0.00 | 0.07 | 0.929 | 125 |
| voyage-3 | entail-only | 0.80 | 0.40 | 0.07 | 0.929 | 1018 |

(Latency measured on CPU; wall-clock, so it varies run to run — the rates above are
deterministic, the ms columns are measurements. The judge column averages over the queries
where the judge actually ran — in the stacked arm, queries the threshold already abstained on
never reach the judge, which is why stacking is much cheaper than entail-only on the semantic
embedders.)

Four honest readings:

- **It works, transferably.** The stacked arm cuts near-miss FCR on the two embedders with
  headroom (1.00→0.60, 0.80→0.50) and never hurts it, *with the identical judge and no
  recalibration anywhere*. The claim "a decision, not a score, sidesteps the transfer problem"
  survives contact with the data.
- **It does not replace the threshold — the ablation kills that idea.** Judge-alone *degrades*
  far-gap detection (gap FCR 0.00→0.40 on both semantic embedders): the QNLI judge, fed
  nearest-noise from an uncovered topic, sometimes calls it an answer. The threshold catches
  far gaps; the judge catches near-misses; **they guard different failure classes and must be
  stacked**. The commenter's thesis needed this refinement.
- **The judge is the new quality bound.** The residual FCR (0.40–0.60) is all one class: the
  4–6 surviving near-misses (per embedder) ask for a *specific absent detail* (a multiplier, a constant, a
  duration, an affected-user count) inside a strongly on-topic memo, and a small QNLI model
  judges "on-topic" as "answers". Finding 2 said gap detection is bounded by the embedder;
  the same law applies one layer up — **abstention-by-entailment is bounded by the judge.**
  A stronger judge (large NLI model, LLM-as-judge) is the obvious next experiment; the harness
  now measures exactly that trade.
- **The cost is real and must be stated.** ~0.1–1.0 s of judge time per query on CPU — from
  ~1.3× total latency on voyage-3 (the judge rides on a ~300 ms cloud round-trip) through ~7×
  on bge-small to >200× on the near-instant offline hashing embedder,
  and a new false-abstain failure: the judge rejects one legitimately answerable query (1/14)
  on *both* semantic embedders — q02, whose gold memo answers by **negation** ("do we retry on
  4xx?" → "we do **not** retry on 4xx"). MRR on answerable queries dips 1.000→0.929
  accordingly. Killing near-misses is not free.

The cross-encoder *reranker* already in the stack is not this feature, and the distinction
matters: it scores query+memo jointly but only **reorders** — the score is still thresholded
downstream, so it inherits the calibration problem. The entailment stage emits a decision.

## 3. Supersession: the timestamp heuristic loses to the declared relation

v0.2 already implements the commenter's proposal — `supersedes:` frontmatter captured at write
time, chain resolution, successor promotion (FINDINGS §4). What the harness lacked was the
explicit strawman-vs-steelman comparison: *would timestamps have been enough?*

We added a recency arm: **"among the confidently-relevant hits, trust the newest"** — the
strongest reasonable version of the timestamp heuristic (pool = hits above the same calibrated
threshold the guards use; a global newest-wins would be a strawman). The stale documents are
re-indexed *after* the successors, simulating the re-sync/touch any living corpus performs
constantly. Re-indexing identical text changes only `indexed_at`, so the other measurements are
unaffected.

| embedder | STR plain search | STR recency (steelman) | STR trust layer |
|---|---|---|---|
| hashing-64 | 1.00 | 0.83 | **0.00** |
| bge-small | 0.83 | **1.00** | **0.00** |
| voyage-3 | 1.00 | 1.00 | **0.00** |

The timestamp heuristic trusts the stale memory 83–100% of the time — and on bge-small it is
*worse than plain relevance ranking* (1.00 vs 0.83): the tie-break actively promotes the
freshly-re-synced stale memo in the one case where ranking had preferred the successor. A
per-document timestamp cannot see a two-document relation; making the timestamp "smarter"
makes it more confidently wrong. The declared relation stays at 0.00 in the same runs.

## 4. The residual failure mode is now lintable (`recall lint`)

The write-time design fails exactly where the commenter said: an author who closes a decision
without declaring the edge leaves an orphan that looks valid forever. That is invisible at
read time — both memos are individually fine — but it is **checkable at write time**.
`recall lint <corpus>` (no DB, no embedder) reports:

- errors that break trust-layer correctness: `dangling-supersedes` (edge to a nonexistent
  file), `self-supersedes`, `supersession-cycle`, `invalid-date`;
- warnings that usually mean a missing edge: `version-sibling-unlinked` (`x_v1.md`/`x_v2.md`
  naming with no edge) and `closure-marker-unlinked` (the body *says* "superseded/replaced by"
  but the frontmatter declares nothing — the relation exists only in prose, where retrieval
  cannot act on it).

Exit code 1 on errors, so it drops into CI in one line. Both corpora in this repo pass clean.

But this static lint only catches edges you *wrote* (and malformed). It is structurally blind
to the edge you *forgot* entirely — a new memo that is really about a prior settled decision but
declares no `supersedes:` at all, with no closure prose to trip the keyword heuristic. That
relation is in the meaning, not the frontmatter.

## 4b. Catching the *missing* edge with retrieval (a reader's idea, built and measured)

A commenter proposed the fix I'd left on the table: run retrieval at write time. It costs
nothing new — same embedder, same index, same trust layer, pointed at the commit path instead
of the query. When a memo lands, search with its text; any high-similarity **closed decision**
it does not reference is a candidate unlinked chain. It is the write-time mirror of
anti-re-litigation: that guard queries before an agent *proposes*; this queries before a memo is
*committed*. Shipped as `recall lint --semantic` (`recall/semantic_lint.py`, opt-in, DB-backed).

The honest result has two halves. On a planted orphan — a pricing-cache revision that omits its
`supersedes:` — at bge-small's **calibrated 0.70** threshold it surfaces the predecessor cleanly
and alone (cos **0.847**, far above the noise), while the correctly-linked control pair and the
unrelated decision stay quiet. Drop the threshold to 0.60 and it drowns: unrelated pairs land at
0.60–0.63, which is exactly bge-small's unanswerable-cosine ceiling (§2). **Finding 2 bites
again, one layer up — the threshold does not transfer, it must be calibrated per embedder.**

The second half is why the human stays in the loop. Dogfooded on the (small, topically dense)
eval corpus at 0.70, it over-surfaces: two *different* cache decisions at 0.82, an incident next
to a rollout flag at 0.76, two RAG hypotheses at 0.71 — none of them missing edges, just co-valid
near-neighbours. So it is a **candidate surfacer with a one-keystroke confirm, not an
auto-linker** — precisely how the commenter framed it. Missing-edge detection is bounded by the
(embedder, threshold) pair the same way gap detection is bounded by the embedder and
abstention-by-entailment is bounded by the judge. The cost asymmetry makes over-surfacing the
right default: a false positive is a keystroke to dismiss, a false negative is a silent orphan,
so you tune it loose. It ships warnings-only (exit 0), so it never blocks CI.

## 5. What changed in the repo

- `recall/entailment.py` — `EntailmentJudge` protocol, `QnliEntailmentJudge`,
  `apply_entailment()`; `trusted_search(..., entailment=judge)` opt-in; verdict `not_entailed`.
- `recall/eval/near_miss.json` — 10 held-out near-miss queries; `run_nearmiss_eval()` scores
  the three arms; `results/nearmiss_effect.png`.
- `run_trust_eval()` — recency-steelman arm (`STR recency` column).
- `recall/lint.py` + `recall lint` CLI — static supersession-graph completeness checks.
- `recall/semantic_lint.py` + `recall lint --semantic` — retrieval-based missing-edge check
  (opt-in, DB-backed; the reader's write-time-retrieval idea, built and measured).
- All measured numbers reproduce with `make eval` (Voyage rows appear when `VOYAGE_API_KEY`
  is set).

## 6. Decision

Entailment ships **OFF by default** as an optional stacked stage — proven useful against
near-misses on this corpus, at a measured latency and false-abstain cost the caller must opt
into, with the judge (not the threshold) now the binding quality constraint. The recency
comparison closes the "why not just timestamps?" question with numbers. The lint closes the
authorial gap in write-time supersession.
