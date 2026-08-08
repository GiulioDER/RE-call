# Handoff prompt: multi-query diversity + nested RRF for RE-call

Paste everything below the line into a new session. It is self-contained.

---

## Task

Investigate, implement and measure **multi-query diversity with nested RRF** as a retrieval lever
for RE-call, on the MTRAG-human dev split. Decide with statistics whether it ships.

## Why this lever, and why now

The chain is measured, not borrowed:

1. **Coverage is the bottleneck.** MTRAG-human dev, 777 judged queries. R@100 was 0.687 with a
   full-pool ceiling of 0.7365, against a ~0.95 saturation threshold above which the bottleneck
   is known to shift to ranking. → `project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06`
2. **SPLADE learned sparse was built and measured (2026-08-06).** It moved reranked R@100 to
   **0.7599, +0.0331 paired, CI [+0.0202, +0.0457], p=0.0002, Holm-significant.** Coverage
   improved, not solved. → `project-recall-splade-learned-sparse-measured-2026-08-06`
3. **The reranker lever is CLOSED, falsified the same day.** Of 123 gold documents reachable only
   via SPLADE, `ms-marco-MiniLM-L-6-v2` (22M) buries 90 below rank 10; `BAAI/bge-reranker-v2-m3`
   (568M, 25x larger) buries **91**. Median rank 29 → 31. Two cross-encoders 25x apart bury the
   same documents, so it is not a capacity problem: those documents are hard to rank from a
   `(query, passage)` pair, which is all a cross-encoder sees.
4. ⇒ **The remaining lever is the QUERY.** MTRAGEval's rank-1 system (AILS-NTUA, nDCG@5 0.578)
   used five complementary reformulations fused with a variance-aware nested RRF and reports
   **no-rewriting R@5 0.483 → nested RRF 0.607, +25.7%**. Their stated principle: *"query
   diversity over a well-aligned retriever is more effective than heterogeneous retriever
   ensembling."*

## ⚠️ Read these before designing anything

- **Do NOT score on MTRAG-UN.** It is the sealed held-out set the official leaderboard used and
  the archived 2026-08-04 Task A baseline already consumed. Dev is MTRAG-human. The harness
  defaults to `--split dev` and a test asserts that default.
- **Single gold rewriting is already measured and is WEAK: +0.0321, INCONCLUSIVE** (CI straddles
  its preregistered bar). Gold is a ceiling, so an LLM rewrite cannot beat it. **This is the
  strongest reason to believe the lever is FUSION OF DIVERSE QUERIES, not rewriting quality.**
  Design the experiment to separate those two things.
- **Feeding more conversation context HURTS: −0.0972.** Concatenating prior user turns is
  measured and negative. Do not reach for it.
- **`hybrid_both` (three legs) was measured worse than `hybrid_splade` (two).** Adding retrievers
  hurt, consistent with rank 1's own report. The lever is query-side, not more legs.
- **No paid API work on RE-call** (standing user decision). LLM reformulations must come from a
  local/rented model, not a commercial API. Budget accordingly, or use the no-LLM path below.
- ⚠️ Two figures about rank 1 were CORRECTED after reading the PDFs: **"HyDE alone +20.9%" is
  false** (their table is cumulative; best single strategy is Corpus-Specific, 0.541), and the
  saturation threshold is **~0.95, not >0.90**. Do not re-import the wrong versions.

## 🔑 There is a FREE first experiment. Do it before renting anything.

MTRAG-human ships **three aligned query files per domain with matching ids**:

    mtrag-human/retrieval_tasks/<domain>/<domain>_lastturn.jsonl    last user turn
    mtrag-human/retrieval_tasks/<domain>/<domain>_questions.jsonl   full-conversation concat
    mtrag-human/retrieval_tasks/<domain>/<domain>_rewrite.jsonl     GOLD human rewrite

So you can run **multi-query fusion with ZERO LLM calls and ZERO GPU**: treat those three as
three reformulations, retrieve with each, fuse with RRF, and measure.

That isolates the mechanism. If fusing three genuinely different queries lifts R@100, the lever
is fusion and an LLM only needs to supply *variety*. If it does not, no amount of LLM rewriting
will help, and the lever is dead for a few hours of CPU instead of a GPU rental and a week.

Note the three are known individually: lastturn is the baseline, gold rewrite is +0.0321
inconclusive, full concat is −0.0972. **Two of the three are individually weak-or-harmful. If
their FUSION beats the best of them, that is the finding**, and it is exactly what "variance-aware
nested RRF" is supposed to buy.

## State: everything is built and staged

**Branch `feat/splade-sparse-leg`** (pushed to origin, 17 commits). Local worktree
`C:/Users/gde00/Documents/recall-splade`. Contains the SPLADE leg, migrations 0012/0013, the
offload harness and the contrast analysis.

**VPS2** `root@100.91.148.25`, key `~/.ssh/contabo_sentiment`:

| path | what |
|---|---|
| `/var/tmp/re_call_splade_20260806/RE-call/` | checkout at the branch tip |
| `/var/tmp/re_call_splade_20260806/.venv/` | venv: torch cpu, transformers, fastembed, sentence-transformers 5.7.0, psycopg |
| `/var/tmp/re_call_splade_20260806/splade.env` | `RECALL_DSN` for the `recall_splade` DB (0600) |
| `/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark/` | the corpora and qrels |
| `/var/lib/recall-benchmarks/2026-08-06-mtrag-splade-learned-sparse/` | archived run, 29 files, SHA256 manifest |

**Database `recall_splade` on VPS2** (deliberately NOT `sentiment_agent`, which is the money-path
DB — do not run DDL there):
- `recall_mtrag_bge_v1_{clapnq,cloud,fiqa,govt}` — dense bge-small, 366,479 passages
- `recall_sparse_v1` — SPLADE vectors, profile `prithivida__Splade_PP_en_v1`, HNSW built
- ⚠️ Query-side embeddings were verified to reproduce the stored vectors at **cosine 1.00000000**.
  If you change the venv, re-verify before measuring.

**Also staged, unused:** `/var/tmp/re_call_splade_20260806/vectors_v3.tar.gz` — a second SPLADE
encode with `naver/splade-cocondenser-ensembledistil` (MRR@10 38.3 vs 37.22), NOT loaded.
⚠️ cc-by-nc-sa-4.0, non-commercial, research only.

## The baseline to beat

Reranked with BGE-reranker-v2-m3, 777 dev queries:

| arm | nDCG@5 | R@100 |
|---|---|---|
| `hybrid_lexical` | 0.3911 | 0.7269 |
| **`hybrid_splade`** | **0.3931** | **0.7599** |

Raw (no reranker), which is the cleaner target for a coverage lever:

| arm | nDCG@5 | R@100 |
|---|---|---|
| `hybrid_lexical` | 0.2926 | 0.6872 |
| **`hybrid_splade`** | **0.3573** | **0.7377** |

**R@100 is the metric this lever must move.** Coverage is the bottleneck; nDCG@5 is secondary and
the reranker has already been shown not to convert extra coverage.

## Method requirements (project standards, non-negotiable)

- **Freeze arms before observing any score.** Declare them in code, as `SPARSE_ARMS` does.
- **Paired bootstrap CI (n>=2000) + sign-flip permutation (n=5000) + Holm-Bonferroni at 0.05.**
  `benchmarks/mtrag/analyse_contrasts.py` already implements this; reuse it rather than
  rewriting. → `reference-validation-standards`
- **Check the deciding cell has power BEFORE preregistering.** Compute n and the achievable range
  of the cell that decides the contrast. A prior session built three guards that could not fire,
  could not pass, or rested on n=8. → `feedback-check-the-deciding-cell-has-power-2026-08-06`
- **A point estimate is not a result.** Today a −0.0043 "regression" was reported, then retracted
  when its CI turned out to span zero.
- **Write the predicted outcome down before running.** Today's reranker null is only worth its
  GPU time because the interpretation was pre-registered in the runbook.

## Traps that cost real time on 2026-08-06

1. **`ef_search` silently truncates.** `_query_learned_sparse` returned **6 of 100** candidates
   until `hnsw.ef_search` was widened 10x. `_query_dense` has its own widening. **Any new
   retrieval path must check it returns the k it asked for** — no test caught this, a timing
   anomaly did.
2. **`|user|: ` prefixes every MTRAG-human turn.** `strip_speaker` in `benchmarks/mtrag/run.py`
   removes it. Feeding it to an encoder depresses everything silently.
3. **`pkill -f <pat>` matches its own ssh command line** and kills the shell running it. Use
   `[p]at`.
4. **Heredocs over ssh mangle quotes.** Use `scp` for anything non-trivial.
5. **`scp` takes `-P`, `ssh` takes `-p`.**
6. **Bulk-load then build the index.** Inserting into a live HNSW ran 25 rows/s; dropping the
   index first gave 580 rows/s.
7. **Check liveness of the JOB, not one of its stages.** A tarball got hashed mid-write because
   the encoder process had exited while `tar` was still running.
8. **Another session shares VPS2.** Check load before long runs; coordinate before killing
   anything.

## Deliverable

A decision, backed by paired CIs with Holm correction, on whether multi-query diversity ships —
plus an archived run under `/var/lib/recall-benchmarks/` with a SHA256 manifest and a `NOTE.md`
carrying the caveats, matching the 2026-08-06 archive's shape.

Start with the free three-file fusion experiment. Only rent a GPU if that shows the mechanism
works and you need an LLM to generate more variety.
