# Instrument status — which abstention claims are checkable

Written 2026-07-28 against `origin/master` @ 9eb3bc1. Read from the artifacts in `results/`,
not from `FINDINGS.md`'s account of itself — **except** the rows marked **current, unretained**
below, for which no artifact exists at all; those are read from `FINDINGS.md`/`RESULTS.md` prose
directly, and flagged as such rather than silently treated as artifact-backed.

**current** — measured on the shipped pipeline, artifact retained under `results/`.
**current, unretained** — measured on the shipped pipeline, and nothing about that code path has
changed since, but no artifact under `results/` records the run; the numbers exist only as prose
in `FINDINGS.md` / `RESULTS.md`. Cheap to reproduce is not the same as already reproduced and
saved — nobody has re-run these since Phase 0 (this cycle) added the machinery to record one.
**stale** — measured on a superseded configuration.
**unfalsifiable** — no artifact retained; the claim cannot be checked at any cost short of a re-run.

| claim | status | artifact | notes |
|---|---|---|---|
| §2 fixed gap threshold (0.50) does not transfer across embedders | current, unretained | — | FCR at the `DEFAULT_GAP_THRESHOLD` fallback — still `0.50` in `recall/guards.py` today — measured per embedder (hashing-64, bge-small, voyage-3\*); reproducible via `recall.eval.calibrate.calibrate()`. Numbers live only in `FINDINGS.md`'s table, no JSON retained (\* voyage-3 row is v0.1-era, not re-runnable key-free) |
| §5 entailment near-miss judge (3 local arms) | current, unretained | — | `threshold` / `threshold+entail` / `entail-only` per embedder, table in `RESULTS.md` §3. `recall/trust.py` confirms `recall[entail]` is still OFF by default — the claim describes an optional, still-shipped path. The one voyage-3 row is v0.3-era, marked not re-runnable key-free. No JSON retained |
| §5b scale — STR-trust holds at 40x query volume | current | `scale/SCALE.md`, `scale-pressure/SCALE.md` | STR-trust 0.00 is the citable claim (Wilson [0.00, 0.02], n=250, Arm A) — retained in both `SCALE.md` files. **`FINDINGS.md` §5b itself disclaims the abstention-accuracy column** (0.00 / 0.86 / 0.99 across the four runs) as a generator artifact — near-duplicate synthetic documents defeat the embedder, not a trust-layer measurement — so treat that column as not-yet-measured here too, per `FINDINGS.md` §6 |
| §6 abstention threshold: measured, and rebuilt | current, unretained | — | **load-bearing** (see gate below): produced the shipped `recall.calibration.best_threshold()` midgap-q05/q95 rule — confirmed current in code, 2026-07-28. Verified end to end: threshold 0.728 ± 0.042, false-abstain 0.015, gap FCR 0.000 (bge-small, 5,450 chunks, 4 fresh HNSW builds). No JSON retained under `results/`; the sweep exists only as a table in `FINDINGS.md` |
| §7 private-corpus abstention | current, unretained | — | abstention accuracy 0.89 [0.57, 0.98], n=9. Corpus is private, so not independently checkable by anyone outside this repo either way — but that is a separate fact from artifact retention, and no JSON is retained under `results/` regardless |
| §8 PEP abstention | current, unretained | — | abstention accuracy 1.00 (11/11), both embedders (`FINDINGS.md` §8, ~line 551). Public corpus and questions, cheap to re-establish: `python -m recall.eval.labelled --corpus peps/peps --questions recall/eval/peps_questions.json --glob '**/*.rst'`. Cheap does not mean done — `recall/eval/labelled.py` has no `provenance_block` call (Phase 0 did not wire it; see "Known gaps"), so even re-running today would not produce a retainable, row-count-bearing artifact |
| §9b LOCOMO abstention, 4 modes | current | `locomo/postfix_abstention.json` | post-#81/#84. `locomo_abstention.py:160` (line numbers are against `9eb3bc1`, this document's baseline — see line 3) passes `calibration=cal` explicitly, so #101's auto-load bug never reached it (design spec §3) |
| §9b abstention with rerank on | **unmeasured** | — | #103 measured the default mode only (0.00, unchanged, confirmed identical across `baseline.json`/`rerank_modern.json`/`rerank_shipped.json`). The calibrated and judge modes have never been crossed with a reranker |
| §9c entailment ROC sweep | stale, no retained artifact | — | the re-run (`postfix_entailment_sweep.log`) died after 9 conversations (conv-26, 30, 41, 42, 43, 44, 47, 48, 49); no JSON was written and nothing noticed. That log is gitignored (`results/locomo/*.log`) and worktree-local — not present in this checkout, not retained in git. The original pre-#81/#84 measurement has no retained raw artifact either |
| §10 LongMemEval, all rows | **unfalsifiable** | — | pre-#81/#84; indexes and output discarded. 6h39m to rebuild the merged index alone |
| every row above | **no row count on any existing artifact** | — | no artifact retained under `results/` before this cycle's Phase 0 (2026-07-28) records the corpus row count it measured. Phase 0 wired `recall/eval/locomo.py`, `locomo_abstention.py` and `locomo_entailment_sweep.py` (`grep provenance_block` confirms all three). It did **not** wire `recall/eval/labelled.py` (§7, §8) or `recall/eval/longmemeval_perq.py` (§10) — a future artifact from either producer will still carry no row count. See "Known gaps" below |

## What this gates

No combined signal, entity-mismatch feature or abstention-policy change is fit against a row that
lacks a retained artifact — marked **stale**, **unfalsifiable**, or **current, unretained** —
until that row is re-measured with an artifact retained, or explicitly demoted. This is written to
bind **§6** specifically: the shipped midgap-q05/q95 threshold rule is exactly the row a new
abstention signal or threshold change would be fit against, and until this row existed the gate
had nothing there to bind.

## Known gaps

Recorded here so they stay visible in the artifact itself, not only in a review transcript.

- **The provenance-wiring test cannot distinguish real wiring from a discarded call.**
  `tests/test_eval_provenance.py::test_every_locomo_runner_embeds_the_provenance_block` asserts
  only that the string `provenance_block(` appears in a module's source — it would pass even if
  the call's return value were assigned and never used. That is the same shape of failure this
  whole cycle exists to prevent: a check that looks like coverage and is not.
  `locomo_entailment_sweep.run()` can be exercised end to end in roughly 0.2s by passing
  `judges=()`; that test has not been written.
- **`recall/eval/locomo_abstention.py` still hardcodes the literal `"locomo_chunks"` twice** — once
  in the `PgVectorStore(...)` call, once in the `provenance_block(...)` call — and the tenant
  formula `f"locomo-{sample_id}"` now exists independently in three modules, each with its own
  discipline around keeping it in sync.
- **`recall/eval/labelled.py` and `recall/eval/longmemeval_perq.py` were not wired with
  `provenance_block` in this cycle** — confirmed by `grep provenance_block` returning nothing in
  either file. They produce §7/§8 and §10 respectively (above), so a future artifact from either
  will still carry no row count; the "no row count" row above cannot become "every row now carries
  one" until these two are wired the same way the three LOCOMO runners were.
- **No artifact or document records the expected clean LOCOMO row count.** It exists in exactly
  one place in this repo: `EXPECTED_ROWS=5882` in `scripts/run_locomo_arms.sh`. Without a
  reference value stated here, a reader who sees `corpus_rows: 11764` in some future artifact has
  no way to tell — from this document alone — that the number is wrong.
