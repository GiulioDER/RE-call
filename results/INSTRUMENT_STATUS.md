# Instrument status — which abstention claims are checkable

Written 2026-07-28 against `origin/master` @ 9eb3bc1. Read from the artifacts in `results/`,
not from `FINDINGS.md`'s account of itself — **except** the rows marked **current, unretained**
below, for which no artifact exists at all; those are read from `FINDINGS.md`/`RESULTS.md` prose
directly, and flagged as such rather than silently treated as artifact-backed.

**Relation to `ARTIFACTS.md`.** [`ARTIFACTS.md`](ARTIFACTS.md) maps a committed artifact to the
configuration that produced it (one artifact → which config). This file maps a published claim to
whether it is still checkable (one claim → which artifact, if any, and its status). They are
complementary, not substitutes — a reader who has found only one of the two may not know the other
exists.

**A note on "provenance," which now names two different mechanisms in this repo.** This branch's
`recall/eval/provenance.py::provenance_block()` is a function eval runners call to stamp a *newly
produced* result with what corpus was actually measured — `corpus_rows`, `table`, `tenants`,
`git_sha`, `git_dirty` (see the "every row above" row below). Master's leading `_provenance` block,
already present inside every committed artifact `ARTIFACTS.md` indexes, instead records which
*configuration* produced that *file* — `generation`, `status`, `superseded_by`, `backs`. These are
not the same mechanism and neither implies the other: every artifact cited in the table below
carries master's `_provenance` block; as of this writing **none** carries this branch's
`corpus_rows`. Seeing one on a file is not evidence of the other.

**current** — measured on the shipped pipeline, artifact retained under `results/`.
**current, unretained** — measured on the shipped pipeline, and nothing about that code path has
changed since, but no artifact under `results/` records the run; the numbers exist only as prose
in `FINDINGS.md` / `RESULTS.md`. Cheap to reproduce is not the same as already reproduced and
saved — nobody has re-run these since Phase 0 (this cycle) added the machinery to record one.
**stale** — measured on a superseded configuration.
**stale, artifact retained** — measured on a superseded configuration, but (unlike bare **stale**)
a JSON artifact recording exactly what was measured is retained under `results/`; a reader can open
it and check the published numbers against the file, even though the configuration it describes is
no longer what ships and no re-measurement on the shipped configuration exists.
**unfalsifiable** — no artifact retained; the claim cannot be checked at any cost short of a re-run.

| claim | status | artifact | notes |
|---|---|---|---|
| §2 fixed gap threshold (0.50) does not transfer across embedders | current, unretained | — (bge-small row only, see note) | FCR at the `DEFAULT_GAP_THRESHOLD` fallback — still `0.50` in `recall/guards.py` today — measured per embedder (hashing-64, bge-small, voyage-3\*); reproducible via `recall.eval.calibrate.calibrate()`. Numbers live only in `FINDINGS.md`'s table, no JSON retains the 3-embedder table as a whole (\* voyage-3 row is v0.1-era, not re-runnable key-free). **Partial exception, checked 2026-07-28:** `cosine/distributions.json`'s `reference_answerable_vs_far_gap` pair retains the raw bge-small cosines this row's bge-small entry describes — answerable 0.702–0.900, far-gap 0.507–0.641, matching this table's "0.70–0.90" / "0.51–0.64" to the same rounding (also quoted at `FINDINGS.md` §10c). hashing-64 and voyage-3 remain unretained, so the claim as a whole — a 3-embedder comparison — stays **current, unretained** |
| §5 entailment near-miss judge (3 local arms) | current, unretained | — (bge-small geometry only, see note) | `threshold` / `threshold+entail` / `entail-only` per embedder, table in `RESULTS.md` §3. `recall/trust.py` confirms `recall[entail]` is still OFF by default — the claim describes an optional, still-shipped path. The one voyage-3 row is v0.3-era, marked not re-runnable key-free. No JSON retains the FCR table itself. **Partial exception, checked 2026-07-28:** `cosine/distributions.json`'s `reference_answerable_vs_near_miss` pair (n=14/10 — the same held-out near-miss set this row cites) retains the raw bge-small cosines and their separability (0.850, CI [0.695, 1.000]). That is a *different*, threshold-free statistic from the FCR-at-a-threshold `RESULTS.md` §3 reports — not a reproduction of that table — but it is computed from the same measured cosines on the same set, so a reader can derive FCR at any threshold from the file. hashing-64 is not covered at all |
| §5b scale — STR-trust holds at 40x query volume | current | `scale/SCALE.md`, `scale-pressure/SCALE.md` | STR-trust 0.00 is the citable claim (Wilson [0.00, 0.02], n=250, Arm A) — retained in both `SCALE.md` files. **`FINDINGS.md` §5b itself disclaims the abstention-accuracy column** (0.00 / 0.86 / 0.99 across the four runs) as a generator artifact — near-duplicate synthetic documents defeat the embedder, not a trust-layer measurement — so treat that column as not-yet-measured here too, per `FINDINGS.md` §6 |
| §6 abstention threshold: measured, and rebuilt | current, unretained | — | **load-bearing** (see gate below): produced the shipped `recall.calibration.best_threshold()` midgap-q05/q95 rule — confirmed current in code, 2026-07-28. Verified end to end: threshold 0.728 ± 0.042, false-abstain 0.015, gap FCR 0.000 (bge-small, 5,450 chunks, 4 fresh HNSW builds). No JSON retained under `results/`; the sweep exists only as a table in `FINDINGS.md` |
| §7 private-corpus abstention | current, unretained | — | abstention accuracy 0.89 [0.57, 0.98], n=9. Corpus is private, so not independently checkable by anyone outside this repo either way — but that is a separate fact from artifact retention, and no JSON is retained under `results/` regardless |
| §8 PEP abstention | current, unretained | — | abstention accuracy 1.00 (11/11), both embedders (`FINDINGS.md` §8, ~line 551). Public corpus and questions, cheap to re-establish: `python -m recall.eval.labelled --corpus peps/peps --questions recall/eval/peps_questions.json --glob '**/*.rst'`. Cheap does not mean done — `recall/eval/labelled.py` has no `provenance_block` call (Phase 0 did not wire it; see "Known gaps"), so even re-running today would not produce a retainable, row-count-bearing artifact |
| §9b LOCOMO abstention, 4 modes | current | `locomo/postfix_abstention.json` | post-#81/#84. `locomo_abstention.py:160` (line numbers are against `9eb3bc1`, this document's baseline — see line 3) passes `calibration=cal` explicitly, so #101's auto-load bug never reached it (design spec §3). **Since this document's baseline, also retained:** the same FINDINGS §9b prose quotes a pre-fix comparator — "calibration now catches 0.574 (**was 0.527**)", "refuses 0.420 (**was 0.370**)", "discrimination 0.154 against a **pre-fix 0.157**" — and that comparator is now itself an artifact, not just an assertion: `locomo_abstention.json` (`_provenance.generation: "pre-#81/#84"`, superseded by this same postfix file). Verified 2026-07-28: its `modes.calibrated.adversarial_abstention.rate` is 0.5269 and its discrimination is 0.5269 − 0.37 = 0.1569 — both match the prose to rounding |
| §9b abstention with rerank on | **unmeasured** | — | #103 measured the default mode only (0.00, unchanged, confirmed identical across `baseline.json`/`rerank_modern.json`/`rerank_shipped.json`). The calibrated and judge modes have never been crossed with a reranker |
| §9c entailment ROC sweep | stale, artifact retained | `locomo_entailment_sweep.json` | **Corrected 2026-07-28 — this row was wrong.** It previously claimed the original pre-#81/#84 measurement had "no retained raw artifact either." False: `locomo_entailment_sweep.json` is tracked in git and carries a `_provenance` block reading `generation: "pre-#81/#84"`, `status: "not re-measured"`, `superseded_by: null`, `backs: ["RESULTS §7b judge sweep", "FINDINGS §9c"]` — a reader can open it and check the two-judge ROC table (`RESULTS.md` §7b) against the file directly; it is the full per-threshold sweep for both `qnli-distilroberta` and `qnli-electra-base`, not a summary. What is genuinely *not* retained is the attempted **replacement**: the post-fix re-run (`postfix_entailment_sweep.log`) died after 9 of LOCOMO's 10 conversations (conv-26, 30, 41, 42, 43, 44, 47, 48, 49) and wrote no JSON; that log is gitignored (`results/locomo/*.log`) and worktree-local — not present in this checkout, not retained in git — and `ARTIFACTS.md` confirms no successor exists ("superseded by: nothing — not re-measured"). So: **stale** because the measured configuration is superseded; **artifact retained** because the record of that superseded configuration was never lost — only its would-be replacement was |
| §10 LongMemEval, all rows | **unfalsifiable** | — | pre-#81/#84; indexes and output discarded. 6h39m to rebuild the merged index alone |
| every row above | **no row count on any existing artifact** | — | no artifact retained under `results/` before this cycle's Phase 0 (2026-07-28) records the corpus row count it measured. Phase 0 wired `recall/eval/locomo.py`, `locomo_abstention.py` and `locomo_entailment_sweep.py` (`grep provenance_block` confirms all three). It did **not** wire `recall/eval/labelled.py` (§7, §8) or `recall/eval/longmemeval_perq.py` (§10) — a future artifact from either producer will still carry no row count. See "Known gaps" below |

## What this gates

No combined signal, entity-mismatch feature or abstention-policy change is fit against a row that
lacks a retained artifact **of the shipped configuration** — marked **stale**,
**stale, artifact retained**, **unfalsifiable**, or **current, unretained** — until that row is
re-measured on the shipped configuration with an artifact retained, or explicitly demoted. A
retained artifact of a *superseded* configuration (§9c) does not clear this bar by itself — that is
exactly the distinction the "artifact retained" suffix is careful not to overclaim. This is written
to bind **§6** specifically: the shipped midgap-q05/q95 threshold rule is exactly the row a new
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
