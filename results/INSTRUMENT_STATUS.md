# Instrument status — which abstention claims are checkable (coverage is test-enforced, status is not)

Written 2026-07-28 against `origin/master` @ 9eb3bc1; rows for the BEAM arm (sections 9e-9p)
added 2026-07-28 against branch HEAD `55dbee6`. Read from the artifacts in `results/`,
not from `FINDINGS.md`'s account of itself — **except** the rows marked **current, unretained**
below, for which no artifact exists at all; those are read from `FINDINGS.md`/`RESULTS.md` prose
directly, and flagged as such rather than silently treated as artifact-backed.

**This document cannot promise its own completeness by being carefully written — it has gone
stale three times in one day this way: once when a merged artifact falsified a row, twice when a
new `FINDINGS.md` section appeared that nobody updated this file for.** The second failure mode
is now a mechanical check, not a hope: whether every abstention-claiming `FINDINGS.md` section has
a row below is enforced by
`tests/test_eval_claims.py::test_every_abstention_claiming_finding_is_in_the_instrument_status_inventory`,
which derives the claim list straight from `FINDINGS.md`'s own numbered headings
(`recall/eval/claims.py`) and fails, naming the missing id(s), the moment a claim-bearing section
merges without a matching row. That test is what caught this file missing sections 9e, 9h, 9i,
9m, 9n and 9o on 2026-07-28, before this fix. What it does **not** and cannot check is whether a
row's STATUS is honest — that a `current` claim really is backed by the artifact it cites, that a
hedged claim's row does not overclaim it — which stays a human read of the evidence, exactly as
before.

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
**unmeasured** — no measurement has happened at all yet; not even a plausible artifact-free number
exists to distrust. Distinct from **unfalsifiable**: unfalsifiable means a measurement WAS made
and its evidence is now gone; unmeasured means the measurement itself has not been made. Already
in use below (the §9b rerank row) before this legend defined it — added here so the table and its
own legend agree.
**retracted** — the claim was published and then explicitly withdrawn by a LATER section of this
same document, on the same code path or a larger successor sample. Distinct from **stale**: stale
means a newer measurement on a *changed* configuration superseded an old one; retracted means the
claim's own conclusion did not hold up and the shipped configuration was never changed to begin
with, so there is no newer "current" row this status could point to instead.

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
| §9e BEAM abstention arm (rate + false-abstain, two independent numbers) | **unmeasured** | — | Design only, no numbers: reports the same shape as §9b (abstention rate on the `abstention` category, false-abstain rate on the other nine) once a paid BEAM arm runs. The section says so itself — "Cells land here when the paid arms run" — and as of this writing they have not: no `results/beam*`/`bench_beam*` JSON exists anywhere in this repo's git history (`git log --all --full-history` on both globs returns nothing). `python -m benchmarks.beam.run --dry-run` validates the harness for $0; the paid run is `--model openai/gpt-5` |
| §9h BEAM abstention category is a hallucination test | current, unretained | — | Scores Mem0's own published per-question BEAM answers (n=70 abstention questions): abstained 38 (mean 0.974), answered 32 (mean 0.016) — Mem0 fabricates an answer 46% of the time. Reproducible from Mem0's published `beam_1m_results.json` (their repo, Apache-2.0), not retained under `results/` here. This section's OWN conv-0-only entailed-count pilot (n=2 unanswerable) and the policy-change it motivated are superseded by §9i's larger probe below — the hallucination finding is not touched by that retraction. `benchmarks/beam/abstention_probe.py` reproduces the entailed-count table; `benchmarks/beam/calibrate.py --out beam_calibration.json` reproduces the raw-cosine table, and per the section's own words that JSON "is written by --out and is not committed" |
| §9i RETRACTION: entailed-count threshold policy does not pay | **retracted** | — | Full probe (30 unanswerable, 270 answerable, conversations 0-14, $0) supersedes §9h's n=2 pilot: every stricter entailed-count policy loses more on the 270 answerable questions than it gains on the 30 unanswerable ones. The shipped `any()` rule in `recall.entailment.apply_entailment` — confirmed current 2026-07-28 (`abstained = not still_ok`, unchanged) — is already the best of the five policies tested. §9h's underlying mechanism claims ("`any()` over ~200 candidates is maximally permissive", "entailment count orders the classes correctly where cosine inverts them") are explicitly NOT retracted, only the policy-change recommendation is. No JSON retained; reproducible via `benchmarks/beam/abstention_probe.py`, $0, no LLM |
| §9m absolute threshold is embedder-fragile | current, unretained | — | **The section's own title says "indicative, NOT yet established" — this status is not claiming more than that.** bge-small vs `text-embedding-3-small` top-1 cosine on the SAME 3 BEAM conversations: median 0.825 vs 0.635; `DEFAULT_GAP_THRESHOLD = 0.50` (confirmed unchanged in `recall/guards.py`, 2026-07-28) starves 0/54 answerable vs 19/270. Single corpus, one arm at n=54 — the section itself names the small-sample-reversal pattern elsewhere in this session (§9i above; a k-sweep result) as reason to distrust a stronger reading. §9n below is the larger follow-up (2 of the 3 corpora this section calls for) and reads as settling the same question more broadly, but does not reproduce this row's specific 3-conversation measurement. No JSON retained (`beam_calibration.json` is written by `--out` and not committed) |
| §9n regime sweep (2 corpora x 3 embedders) settles the threshold problem | current, unretained | — | "Established", per the section's own heading. `DEFAULT_GAP_THRESHOLD = 0.50` sits at the 0th percentile of five of the six distributions and the 16th of the sixth; four candidate replacements (per-query percentile, gap, corpus quantile on real queries, corpus quantile on self-queries/H4) are all measured and rejected. Predictions committed beforehand and git-tracked: `results/PREDICTIONS-regime-sweep.md`. The raw per-cell scores backing the summary table are named as living in `regime_sweep.json` by `results/HYPOTHESES-cosine-regimes.md`, but that file is absent from this checkout AND from git history under that name (`git log --all --full-history` on the glob returns nothing) — so the raw artifact is unretained. **Partial exception**, in the same spirit as the §2 row above: the SUMMARY table itself (medians, ranges, starve rates) is retained as prose in the git-tracked `results/HYPOTHESES-cosine-regimes.md`, so a reader can check the published numbers against that file even though the per-query scores behind them are gone |
| §9o entailment guard does not discriminate on an ordinary corpus | current, unretained | — | Held-out probe (120 of 787 curated memos held out, the other ~660 indexed, each held-out memo's own `description:` as its query): cosine AUC 0.78 vs entailment-guard AUC 0.59 — the edge the entailment guard showed on BEAM's adversarial construction (§9h) does not reproduce on a corpus built to be fair to both signals. False-abstain cost to catch half the unanswerable questions: 13.7% (cosine) vs 26.0% (entailment). Predictions committed in the module docstring before the run (commit `e273c99`, confirmed: that commit added only `benchmarks/beam/heldout_probe.py` and `lexical_probe.py`, no result JSON). No artifact retained under `results/`; this is the measurement "the abstention lane has no remaining candidate with an empirical basis" rests on |
| §10 LongMemEval, all rows | **unfalsifiable** | — | pre-#81/#84; indexes and output discarded. 6h39m to rebuild the merged index alone |
| every row above | **no row count on any existing artifact** | — | no artifact retained under `results/` before this cycle's Phase 0 (2026-07-28) records the corpus row count it measured. Phase 0 wired `recall/eval/locomo.py`, `locomo_abstention.py` and `locomo_entailment_sweep.py` (`grep provenance_block` confirms all three). It did **not** wire `recall/eval/labelled.py` (§7, §8) or `recall/eval/longmemeval_perq.py` (§10) — a future artifact from either producer will still carry no row count. See "Known gaps" below |

## What this gates

No combined signal, entity-mismatch feature or abstention-policy change is fit against a row whose
status is anything **other than plain `current`** — i.e. `stale`, `stale, artifact retained`,
`unfalsifiable`, `current, unretained`, `unmeasured` or `retracted` — until that row is re-measured
on the shipped configuration with an artifact retained, or explicitly demoted. Phrased as "every
status but `current`" rather than as a fixed enumeration, on purpose: a fixed list needs a human to
remember to extend it every time a new status is introduced (as `unmeasured` and `retracted` were,
2026-07-28), and a forgotten extension would silently let a change get fit against exactly the kind
of row this gate exists to block. A retained artifact of a *superseded* configuration (§9c) does
not clear this bar by itself — that is exactly the distinction the "artifact retained" suffix is
careful not to overclaim. This is written to bind **§6** specifically: the shipped midgap-q05/q95
threshold rule is exactly the row a new abstention signal or threshold change would be fit against,
and until this row existed the gate had nothing there to bind. It now also binds **§9i**
explicitly: the retracted entailed-count policy is not eligible to be re-proposed without new data.

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
- **None of `benchmarks/beam/*.py` writes a retained artifact at all — not even an unwired one.**
  The three LOCOMO runners at least write a JSON that merely lacks a row count (previous gap
  above); `heldout_probe.py`, `calibrate.py`, `abstention_probe.py` and `regime_sweep.py` either
  print to stdout or write to an explicit `--out` path nobody commits, so every §9e/9h/9i/9m/9n/9o
  row added 2026-07-28 is `unmeasured` or `current, unretained` and stays that way structurally,
  not just by omission. Confirmed 2026-07-28: `git log --all --full-history` finds zero commits
  ever touching a `results/beam*` or `results/**/bench_beam*` path.
- **The two provenance mechanisms are not integrated, and `results/locomo/prereq_index.json` is the
  concrete instance, not a hypothetical.** The "note on provenance" above (this document, ~line 29)
  already flags that master's `_provenance` block and this branch's `provenance_block()` are
  separate and neither implies the other. On 2026-07-28 that stopped being abstract:
  `recall/eval/locomo.py` produced `prereq_index.json` as the apparatus check ahead of the §9c
  re-measurement, and because the runner only calls `provenance_block()`
  (`corpus_rows`/`table`/`tenants`/`git_sha`/`git_dirty`), the file carried no master `_provenance`
  block at all — it sat committed-but-untracked, failing all four of
  `tests/test_results_artifact_provenance.py`'s per-file checks, until it was hand-stamped (see
  `results/ARTIFACTS.md`'s row for it). **Every artifact any of the three wired LOCOMO runners
  produces from here on will fail the same guard the same way** until either a runner is changed to
  also emit `_provenance` — a design change, out of scope for that fix and not attempted — or
  whoever commits the artifact stamps it by hand, as was done here.
