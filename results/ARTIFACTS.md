# Result artifacts — which configuration each one measured

[`RESULTS.md`](RESULTS.md) is the numbers. This file is the **map from a committed artifact to the
configuration that produced it**, because two of them are one character apart in name and five
versions apart in meaning.

Every artifact below also carries the same information **inside the file**, as a leading
`_provenance` block — so a file that gets opened, copied or linked on its own still says what it
is. The block never touches a measured value.

## Why this file exists

`results/locomo_abstention.json` reads `calibrated: 0.5269`. [`RESULTS.md` §7b](RESULTS.md)
publishes **0.574**. Both are correct. They are different configurations, and nothing in the
filename said so.

The LOCOMO harness has two generations:

| generation | sparse leg | dense scan | |
|---|---|---|---|
| **pre-#81/#84** | inert — it ANDed every query term, and LOCOMO questions average ~8 content terms against single-turn documents | capped near `hnsw.ef_search=40` | effectively **dense-only** |
| **post-#81/#84** | firing ([#81](https://github.com/GiulioDER/RE-call/issues/81)/[#82](https://github.com/GiulioDER/RE-call/pull/82)) | widened ([#84](https://github.com/GiulioDER/RE-call/pull/84)) | what the published tables describe |

A pre-fix artifact is **not wrong**. It is a correct measurement of a configuration this library no
longer ships, kept as that configuration's record — and in two cases it is the *only* evidence for a
"was X" figure the current documents quote. The hazard was never the numbers; it was that the
marker was on the **post**-fix files (`postfix_`) so the absence of a marker read as "the result"
when it meant "the older one".

## The artifacts

### Post-#81/#84 — the configuration the published tables describe

| artifact | backs |
|---|---|
| `locomo/postfix_pool20.json` | §7a depth curve; §7b `default` row |
| `locomo/postfix_abstention.json` | §7b four-mode ablation; FINDINGS §9b |
| `locomo_rerank/baseline.json` | §11 *no rerank* — reproduces `postfix_pool20` to four decimals |
| `locomo_rerank/baseline_verified.json` | §11 — the row-count-verified baseline re-run |
| `locomo_rerank/rerank_shipped.json` | §11 *ms-marco-MiniLM-L-6-v2* |
| `locomo_rerank/rerank_modern.json` | §11 *bge-reranker-base* |
| `cosine/distributions.json` | §12 cosine distributions |
| `wrrf/arm_C_rrf_pool100.json` | §9a's pool-100 column, **clean corpus** — 0.6615 at k=5. Replaces the withdrawn `locomo/postfix_pool100.json` |
| `wrrf/arm_A_rrf_pool20.json` | §9a's apparatus check — reproduces the published pool-20 column to Δ 0.0000 |

### Deliberately contaminated — evidence for the §9a retraction, never results

These two exist to be *wrong in a known way*. A doubled corpus reproduces the withdrawn pool-100
column to ±0.0013 at every depth; the same doubling is plainly visible at pool 20, where the
published column does not show it. That pair is what establishes the two published columns came from
different corpus states. **No number in either file describes what this library does.** Full account:
[`wrrf/FINDINGS_pool100_contamination.md`](wrrf/FINDINGS_pool100_contamination.md).

| artifact | k=5 | shows |
|---|---|---|
| `wrrf/doubled_pool100.json` | 0.5944 | reproduces the withdrawn 0.5957 — ±0.0013 at every depth, exact at k=20 |
| `wrrf/doubled_pool20.json` | 0.6081 | doubling costs −0.0625 at pool 20, so the defect is not depth-specific |

### Pre-#81/#84 — kept as the record of that configuration

| artifact | k=5 | backs | superseded by |
|---|---|---|---|
| `locomo/depth_curve_pool20.json` | 0.6243 | FINDINGS §9a's retained pre-fix anchor (0.624 / 0.798) | `locomo/postfix_pool20.json` |
| `locomo/depth_curve_pool100.json` | 0.6237 | the pool-100 control **retracted** in §9a — kept as the record of a retracted control, not as evidence | `wrrf/arm_C_rrf_pool100.json` |
| `locomo_fastembed_k5.json` | 0.6152 | the **withdrawn "hit@5 0.615"** figure — see below | `locomo/postfix_pool20.json` |
| `locomo_abstention.json` | — | FINDINGS §9b's *"(was 0.527)"*, *"(was 0.370)"* and *"pre-fix 0.157"* | `locomo/postfix_abstention.json` |
| `locomo_entailment_sweep.json` | — | §7b judge sweep / FINDINGS §9c | **nothing — not re-measured** |

Two of those rows deserve the emphasis:

- **`locomo/postfix_pool100.json` is gone, not moved.** It was withdrawn 2026-07-28 for having been
  measured on a doubled corpus (FINDINGS §9a's retraction notice), and **deleted rather than
  annotated** — an annotated wrong number in `results/` is still a number someone can read off a
  table. Anything that pointed at it now points at `wrrf/arm_C_rrf_pool100.json`, the clean re-run.
  Note this makes `depth_curve_pool100.json` a pre-fix record whose *immediate* successor was itself
  retracted; the chain skips it deliberately.
- **`locomo_entailment_sweep.json` has no successor.** FINDINGS §9c states the sweep is the pre-fix
  run and has not been re-measured, which is why that section rests on a *within-sweep* comparison
  of two judges rather than on a cross-harness check. Re-running it is open work, not a gap in this
  index.
- **`locomo_fastembed_k5.json` records 0.6152 — the withdrawn 0.615.** Until
  [#111](https://github.com/GiulioDER/RE-call/pull/111) that figure had no committed artifact, and
  both the README's withdrawn list and FINDINGS §9a removed it on exactly that ground. The artifact
  now exists, so those two statements were corrected. **The figure itself stays withdrawn** — the
  claim it was used for (reading its spread against 0.624 as HNSW build noise) is a *different*
  defect, and having the artifact does not repair it. What changed is the reason: it is no longer
  "uncheckable", it is "checkable and still not evidence for that claim".

## The model stack — why every artifact now carries one

`_provenance` says which **code** produced an artifact. For anything that passes through a model
that is not enough, and `locomo/postfix_abstention.json` is the proof: it publishes four abstention
modes, two of them route through a QNLI cross-encoder, and it named no `sentence-transformers`,
`transformers` or `torch` version at all.

Re-running it on a corpus asserted clean by row count moved the `entail` row by **0.0525**, while
the calibrated thresholds — fit directly on the distribution a doubled corpus would move —
reproduced *exactly*. So the corpus was not the variable, and nothing recorded said what was. Two
of four published rows could not be reproduced by anyone.

Three causes were then eliminated **by measurement, not argument**: corpus doubling (the thresholds
reproduce exactly), the judge's weights (`DEFAULT_QNLI_REVISION` predates the artifact), and
`sentence-transformers` (a run pinned back to 5.6.0 returned every mode identical to four decimals
— which also shows this pipeline is deterministic across independent environments, so run-to-run
noise is not it either). What remains is most consistent with an **independent HNSW index build**,
and that is now unfalsifiable: the 07-26 index is gone. The record that could have settled it is
the one nobody wrote.

Runs now emit `stack` alongside `elapsed_s`, from `recall.eval.provenance.model_stack()`. Artifacts
that predate this carry `"stack": "unrecorded"` — **the honest value, not a guess**: inventing
plausible versions would make an unreproducible row look reproducible. The set allowed to say
`"unrecorded"` is pinned by name in `tests/test_results_artifact_model_stack.py`, so a new artifact
cannot join it by omission. That list shrinks when a run is redone; it never grows.

What is *not* the gap: `recall.entailment.DEFAULT_QNLI_REVISION` pins the judge's Hub commit, so
the weights are immutable, and that pin predates every artifact here. The gap was the stack running
the model.

## Writing a new run

The committed artifacts are records of specific configurations. **Do not point `--out` at one** —
the harness docstrings deliberately no longer suggest a path that would overwrite a retained record.
Write new runs to a new filename, and if the run is meant to replace a published table, stamp its
`_provenance` and update the superseded row here in the same change.
