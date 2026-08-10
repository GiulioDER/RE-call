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

### Reasoning Session 1 control

| artifact | backs |
|---|---|
| `reasoning_session1_baseline.json` | `docs/REASONING_CONTRACT.md` Session 1 control harness. Synthetic, frozen baseline observations before reasoning is enabled. Records direct, multi hop, near miss, contradiction, missing supersession, ambiguous entity, empty corpus, and stale corpus cases. |
| `reasoning_session6_controls.json` | `docs/REASONING_CONTRACT.md` Session 6 control harness. Frozen offline benchmark with direct QA, multi hop, temporal reasoning, supersession recovery, near miss abstention, contradiction detection, entity disambiguation, missing evidence, and clarification decisions. Separates synthetic and real corpus controls, records pre registered thresholds, ablation arms, nearest neighbor, shuffled edge, removed edge controls, per query observations, and evaluator leakage audits. |
| `reasoning_session8_reproducibility.json` | `docs/REASONING_SESSION8_AUDIT.md` Session 8 final audit bundle. Records code base revision, dependency versions, model stack identifiers, corpus and generation fingerprints, validation commands, regenerated reasoning artifact hashes, evaluation results, release decision, and remaining limitations. |

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
| `beam_voyage/ksweep.json` | the `k` choice for voyage-4-large — a **proxy** (nugget coverage), not a judged score; see its `_provenance.note` |
| `wrrf/arm_C_rrf_pool100.json` | §9a's pool-100 column, **clean corpus** — 0.6615 at k=5. Replaces the withdrawn `locomo/postfix_pool100.json` |
| `wrrf/arm_A_rrf_pool20.json` | §9a's apparatus check — reproduces the published pool-20 column to Δ 0.0000 |
| `store_latency/chunks_20k/splits.json` | the per-leg latency split behind the store-share figure — embed / dense / sparse / meta / fusion / rerank at 20,050 chunks, the evidence for whether a store backend swap could pay for itself. ⚠️ **SYNTHETIC corpus**, so the sparse leg does NOT generalise: `9a5165b` measured sparse median 496 ms on a real 72k-chunk corpus where this measures single-digit ms. Latency is the most host-dependent quantity here — read `stack` and `generated_at` before comparing it to anything. **Supersedes an earlier UNSTAMPED run of the same configuration**, whose figures (271.6 ms dense, 91.3%, 2.9%) appear in commit `66459ae`'s message and are reproducible from no file in the tree; superseded, not retracted — the shares agree to within 0.31 points |

### The Mem0 head-to-head — the table that had no artifact until 2026-08-06

This section is the one this file was missing. §9d, the five-row paired comparison against Mem0, is
the loudest claim in the README, and until 2026-08-06 it was the **only** published figure with
nothing in `results/` behind it: no `mem0` path in the tree, no `9d` entry here. The raw runs
(~123 MB of generated answers and retrieved contexts) were never committable, and nothing derived
from them was committed instead, so the gap was invisible rather than declared.

Its first consequence was immediate: the first verified build found row 4's `paired p` published as
`0.00018` when it is `0.00017` — row 2's value, copied down. Two weeks, and no mechanism existed
that could have caught it.

| artifact | backs |
|---|---|
| `head_to_head/paired_accuracy.json` | §9d's five-row table, the Holm maximum, and the `text-embedding-3-small` as-shipped paragraph. Derived by `benchmarks/h2h_artifact.py` **through `benchmarks/analyze.py`** — the same code that produced the published table, not a second implementation of McNemar. Carries the sha256 of both raw runs per row, so a re-scored run cannot be substituted silently |
| `head_to_head/outcomes/*.jsonl` | one line per paired question, `{q, recall, mem0}`. This is what makes the paired test **recomputable by a reader** without the raw runs, an API key, or trust. 1,540 lines per row, ~60 KB each |
| ⚠️ `head_to_head/outcomes/as_shipped__*.jsonl` | **two replicates of one configuration**, 25 seconds apart, byte-identical configs, 0.4117 and 0.4221. The README's `0.42` was the higher one. Keyed by the run FILENAME's stem, and a repeated key is an error rather than a last-writer-wins — the loss of a replicate is how a measured spread silently becomes a point estimate. `write_artifact` also prunes any vector this build did not write, so the directory is a function of the build and cannot accumulate files that back nothing |

`tests/test_h2h_artifact_backs_findings.py` asserts the committed artifact and the §9d table agree,
so the two cannot drift apart again. The raw runs stay out of the repository; what the claim rests
on does not.

### Promotion decisions — what the gate was asked, and what it answered

`recall/promotion.py`'s gate had no producer until `recall/eval/promotion/`. These are its first
real inputs and its first real output. The directory holds three kinds of file, and only one of
them is an artifact in this file's sense:

| file | kind | |
|---|---|---|
| `promotion/labelled.manifest.jsonl` | **input**, frozen | question ids and input hashes, fixed BEFORE either arm ran. Carries its own digest and refuses an edited body. No `_provenance`, deliberately: a timestamp inside a digest-covered body makes the digest a function of the clock |
| `promotion/{baseline,candidate}.*.jsonl` | **raw rows** | one record per question per arm. The filename carries the arm label, the embedding profile id, and the first 16 hex of the profile FINGERPRINT, so two arms sharing a profile id and differing in artifact digest cannot land in one ledger |
| `promotion/decision.null-difference.json` | **artifact** | the decision, with `_provenance` |

| artifact | backs |
|---|---|
| `promotion/decision.null-difference.json` | the end-to-end proof that the producer works and that the gate refuses a null difference. Baseline and candidate are the SAME configuration; all 25 rows share an `output_hash` across the two arms, so the delta is zero by construction rather than by measurement. `promoted: false` on **five** counts: the bootstrap interval does not clear zero, no corpus reaches Holm-corrected significance, the superseded trust rate was NOT MEASURED, security is not green (unverified is not green), and **latency is PENDING** |

⚠️ **This run is DEGRADED, and the artifact says so twice.** `trust_verdicts` reads
`{"unverified": 25}` for both arms: a plain store has no generation-bound certified calibration,
so it ran under `--trust-policy development`. `recall/trust.py` overwrites **every** verdict with
`unverified` in that mode, *after* the trust layer has computed the real one, so a superseded hit
and a clean one leave identical rows. That is why `superseded_trust_rate` is `null` rather than
`0.0`: a rate of zero would have satisfied the gate's zero-tolerance check by never having
measured it. The same reason makes `false_confidence: 1.00` a fact about a degraded system and
**not** a measurement of this library's abstention. The two arms stay comparable to each other;
neither is comparable to a trusted run. A strict-mode run needs a generation-bound certified
calibration, which no session has wired end to end yet.

`n_manifest_questions: 25` beside `n_paired_questions: 14` is the other thing to read: the gate
tested the 14 answerable questions and the digest covers all 25, and both numbers are recorded so
a reader never has to assume they are the same.

⚠️ **`latency` is PENDING, not measured.** `gate_input_p95_ms` is `null`, and that BLOCKS
promotion. The figures under `observed_diagnostic_only` come from a developer laptop and describe
it, not a reference environment; the program has no idle 16-vCPU reference host (see
`docs/ENTERPRISE_PROGRAM_STATUS.md`'s standing blockers).

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

`generated_at` travels with it, for the companion question — and the one that cost the most to
answer without it. Deciding whether `postfix_abstention.json` predated the double-index guard meant
reading git for the commit that *added* the file, and a commit date is when someone committed, not
when the run happened. For `3ee36ed` those differ by an unknown amount: exactly the gap that let a
07-26 run and a 07-28 guard pass each other. **Two facts identify a run — which stack, and when.**

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

## `results/promotion/generation-parity.json`

**What it measured:** that a context mode changes the text EMBEDDED and never the raw chunk content
or raw content hash STORED. Four generations built over the PEPs corpus (746 sources, 21,924 chunks
each) under `bge-small-symmetric-v1` and the three `bge-small-context-*-v1` profiles, compared
pairwise against the baseline with the shipped `recall.migration.validate_generation_parity`.

Result: parity holds on all three, 0 missing sources, 0 extra, 0 hash mismatches, equal chunk
counts, 746/746 coverage, 0 degenerate hashes, and a positive control that fired on exactly one
changed file.

⚠️ **These are NOT the 2026-08-06 promotion campaign's own generations.** That harness indexes into
a `promo_<uuid8>` table and drops it in a `finally`, so its generations no longer exist. This is a
rebuild over the same corpus with the same embedder and pinned artifact tree, and the file says so
in its `reconstruction_note`.

⚠️ **Its `_provenance` block was STAMPED AFTER THE FACT, and the block says so.** This artifact was
produced before `benchmarks/check_generation_parity.py` emitted one. Rather than leave it outside
the convention, the block was added by hand from the run's own driver log and the versions installed
on the host that ran it. It carries `stamped_after_the_fact: true`, `measured_at` (the run) kept
separate from `provenance_stamped_at` (the edit), and the digest of the archived original.

**The block is the only difference, and that was ENFORCED rather than asserted.** The stamping tool
refused to write until a round trip through `json.dumps` reproduced the file byte for byte, so
re-serialisation could not smuggle in a change, then compared all nine pre-existing keys before and
after. `git diff` records **24 insertions, 0 deletions**. (Its first version was refused by its own
guard over a single trailing newline, which is the guard working on the tool rather than on the
artifact.)

The archive is the run record and was deliberately **not** modified, so the two copies differ by
exactly this key:

| copy | sha256 |
|---|---|
| archived original, covered by that directory's `MANIFEST.sha256` | `073628143b35299e…a2b50147` |
| this committed copy, with the stamp | `d2ee470e5da874b5…c84d7e3e` |

Full run record: `/var/lib/recall-benchmarks/2026-08-06-context-mode-generation-parity/`.
⚠️ Regenerating a natively-stamped artifact now needs a **full four-arm re-index**, not a compare
re-run: the generations were dropped once this work merged.
