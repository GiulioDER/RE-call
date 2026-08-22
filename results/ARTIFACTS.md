# Result artifacts — which configuration each one measured

[`RESULTS.md`](RESULTS.md) is the numbers. This file is the **map from a committed artifact to the
configuration that produced it**, because two of them are one character apart in name and five
versions apart in meaning.

Small JSON artifacts remain in this repository. Raw per-question payloads, logs, and gzipped run
packs are archived outside the source tree and represented here by filenames, checksums, and
summary documents. That keeps the Python library checkout small while preserving the verification
contract.

Every retained JSON artifact below also carries the same information **inside the file**, as a leading
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

### Truth-extraction labelled set

| artifact | backs |
|---|---|
| `results/truth_extraction/census.json` | the 17.0% prose recall ceiling; counts recomputable from `python/peps` at the recorded SHA |
| `benchmarks/labelling/truth_extraction/gold.manifest.jsonl` | 47 gold positives (authored PEP headers) + 4 transplanted negatives, frozen |
| `benchmarks/labelling/truth_extraction/adjudication.csv` | the blind negative-adjudication pack, 38 rows from the 30 of 175 marker-without-header PEPs that name a target; `adjudication_key.json` un-blinds it and must not be opened until labelling is finished |
| `recall/eval/peps_trust_queries.json` | 62 trust queries (42 successor / 20 abstain), one row per superseded PEP rather than one row per edge, replacing a shipped successor arm of n=4 |

**What this set measures well, and what it does not.** Of 47 authored header edges, only **8** are
restated in prose with the marker and the partner PEP in the same sentence. A perfect prose
extractor therefore scores recall **8/47 = 0.170** against the header denominator, and the usable
positive class for recall is **8, not 47**: an n on which a Wilson interval is about as
uninterpretable as the n=4 this set's trust arm was built to fix. Precision is the axis this set
measures with power, from the 38 adjudicable candidate pairs drawn from those markers.

Note the two counts are not interchangeable. 175 PEPs carry a closure marker that no header
confirms, but only 30 of them name a candidate target in the marker's own sentence, yielding 38
`(sentence, target)` pairs. The remaining 145 state a closure with no target named, and under
`recall/fix.py`'s rule an unprovable target is reported for a human rather than guessed at, so
there is no pair to put in front of an adjudicator. They are counted in the census and excluded
from the pack.

`census.py` counts an edge as restated if EITHER end states it, and of the 8 restated edges, 3 are
stated by the superseded PEP itself and 5 are stated only by the successor. That split matters
because `build_gold.py` hashes only the superseded PEP's body as the frozen gold item's input: the
3 figure is the one describing the superseded document alone, and for the other 5 the sentence
this census counted is not present in the text a prose extractor would actually read.

**PEPs are not memos.** They cite each other as `PEP 3106`, not `[[wikilink]]`, and they are
written under an editorial process a personal memo corpus does not have. A precision measured
here does not transfer to a memo corpus. What transfers is the **error mix**, which the four
transplanted fixtures in `benchmarks/labelling/truth_extraction/fixtures/` make checkable: they
reproduce, verbatim, the reported speech, hedging and two partial-scope failures measured on the
private 792-memo corpus and quoted in `recall/fix.py`.

**Two runnability caveats, both true today.** `run_trust_eval` (`recall/eval/harness.py:459`)
indexes the corpus with `recall.lint.DEFAULT_GLOB = "**/*.md"`, so pointed at a directory of PEP
`.rst` files it indexes zero documents. The trust set is not yet runnable by the only consumer of
its schema: "replacing a shipped successor arm of n=4" describes an intended substitution, not a
working one.

The `:0` chunk-id convention the trust set's `stale_ids`/`successor_ids` use is exhaustive on the
22-file memo corpus under `recall/eval/corpus` (every file chunks to exactly one chunk at
`DEFAULT_MAX_CHARS=800`), but not on PEPs, which chunk to between 5 and 153 chunks per file among
the 77 PEPs named in the trust set's successor arm. `pep-0387.rst:0` names 1 of that file's 14
chunks, and the scoring in `harness.py` compares the exact `file:ord` pair, so a hit anywhere else
in the same file scores as a miss.

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
| `enterprise_rag/dense_floor_strat100.retrieval.json` | the dense-cosine floor probe behind the Track B go/no-go: how many EnterpriseRAG questions the trust layer's **uncalibrated 0.50 threshold** would strip of all evidence once the benchmark goes through `reason()` instead of calling `HybridRetriever.search()` directly. **9 of 100 sampled, a population estimate of 5.7%**, so no mass abstention. ⛔ **ARM-DEPENDENT, and it is a lower bound on the submitted arm.** The score is `max_returned_dense_score`, the best dense cosine among the hits retrieval RETURNED, which is exactly what the floor gates on. That makes it depend on which hits come back: this run is `sparse_backend=lexical, reranker=null` at `scored_over_top_k=8`, while the submitted arm is `both` plus the Voyage reranker. A reranker reorders without rescoring, so it can only push the highest-cosine chunk out of the returned k, which can only move a question BELOW the floor. Visible already with no reranker: 9 rows have the dense argmax outside the returned 8, and `qst_0472` crosses the floor purely because of it (0.4700 against 0.4554). The summary carries `scored_arm` and `scored_over_top_k` so two runs can be told apart. ⚠️ **An ESTIMATE, not a bound, on the population.** The per-question measure is exact; the SAMPLING carries the error. Nine of ten strata are 10-question samples of populations from 20 to 175, and only `high_level` is a census. 🔑 **One sampled question carries 17.5 of the 28.5 estimated counts**: `basic` is 1 of 10 against a population of 175, so a single row is worth **3.5 points** of the 5.7% headline, 2.2% without it and 9.2% with one more. Spend any further sampling budget there. **Mind the three denominators**: 9/100 is over every sampled row and is an equal-allocation draw, so it estimates no population rate; `threshold_metrics['0.5']` is over the 80 rows carrying `expected_docs`, which excludes all 10 `high_level` and all 10 `info_not_found` rows; 5.7% is over the 500-question population after reweighting. ⚠️ **NOT a retrieval quality measurement**: `doc_ids_by_k` and `k_metrics` do not describe the submitted arm and must not be read as recall. 🔁 **Supersedes a run of the same name whose probe was too narrow**, evidenced by `enterprise_rag/dense_floor_probe_width.json`. The summary is derived by `benchmarks/enterprise_rag_contract.summarize`, which refuses a sample missing any stratum, and the runner writes through `write_dense_floor_artifact`, which validates and requires provenance first and leaves no file behind on mismatch |
| `enterprise_rag/dense_floor_probe_width.json` | the evidence that `query_dense(k=1)` under-reports, which is why the artifact above was re-measured. One query vector per question, scored at k=1 and at k=200, over the same 100 questions as that artifact. `k=1` never trips the `hnsw.ef_search` widening in `recall/store.py`, so it walks at ef_search=40 while k=200 walks at 800, and the same file records 0.385 recall at 40 against 0.942 at 200. **31 of 100 disagree, worst under-report 0.2234, and 4 questions have their floor verdict flipped by probe width alone.** ⚠️ **The narrow probe is not stable either.** An earlier run of this same comparison, same vectors and same index, returned 25 disagreements and 3 flips rather than 31 and 4. Two draws of an unstable instrument, which is a stronger reason to widen it than the bias alone |
| `store_latency/chunks_20k/splits.json` | the per-leg latency split behind the store-share figure — embed / dense / sparse / meta / fusion / rerank at 20,050 chunks, the evidence for whether a store backend swap could pay for itself. ⚠️ **SYNTHETIC corpus**, so the sparse leg does NOT generalise: `9a5165b` measured sparse median 496 ms on a real 72k-chunk corpus where this measures single-digit ms. Latency is the most host-dependent quantity here — read `stack` and `generated_at` before comparing it to anything. **Supersedes an earlier UNSTAMPED run of the same configuration**, whose figures (271.6 ms dense, 91.3%, 2.9%) appear in commit `66459ae`'s message and are reproducible from no file in the tree; superseded, not retracted — the shares agree to within 0.31 points |

### ATM-Bench: the full split, scored by the benchmark's own evaluator

| artifact | backs |
|---|---|
| `atm/atm_bench_full_20260821.json` | [`docs/ATM_BENCH.md`](../docs/ATM_BENCH.md), the README's ATM-Bench row and the ATM row in `docs/EVIDENCE.md`. QS **68.4264** and the three per-type accuracies are the official evaluator's own `atm_openai_gpt-5-mini_summary.json`, copied without edit; the retrieval figures (**Recall@10 92.8924**, Recall@10GT 86.9694) are recomputed from the run's `retrieval.jsonl` against the released ground truth and reproduce the submitted values to four decimals. ⚠️ **Three limits travel with these numbers and are recorded inside the file.** The judge kept the official prompt and the `gpt-5-mini` identity but ran over an **OpenRouter transport**, disclosed to the maintainers and not yet ruled on (`judge.transport_is_official` is `false`). The QS column is **not answer-model-matched** to the published baselines. And the leaderboard row is an **open pull request**, not an accepted placement. |

| `atm/atm_answer_diagnosis_20260822.json` | [`docs/ATM_BENCH.md`](../docs/ATM_BENCH.md) section 5, the answer-side decomposition. Produced by `benchmarks/atm_answer_diagnosis.py` from the archived package with **zero provider calls**, by aggregating the official evaluator's own per-question judgements, so unlike the run above it can be regenerated at any time rather than trusted. The script **refuses to write this file unless the replay reproduces the published QS**, because a decomposition that does not reproduce the score is describing a different run. Abstention and tokenisation come from the evaluator's own normalizer, never a local reimplementation. Aggregates only: no question text, no gold answer, no model answer, no per-question row, because the corpus is third-party data. 🔁 **Its modality-floor figure supersedes a published 1.97 QS over 20 questions**, which counted two questions whose token coverage is *unmeasurable* as coverage 0.0 through `(cov or 0)`; the corrected figure is 1.78 over 18, and the retraction is registered in `WITHDRAWN.json`. |

The per-question payloads (`answers.jsonl`, `retrieval.jsonl`, the two full judge outputs) are
archived outside this tree, per the policy at the top of this file. The artifact carries their
SHA-256 checksums under `package_sha256`, the four dataset hashes under `data_sha256`, and the
evaluator file hash under `judge.evaluator_sha256`, so an auditor handed the package can prove it is
the one these numbers came from.

⛔ **The commit that produced the run, `6c0ec26b`, is not on a public branch.** The configuration is
fully recorded in `config`, and the harness is published on
`claude/atm-answer-selection-public`, but its runner differs from the one used here. A byte-exact
re-execution from public code is therefore not possible today, and the artifact says so in its
`_provenance.note` rather than leaving a reader to discover it.

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
| external raw rows: `head_to_head/outcomes/*.jsonl` | one line per paired question, `{q, recall, mem0}`. These rows make the paired test **recomputable by a reader** without the raw runs, an API key, or trust, but they are no longer part of the library checkout. The retained artifact is `head_to_head/paired_accuracy.json` |
| external raw rows: `head_to_head/outcomes/as_shipped__*.jsonl` | **two replicates of one configuration**, 25 seconds apart, byte-identical configs, 0.4117 and 0.4221. The README's `0.42` was the higher one. Keyed by the run FILENAME's stem, and a repeated key is an error rather than a last-writer-wins — the loss of a replicate is how a measured spread silently becomes a point estimate |

`tests/test_h2h_artifact_backs_findings.py` asserts the committed artifact and the §9d table agree,
so the two cannot drift apart again. The raw runs stay out of the repository; what the claim rests
on does not.

### Promotion decisions — what the gate was asked, and what it answered

`recall/promotion.py`'s gate had no producer until `recall/eval/promotion/`. These are its first
real inputs and its first real output. The directory holds three kinds of file, and only one of
them is an artifact in this file's sense:

| file | kind | |
|---|---|---|
| external raw input: `promotion/labelled.manifest.jsonl` | **input**, frozen | question ids and input hashes, fixed BEFORE either arm ran. Carries its own digest and refuses an edited body. No `_provenance`, deliberately: a timestamp inside a digest-covered body makes the digest a function of the clock |
| external raw rows: `promotion/{baseline,candidate}.*.jsonl` | **raw rows** | one record per question per arm. The filename carries the arm label, the embedding profile id, and the first 16 hex of the profile FINGERPRINT, so two arms sharing a profile id and differing in artifact digest cannot land in one ledger |
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
`docs/archive/ENTERPRISE_PROGRAM_STATUS.md`'s standing blockers).

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

### Agent A/B — what the agent did, not what the retriever returned

| artifact | backs |
|---|---|
| `results/agent_ab/agent_ab_additive_2026-08-21.json` | every number in [`RESULTS.md` §13](RESULTS.md): the primary trap rate and its exact McNemar, the per-trap table and cluster CI, the control at p=1.0000, the Ragas quality rows, and the cost medians |
| `docs/preregistrations/2026-08-21-claude-md-plus-recall-additive.md` | the predictions, committed before the run, and the result appended beneath them without editing anything above |
| `benchmarks/agent_ab/trap-qualification.json` | where each trap's governing fact lives, measured against the real corpus and the real static prompt **before** any session ran |
| `benchmarks/agent_ab/calibration/memory-query-set.json` | the 50 answerable and 50 unanswerable labelled queries the corpus threshold was fitted on |

**What makes this artifact different from the twelve above it.** Everything else in `results/`
measures retrieval quality against a fixed corpus. This one measures an agent's behaviour, so the
things that can go wrong are different and two of them are recorded in the file rather than in prose.

**The arms differ by one thing, and it is asserted rather than assumed.** Both sessions receive the
same `CLAUDE.md` plus `MEMORY.md` byte for byte; the harness checks that the treated arm's prompt
contains the control arm's verbatim and adds no more than 2,000 characters, and aborts otherwise. A
pair is also discarded unless the treated session's own tool list contains a `mcp__recall*` tool and
the control's does not, because an earlier run of this design completed and reported success with no
memory tools attached at all.

**The primary endpoint uses no judge.** Deterministic checkers read the transcript for the known
wrong action. Ragas scores only §13's answer-quality rows, with a judge from a different model
family than the agent under test, against references written before the run.

⚠️ **Two caveats that live in the artifact, not only here.** `agent-ab-additive-002`'s environment
record was **reconstructed from the saved transcripts** after the runner was stopped mid-run, so its
per-session wall times come from each session's own `duration_ms` rather than from the runner timing
the subprocess; the two measure slightly different spans and are never pooled. And the `shared_db`
control trap was **excluded post hoc** because it stalled on a deliberately denied tool, which is
recorded with its reasoning in the preregistration. It was a control the memory layer was expected
to draw, not one it was expected to win.
