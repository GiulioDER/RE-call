# Pre registration: serving a corpus before it has a calibration

**Date:** 2026-08-18   **Status:** measured 2026-08-18. Q1 confirmed, with one registered sub
prediction **not tested** and one registered conjunct **withdrawn**. Q2 half confirmed and half
falsified, on a substituted corpus. Predictions and falsifiers below are unedited; every result is
appended below a horizontal rule.

> ⚠️ **One post measurement edit was made to the sections above the first rule, and it is declared
> here rather than left to be discovered.** Private infrastructure identifiers (a host alias, an
> absolute server path, a unix socket directory, a port and an internal corpus name) were replaced
> with neutral descriptions before this file was first published, because this repository is public.
> No prediction, no falsifier, no threshold and no number was changed. Where the original named a
> specific machine, the text now says "a remote read only Postgres instance", which is what the
> measurement actually depended on.

## The question

Two questions, both raised by the redesign in `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`.

**Q1 (adoption fidelity).** Can a corpus already in the legacy `chunks` table be adopted into a
generation without re embedding, such that the adopted generation's vectors are provably the
vectors that pipeline produced from those bytes?

**Q2 (provisional threshold quality).** Is a threshold fitted on a machine generated query set
(`recall/wizard/queryset.py`) better than the uncertified 0.5 demonstration constant that
`RECALL_TRUST_MODE=development` uses today?

## What I predict

**Q1.** At least 95 percent of sources in the `memory` tenant of a remote read only Postgres corpus
will adopt with a verified binding, meaning `metadata->>'content_hash'` equals a fresh sha256 of the
file on disk AND `metadata->>'embedding_profile'` equals the profile id of the configured embedder.
The remaining failures will be sources edited since indexing, not sources missing the metadata keys.
A re embedding attestation sample of 20 chunks will reproduce the stored vector to within cosine
0.9999. Adoption of 8,671 chunks will complete in under 5 minutes, against the ~10 hours a full
`generation build` costs on the same corpus.

**Q2.** The provisional threshold will beat 0.5 on false confidence rate by at least 0.10 absolute,
measured on a held out human labelled set. It will NOT reach the separability bar that
`Calibration.certified` requires, which is why the design gives it a distinct status rather than
calling it certified.

## What would falsify this

**Q1** is falsified if under 95 percent of sources verify, if the failures are dominated by absent
metadata rather than by edited files, if any attestation sample chunk reproduces below cosine
0.9999, or if adoption takes over 5 minutes. Any of those means the legacy table cannot establish
the binding and the adoption path must be dropped rather than weakened.

**Q2** is falsified if the provisional threshold's false confidence rate is within 0.10 of 0.5's, or
worse than it. That would mean a generated query set buys nothing over the constant, and the design
should keep the constant and be honest about it instead of adding a status for it.

⚠️ Q2 is the one that could talk me into shipping something dishonest, so the falsifier is stated
against a **human labelled** held out set specifically. Scoring a generated set against itself would
measure only that the fitting worked.

## How it will be measured

**Q1.** Against the `memory` tenant of a remote read only Postgres instance, 1,080 files and
8,671 chunks. For every distinct `source` in `chunks`: read the file, sha256 it, compare to
`metadata->>'content_hash'`. Record the four way split (verified / file changed / file missing /
metadata absent). Then draw 20 chunks uniformly at random, re embed their `text` with the configured
embedder, and record cosine against the stored `embedding`. Sample drawn with a fixed seed passed in
rather than generated, and the full population enumerated before sampling, per the sorted sample
plus early stop lesson. Time the adoption transaction with the DB clock, not wall clock in Python.

**Q2.** Build one generation over the `memory` tenant. Generate a query set with
`recall/wizard/queryset.py` at `DEFAULT_PER_CLASS`. Fit a threshold on it. Separately, hold out the
existing human labelled set. Score three arms on the held out set: the 0.5 constant, the provisional
threshold, and (as a ceiling) a threshold fitted on the held out set itself. Primary metric is false
confidence rate, matching `recall/cli.py`'s `fcr_at_050` reporting. Report the Hanley McNeil
interval, not just the point.

## What I already know

- Calibration binds to a `ready` generation, not only an `active` one
  (`recall/calibration_v2.py:349`), and `pin_generation` accepts the same three states
  (`recall/generation_store.py:146`). So neither calibration nor serving actually requires promotion.
- The legacy `chunks` table has no `source_sha256` column (`recall/migrations/sql/0001_v08_baseline.sql`),
  but chunk metadata carries `content_hash`, `index_fingerprint` and `embedding_profile`
  (`recall/index.py:707`), all written at embed time.
- `recall/eval/synthetic.py:69-75` already measured that perturbation derived unanswerable queries
  are not separable. `queryset.py` builds them off topic instead, which is why Q2 is worth asking at
  all rather than being obviously false.

## Confounds I can name now

The `memory` tenant is small and is my own writing, so its vocabulary is unusually consistent and Q2
may flatter the generated set. `content_hash` proves which bytes were read, not which chunker ran,
which is why the attestation sample exists as a separate check rather than being folded into the
verification count. Voyage query embeddings are not deterministic, so Q2 must use fastembed. The
attestation cosine bar of 0.9999 assumes ONNX determinism on one host; a different host is a
different measurement and would need its own record.

---

## A note on line numbers in every result below

All source citations **below this rule** are re measured whenever this file is edited, against the
commit the edit lands on.

⚠️ **Line numbers in this repository drift faster than a document can be written.** Measured over
this work: an intervening merge moved **13 of 13** checked citations while the first draft was in
progress; a second merge moved a further one; a third moved another. Three separate concurrent
merges in one working session. So a citation here is accurate as of its commit and carries no
promise beyond it, and any reader finding one off by a few lines should search for the quoted text
rather than assume the claim is wrong. The durable fix is a check in CI that resolves every
`path:line` in `docs/` and fails when one stops matching its quoted anchor; that does not exist yet
and is filed as follow up work.

🔁 **Built 2026-08-18: `scripts/check_doc_citations.py`, run by the `citations` job in CI.** The
shape above turned out to be the wrong one and is recorded here because the correction is the
useful part. Matching against a "quoted anchor" was implemented first and **produced 33 findings on
a tree whose citations had just been repaired by hand**, nearly all of them correct citations
flagged wrongly: a documentation line routinely carries several backticked symbols and several
citations, with no reliable pairing between them. Thirty false alarms do not make a strict check,
they make a check somebody disables. What shipped instead asks git how each cited file moved between
the citing document's last commit and HEAD, which turns drift into arithmetic and needs no anchor.
Its own limits are stated in the script: it cannot see uncommitted edits, it needs `fetch-depth: 0`,
and its suggested destination is exact only where the citation was accurate to begin with, which in
this repository is not always true.

⚠️ Citations **above** this rule are left exactly as registered, because a prediction is a
historical record and must not be edited. One of them has since moved: `recall/index.py:707`, cited
in "What I already know" for the metadata stamp, is `recall/index.py:836` at the time of writing.

## Reproducing any of this

⚠️ **The harness is not in the tree.** Every number below was produced by throwaway scripts, so no
reader and no future session can re run these measurements without rebuilding them from the method
sections. That is a real gap against this project's own rule of shipping the re measure command with
the number, and it is recorded rather than hidden. What is reproducible from this document alone:
the census predicate, the attestation sampling rule, the seeds, the generator entry points and the
embedder identities are all stated exactly.

---

## Q1 result

Measured 2026-08-18 against the `memory` tenant of a remote read only Postgres instance. Corpus as
found: **1,080 sources, 8,716 chunks** (the registration said 8,671; 8,716 is what the table holds
today). Every embedding call ran on the local workstation and the remote instance was read only:
the only statements issued against it were `SELECT`s. Its load average was 11.07 on 12 cores
throughout, which is *why* embedding was kept local, not evidence that it was.

### Verification census

Every source read exactly as `recall/index.py:697` reads a markdown source
(`read_text(encoding="utf-8-sig")`, universal newlines, then `_strip_nul`, then
`sha256(text.encode("utf-8"))` at `recall/index.py:716`). A `sha256sum` over raw bytes would have
been wrong for any file with CRLF or a BOM. **Every source in this corpus is markdown**, which
matters because `bd582316` made the derivation media type dependent: a non markdown source is
hashed as raw bytes instead (`recall/index.py:718`).

| Bucket | Sources | Chunks |
|---|---:|---:|
| verified | **1,080** | **8,716** |
| changed | 0 | 0 |
| missing | 0 | 0 |
| metadata absent | 0 | 0 |
| unreadable | 0 | 0 |

**100.00 percent of sources and of chunks.** Metadata completeness is total: all 8,716 rows carry
`content_hash`, `embedding_profile`, `file`, `index_fingerprint` and `ord`.

🔑 **This measures ONE of the two registered conjuncts.** The registration defined a verified
binding as `content_hash` matching **AND** `embedding_profile` matching. The census implements only
the `content_hash` conjunct. The second was never evaluated per source, and the falsification below
shows it would have been vacuous: both sides are the same literal fallback string. So 100.00 percent
is the `content_hash` rate alone, on a criterion narrowed after registration. See the verdict table.

⚠️ **Negative control run, because an always equal comparison produces exactly this table.** A four
row synthetic input (real file with its real hash, real file with a wrong hash, nonexistent path,
empty hash) classified 1 / 1 / 1 / 1 across verified / changed / missing / no_hash. The classifier
discriminates. Note the census has **five** buckets, not the four the registration named; the fifth,
`unreadable`, separates an I/O error from a missing file.

### Pipeline attestation

20 chunks, sampled deterministically over the enumerated population
(`ORDER BY md5(id || 'q1-attest-2026-08-18') LIMIT 20`), re embedded on the workstation with
`fastembed:BAAI/bge-large-en-v1.5` through the indexer's own call path (`embed_passages`, which is
what `embed_with_cache(..., purpose="passage")` reduces to at `recall/cache.py:102`). `context_mode`
is `none` for this tenant, so the embedded text is the chunk text (`recall/context.py:344`).

- 20 of 20 at or above the registered 0.9999 bar; min 1.0000
- ⚠️ control, chunk *i* stored against chunk *i+1* fresh: max **0.709**, mean **0.624**

The script also computed an exact `n_bitwise_identical` statistic, which would have been the
stronger claim, but its value was not retained in any saved artifact and is not quoted here.

⚠️ **The attestation re embeds the STORED CHUNK TEXT, so it tests the embedder and never the
chunker.** Chunking happens upstream of the text it compares. A chunker level counterpart would
have to re chunk verified files and compare chunk texts and ordinals, and was not run.

### Timing

⚠️ **These are a proxy, not a measurement of adoption.** No adoption path exists to run: nothing in
the design is implemented. What was timed is its dominant cost.

- vector copy, 8,716 rows into a temp table shaped `LIKE recall_chunks_v1`, timed with
  `clock_timestamp()` inside the transaction: **1.448 s**
- source verification, 1,080 files hashed: **1.82 s**
- **total ~3.3 s.** The attestation re embedding is **not** in this total and was never timed, so
  3.3 s is a lower bound on adoption as the design now defines it.

The ~10 hour comparator for a full re embed is carried over from the registration as prior
knowledge. It was **not** re measured here and is an estimate, not a measurement of this run.

### Verdict against the prediction

| Registered prediction | Measured | Verdict |
|---|---|---|
| ≥95 percent of sources adopt with a verified binding, defined as `content_hash` equal **and** `embedding_profile` equal | 100.00 percent on the `content_hash` conjunct | **confirmed on a narrowed criterion.** The `embedding_profile` conjunct was **withdrawn, not met**: it is unsound (below) and was replaced by the attestation sample |
| failures dominated by edited files, not absent metadata | zero failures of any kind | ⚠️ **not tested** |
| attestation sample reproduces to cosine ≥0.9999 | 20 of 20, min 1.0000 | **confirmed** |
| adoption under 5 minutes | 3.3 s for a proxy, attestation excluded | **confirmed for the proxy**, which is a lower bound |

Two rows are the honest ones. There were no failures, so the claim about their *composition* was
never put at risk. And the first row's criterion was narrowed after registration, which is recorded
here rather than absorbed into a clean "confirmed".

## What the measurement falsified in the design

⛔ **Step 4 of the adoption path in `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` is unsound as written.**
It proposed comparing `metadata->>'embedding_profile'` to the configured embedder's profile id as
the pipeline check. At the measured tree, the fallback returned the **literal string**
`"bge-small-symmetric-v1"` for any model without a registered profile, so `profile_id` did not
track the model at all:

- the corpus stores 1024 dimensional vectors labelled `bge-small-symmetric-v1`, while the registered
  profile of that name is 384 dimensional (`recall/embedding_registry.py:228-230`);
- a locally constructed `fastembed:BAAI/bge-large-en-v1.5` embedder reports
  `name='BAAI/bge-large-en-v1.5' dim=1024 profile_id='bge-small-symmetric-v1'`.

`index_fingerprint` inherits the defect, because `_index_fingerprint` hashes
`embedding_profile_id(embedder)` (`recall/index.py:472`). So **neither stored field can identify the
model**, and two different unregistered models of equal width would compare equal.

The consequence for the design is a promotion, not a retreat: the **pipeline attestation sample is
the only sound embedder check available**, and must be a required step of adoption rather than the
supplementary evidence the design called it. Its cost was not measured; what was measured is that a
20 chunk sample runs in seconds.

🔁 **Fixed upstream the same day by #370**, which this measurement prompted: `_fallback_profile_id`
(`recall/embeddings.py:699`) now derives `unregistered__{model}__{dimension}__{kind}`
(`recall/embeddings.py:750`). The measurement above stands as a dated record of the tree it ran
against, and **the conclusion is unchanged**: every corpus indexed before #370 still carries the old
literal, and those are precisely the rows an adoption path reads. Fixing a writer does not repair
rows already written.

🔁 **The second half was fixed the same day by #381**, and the sentence above about
`_index_fingerprint` is now a record of the measured tree rather than of the code. It no longer
hashes `embedding_profile_id(embedder)`: `79a0d6ed` widened it to
`embedding_profile(embedder).fingerprint()` (`recall/index.py:472`), which covers `model_name` and
`dimension`, so two different unregistered models of equal width no longer compare equal. The
citation is kept pointing at that call site because it is the same position in the tuple, but the
expression it names has changed, which is why this note exists rather than a silent renumber.

⚠️ **This changes nothing about the conclusion drawn from it, for the same reason as #370.** Both
fixes repair a WRITER. Every corpus indexed before them still carries the defective
`index_fingerprint` and the literal `embedding_profile`, and those are exactly the rows an adoption
path reads, so the pipeline attestation sample remains the only sound embedder check available and
must stay a required step. The one practical difference is forward looking: after #381 those
corpora re embed on their next index, because their stored fingerprint no longer matches.

## Scope, so this is not over read

The `memory` tenant measured here is a **static copy**, edited by nothing since indexing, which is
the most favourable possible corpus for a content hash check. A live store would show a real
`changed` population: the working copy of that same directory acquired an edit within hours of this
measurement. So 100 percent is evidence that the mechanism works and that the metadata is complete,
and **not** an estimate of the verified rate on a corpus that is still being written to.

---

## Q2 result

Measured 2026-08-18. **One registered half is confirmed and the other is falsified**, and the
falsified half changes the reasoning behind the design rather than the design.

### ⚠️ Two deviations from the registered method, declared before the numbers

**First, the corpus.** The registration said "build one generation over the `memory` tenant" and
score against "the existing human labelled set". **Those two do not pair.** The only hand labelled
query set in the tree is `recall/eval/queries.json` (20 answerable / 20 unanswerable, plus 6 `trust`
entries that `measure_top_cosines` skips), and every one of its `relevant_ids` resolves to a file in
`recall/eval/corpus` (22 files, verified: zero referenced files missing). It is written for that
fixture corpus, not for the memory store. **No human labelled set exists for the `memory` tenant at
all**, so the registered falsifier could not be evaluated on the registered corpus. The primary
below therefore runs on `recall/eval/corpus`, and the `memory` corpus is reported separately as a
descriptive that needs no human labels. Substituting the corpus is a real weakening: the fixture is
22 chunks of synthetic technical prose, and nothing here licenses a claim about a large corpus.

**Second, the sample size.** The registration said `DEFAULT_PER_CLASS`, which is 40
(`recall/wizard/queryset.py:55`). A 22 chunk fixture cannot supply 40 distinct answerable chunks and
refuses both 40 and 30, so the primary ran at **`per_class=20`, exactly the certification floor**.
The secondary on `memory` ran at the registered 40. The falsified prediction turns partly on a
sample size that was not the registered one.

### Primary: fixture corpus, scored on the human held out set

Embedder `fastembed` bge-small, 384 dim, deterministic. The provisional threshold is fitted on the
generated set **only** and never sees a human label. n = 20 per class, so every rate below is k/20.

| Arm | Threshold | False confidence | False abstain |
|---|---:|---:|---:|
| shipped constant | 0.500 | **1.00** (20/20) | 0.00 (0/20) |
| **provisional, fitted on generated** | 0.674 | **0.05** (1/20) | 0.05 (1/20) |
| oracle, fitted in sample on the human set | 0.684 | 0.05 (1/20) | 0.10 (2/20) |
| ⚠️ control, a threshold that games the metric | 1.010 | 0.00 (0/20) | 1.00 (20/20) |

**FCR improvement over the constant: 0.95 absolute**, against a registered bar of 0.10.
**Confirmed.**

Two things worth more than the headline:

- **The provisional threshold matches the oracle on false confidence and is no worse on false
  abstain** (1 of 20 against 2 of 20, a difference of a single query, and at or below that across
  all 20 seeds below). A threshold fitted on questions a machine invented did as well on human
  questions as one fitted on the human questions themselves. I did not predict that.
- **The constant abstains on nothing.** Every one of the 20 human unanswerable queries scores above
  0.50. A prior internal record measured 0.50 sitting at the 0th percentile of five of six top-1
  distributions, but that record is not in this repository, so treat any generalisation beyond the
  two corpora here as unverifiable from this document. What is verifiable is the 20 of 20 and the
  39 of 40 below.

### Robustness: was seed 0 lucky?

The generated set is seeded, so a single seed proves little. Re generated at **20 seeds**, refitting
each time and scoring all of them on the same human samples:

| | min | median | max | stdev |
|---|---:|---:|---:|---:|
| threshold | 0.650 | 0.667 | 0.674 | 0.0076 |
| FCR on human | 0.05 | 0.05 | 0.05 | 0 |
| false abstain on human | 0.00 | 0.00 | 0.05 | 0.0222 |

**20 of 20 seeds beat the constant by at least 0.10.** Seed 0, the one reported above, sits at the
worst observed false abstain of 0.05, which five of the twenty seeds share; the other fifteen score
0.00.

### Secondary: the corpus the registration actually named

The remote `memory` tenant, 8,716 chunks, bge-large 1024 dim, generated set at the registered
`per_class=40`. No human labels exist, so only the constant can be scored honestly here, which is
enough for the question that matters. FCR at 0.50 is a genuine measurement even in sample because
0.50 is a fixed constant chosen before these samples existed (`recall/eval/calibrate.py:132`).

| | value |
|---|---:|
| generated threshold | 0.622 |
| AUC [95% Hanley-McNeil] | 0.9656 [0.9243, 1.0000] |
| FCR at 0.50 / false abstain | **0.975** (39/40) / 0.000 |
| FCR at 0.622 / false abstain | 0.075 (3/40) / 0.075 (3/40) |

Same direction, same magnitude: the constant lets **39 of 40** unanswerable queries through.

⚠️ **Confound specific to this corpus.** `measure_top_cosines` retrieves at `k=1`
(`recall/eval/calibrate.py:64`), and on this index `query_dense(k=1)` disagreed with `k=200` on the
top score for **3 of 10** probe queries. Every statistic in the table above is computed from those
partly wrong cosines. The direction is far outside that noise; the values are not trustworthy to
three decimals, and the four decimal AUC should not be read as precise.

⚠️ **No artifact was retained for this secondary run.** It printed to stdout and was never written
to a file, so unlike the primary its numbers cannot be re read from disk.

### ⛔ The second registered prediction is falsified

I predicted the provisional threshold "will NOT reach the separability bar that
`Calibration.certified` requires". **It does, on both corpora measured here.**

| Set | AUC | 95% lower bound | n per class | certifies |
|---|---:|---:|---:|:--:|
| generated, fixture | 1.0000 | 1.0000 | 20 | **yes** |
| generated, memory | 0.9656 | 0.9243 | 40 | **yes** |
| human, fixture | 0.9950 | 0.9725 | 20 | yes |

Certification is two independent gates (`recall/calibration.py:229` then `:233`): at least 20
samples per class, **and** the interval's lower bound at or above 0.90. Every row clears both. This
replicates a 2026-08-16 result on a second and third corpus. Nothing here licenses the general claim
that a generated set certifies on an arbitrary corpus: two corpora, one embedder each.

**What that does to the design.** The case for a distinct `PROVISIONAL` status **cannot rest on
statistical failure**, which is what I assumed when I registered it. It has to rest on
**provenance**: the statistics are real and the questions were invented by the same process being
scored, so certification here measures separability on machine questions, not on questions people
ask. `docs/CALIBRATION.md:169-176` already says exactly this: a certified calibration "does not mean
the labelled set was a good one". So `PROVISIONAL` survives and its justification changes, from "the
numbers are too weak to certify" to "the numbers are fine and the query set's origin is not a
human". That is a weaker sounding claim and a more defensible one.

### Controls, all run rather than asserted

1. **Metric gaming.** A threshold of 1.01 scores FCR 0.00 and false abstain 1.00, confirming that
   FCR alone is not a quality measure and that the provisional arm is not winning that way. This is
   why every table above carries both columns.
2. **Noise.** 🔁 **Re run 2026-08-18, because the first attempt was a guard that could not fail.**
   That attempt split 20 human answerable scores into 10 and 10 and reported "does not certify".
   But `Calibration.certified` returns False on `min(n) < 20` (`recall/calibration.py:229`) **before**
   separability is consulted (`:233`), so at n=10 a *perfect* AUC of 1.0 would have failed
   identically. The control demonstrated the sample floor, not discrimination. Re run at 20 per
   class, with a same distribution null built from 40 generated answerable scores drawn from two
   seeds over one corpus, and a positive control through the identical code path:

   | Arm | n per class | AUC | 95% CI | certifies | binding gate |
   |---|---:|---:|---|:--:|---|
   | null (same distribution) | 20 | 0.5487 | [0.3686, 0.7289] | **no** | separability |
   | positive (real generated set) | 20 | 1.0000 | [1.0000, 1.0000] | **yes** | separability |

   Now the separability arm is the gate that fires in both rows, and the pair shows the path
   discriminates rather than merely refusing thin sets.
3. **Determinism.** `generate_offline(seed=0)` reproduced byte identical output.
4. **Retrieval fidelity.** The `k=1` against `k=200` probe was run on the **memory** corpus only
   (3 of 10 disagreed). On the fixture corpus a separate check found 0 empty hit lists and 0
   disagreements across all 40 human queries. The generated set's 40 queries were not probed.

### Verdict against the prediction

| Registered prediction | Measured | Verdict |
|---|---|---|
| provisional beats 0.50 on FCR by ≥0.10 on a held out human set | 0.95 on the fixture; 20/20 seeds clear the bar | **confirmed** |
| provisional does NOT reach the certification bar | certifies on both corpora | ⛔ **falsified** |
| (method) score on the `memory` tenant against a human set | impossible, no such set exists | ⚠️ **substituted** |
| (method) generated set at `DEFAULT_PER_CLASS` = 40 | primary ran at 20, the certification floor | ⚠️ **deviated** |

### Scope, so this is not over read

- The fixture corpus is **22 chunks**. `per_class=20` is exactly the certification floor with two
  chunks spare, and the generated answerable class therefore draws from 20 of 22 chunks, covering
  nearly the whole corpus. That is an easy case.
- With 20 unanswerable samples the smallest non zero FCR is **0.05**, so "provisional matches the
  oracle at 0.05" means both let exactly **one** query through. n is too small to separate 0.05
  from 0.00, and the same resolution argument applies to the 0.05 against 0.10 false abstain
  comparison, which is one query against two.
- One embedder per corpus, and the two corpora use different embedders, so nothing here separates
  a corpus effect from a model effect.
- The memory corpus numbers carry the `k=1` confound above and have no retained artifact.
