# Pre-registration: can a model backed extractor recover a supersession that rule based extraction could not?

Written 2026-08-14, **before either arm has been run and before `adjudication.csv` has been labelled**.
Branch `claude/truth-extraction-prereg` off `0a5da42`. Committed ahead of every result so the
prediction cannot be revised to fit it.

## Registration

```yaml
registration_commit: 938caad28409b1543b1670be75e9040d808f736c
registration_authored: 2026-08-14T19:30:04+00:00
gold_manifest_digest: 9e53dc7b20ee18cd1386eb25d2053bdaa4f026d42a621c8b581c25b113bad297
gold_manifest_questions: 51
```

I5 is asserted against this block by `tests/test_prereg_authority.py`, which refuses any arm
artifact generated at or before `registration_authored` and any gold manifest whose digest has
moved. The commit is the anchor a reader can check without trusting this file: it was pushed to
`origin/claude/truth-extraction-prereg` on the public repository before any arm ran, so
`git show 938caad` dates it independently of whatever the working tree now says. That external
check matters because the branch is expected to reach `master` as a squash, after which
`938caad` is no longer reachable from `master` and only the pushed branch carries the date.

**Prior work** (searched before writing, per `benchmarks/EXPERIMENT-CONVENTION.md`):

- 🔑 `recall/fix.py:1-31`. The decisive prior and the reason this experiment exists. Rule based
  extraction of **this exact relation** on a real 792 memo corpus: 60 memos closed a decision in
  prose against **2** declared frontmatter edges, the tool proposed **ZERO** edges after its refusal
  rules, and of the four candidates that survived the mechanical rules **all four were wrong on
  review** (one reported speech, two superseding a claim or scope *inside* the target rather than the
  target, one hedged, whose author when asked said *augments*).
- ⚠️ `benchmarks/beam/dedup_probe.py:20-25` concluded "the relation is semantic, not declared, so no
  amount of parsing prose will find it." That is about BEAM, where users never write the words at
  all. Here the words **are** written, 60 of them. Different claim. Do not merge them.
- `benchmarks/archive/preregistrations/PREREGISTRATION-peps-rerank-pool.md:26` establishes PEPs as
  this repo's public reproducible corpus: 733 `.rst`, isolated pgvector on a per worktree port.

---

## Already measured

Arm independent, no model and no human judgement. From `results/truth_extraction/census.json`,
generated 2026-08-11 against `python/peps` at `5981b2a2`, validated at its write site by
`benchmarks/labelling/truth_extraction/artifact_contract.py:validate_census`.

| quantity | value |
|---|---|
| private corpus prose closure markers | 60 |
| private corpus declared `supersedes:` edges | 2 |
| private corpus rules arm precision | **0 of 4** |
| PEPs `n_files` | 733 |
| PEPs `n_header_edges` | **47** |
| PEPs `n_prose_marker_files` | 209 |
| PEPs `n_marker_without_header` | **175** |
| PEPs `n_restated_in_prose`, either end | **8** |
| PEPs `recall_ceiling`, either end | **0.1702** |
| **restated by the superseded PEP itself** | **3** |
| **operative recall ceiling for the frozen gold input** | **3 / 47 = 0.0638** |
| frozen gold manifest | 51 questions, digest `9e53dc7b20ee18cd` |
| adjudication rows, currently unlabelled | 38 |
| transplanted private failure fixtures | 4 |

### The operative ceiling, and why it is 0.064 rather than 0.17

`build_gold.py:42-53` freezes **only the superseded PEP's body** as the question input, with the
comment *"The extractor's INPUT is the superseded PEP's prose, so that is what the hash covers."*

The census counts an edge as restated if **either** end states it, and records the split in its own
`_provenance.note`: of the 8, **3 are stated by the superseded PEP and 5 only by the successor**. For
those 5, the sentence the census counted **is not in the input the extractor is given**.

So an extractor reading the frozen gold question can recover at most **3 of 47 edges**. Recall on
PEPs is not a measurement of the model. It is a property of how PEP authors write, and it is
published as such.

---

## Scored already, and wrong: the census prediction

The plan that commissioned this census predicted the ceiling at **0.60, interval [0.45, 0.80]**.
Measured: **0.1702** either end, **0.0638** operative. Three of five census quantities landed outside
their stated intervals (`n_header_edges` low at 47 against [55, 95], `n_prose_marker_files` high at
209 against [90, 200], `n_marker_without_header` high at 175 against [50, 140]).

**The belief that failed:** that a document declaring a relation in a structured header would usually
also narrate it in prose.

**Why it was wrong, and what it updates:** authors who have a field for a fact use the field and move
on. Prose restatement is not a constant of writing, it is what happens when there is **nowhere else
to put it**. The more structured the corpus, the less its prose restates. PEPs have a
`Superseded-By:` header, so they restate 17% of the time and only 6% on the side that matters. The
private memo corpus has no header field at all, and restates 60 times against 2 declarations, which
is the same law read from the other end.

This is recorded here rather than quietly corrected because a wrong prediction with explicit
reasoning says exactly which belief to update, and this one now predicts where the feature is worth
deploying: corpora without a structured field for the relation.

---

## What each instrument can and cannot measure

Stated before the numbers exist, because the temptation afterwards is to read whichever one moved.

| instrument | n | measures | cannot measure |
|---|---|---|---|
| 47 gold edge questions | 47 | recall, **bounded at 3** | precision, and any recall claim about the model |
| 38 adjudicated `marker_without_header` rows | 38 | **precision**, the primary | recall, since these have no authored header |
| 4 transplanted fixtures | 4 | the four private failure modes, publicly | anything statistical |
| private 792 memo corpus, deferred | ~60 | recall, the only corpus that can | anything auditable by a reader |

**Precision is measured on the 38 adjudicated rows, not on the 47 gold questions.** On the gold
questions a well behaved arm reading a superseded PEP that never names its successor should propose
nothing at all, so its precision there would be computed over a handful of proposals and would be
uninformative by the decision rule below whichever way it fell.

---

## Predictions

**Primary, the 38 adjudicated rows.** Precision counts `status == "candidate"`; `requires_review` is
reported as a separate referral rate and never folded in
(`recall/reasoning_proposals/_metrics.py`, `ASSERTED_STATUSES` / `REFERRED_STATUSES`).

| # | quantity | arm | point | interval |
|---|---|---|---|---|
| P1 | precision | R1 rules | 0.80 | 0.60 to 0.95 |
| P2 | precision | M1 model | 0.80 | 0.65 to 0.92 |
| P3 | Δ precision, absolute | | 0.00 | **abs < 0.15** |
| P4 | proposals made, R1 | | 8 | 2 to 20 |
| P5 | proposals made, M1 | | 22 | 10 to 34 |
| P6 | referral rate, M1 | | 0.15 | 0.05 to 0.40 |
| P7 | targets naming a file outside the corpus, M1 | | **0** | exactly 0 |

**Secondary, the 47 gold edge questions.**

| # | quantity | point | interval |
|---|---|---|---|
| P8 | recall, M1 | 0.043 (2 of 47) | 0.000 to 0.064, **capped by construction** |
| P9 | recall, R1 | 0.021 (1 of 47) | 0.000 to 0.064 |

**The public bridge, and the one that decides shipping.**

| # | quantity | point |
|---|---|---|
| **P10** | transplanted fixtures REFUSED by M1 | **4 of 4, exactly** |

**Deferred, the private 792 memo corpus, only if that host frees up.**

| # | quantity | point | interval |
|---|---|---|---|
| P11 | recall vs the 60 markers, M1 | 0.55 | 0.35 to 0.75 |
| P12 | precision, M1 | 0.75 | 0.55 to 0.90 |

**Ordering predictions**, independent of the levels and much harder to hit by luck:

- **O1.** M1 proposes **more** than R1 (P5 > P4) while the two stay within 0.15 on precision. The
  model's value is coverage, not judgement; precision is held by the shared deterministic post
  filters both arms pass through.
- **O2.** M1's residual false positives are **majority hedged or partial scope**, not target
  resolution. Target resolution errors are refused mechanically, which is what P7 pins.
- **O3.** R1's residual false positives are **majority reported speech**, because PEP 0 and PEP 1
  narrate other PEPs' supersessions and `_is_index` (`recall/fix.py:110`) does not match the stem
  `pep-0000`.

---

## Reasoning, heaviest first

1. **Precision is held by the shared deterministic post filters, not by the proposer.** Both arms
   pass through `recall/fix.py:220` `propose_fixes`: the target must resolve to exactly one file, self reference
   is dropped, an existing declaration is never overwritten, duplicates collapse. That is why P3
   predicts equality and why P7 is exactly 0. If M1's precision diverges sharply from R1's, suspect
   the filter is not being applied to it before believing the model did something.
2. **The model's one large lever is target resolution, and PEPs barely need it.** On the private
   corpus 56 of 60 markers never became proposals because `_REF` demands a literal reference within
   40 characters. PEPs use a conventional `:pep:`NNN`` form a regex handles. **This is precisely why
   a PEPs result cannot settle the model's value**, and why P11 and P12 exist.
3. **Hedging is the residual, and I predict the model does not fix it.** *"Supersedes/augments"*,
   where the author said *augments*. The correct output is a refusal, and refusing is the behaviour
   models are measured worst at in this repo (`results/mtrag_generation/PREREGISTRATION-official-prompt-2026-08-08.md:31`).
4. **P1 is 0.80 rather than 0.00 because the private prior does not transfer.** The 0 of 4 is a fact
   about personal memos: hedged, chatty, full of inline code. PEP prose is edited and conventional.
   Transplanting the number would be the category error `docs/RESEARCH_PROTOCOL.md:85` names.

---

## Invariants, and the vacuous version of each

Every one is asserted in code, and **every one is mutated and watched go red before it is believed**.
Enumerate the mutants from the source, not from this list; a previous round in this repo mutated a
written list, killed 17 of 25, and a later pass enumerating from source found 26 survivors.

| # | invariant | assert how | vacuous version to avoid |
|---|---|---|---|
| I1 | the model never sees an authored declaration | capture the bytes **at the provider boundary** with a recording fake and scan for `^(Superseded-By\|Replaces\|supersedes\|valid_from\|valid_until)\s*:`; plus an unstripped positive control that must score **higher** | asserting on the prompt template, or on the loader's output rather than what the provider received |
| I2 | determinism | two runs **with the claim cache disabled**, plus assert the fake engine's call counter incremented | two runs that both hit the cache, which passes without the model being deterministic |
| I3 | fixed point | assert the first pass is **non empty first**, then apply, then assert the second adds nothing | asserting zero overall, which passes if extraction silently broke |
| I4 | prose immutability | `read_bytes()` through the same excise function the writer uses, plus BOM, CRLF and mode; plus a **mutate one character positive control** | `before.strip() == after.strip()`, or comparing parsed bodies that already discard the difference |
| I5 | labels frozen before arms | this file's and `gold.manifest.jsonl`'s git commit timestamps precede every arm artifact, checked by the runner | a comment claiming "labelled first" |
| I6 | direction | feed a **deliberately inverted** gold set and assert precision collapses to ~0 | testing direction only on R1, where a regex fixes it |
| I7 | cost auditability | `validate_cost_claim` at the write site, plus `not (n_calls > 0 and cost == 0.0)` | a validator that runs only in tests |
| I8 | non vacuity | both arms propose at least 1 on the 38 rows; the gold denominator equals `census["n_header_edges"]` | `assert proposals is not None` |
| I9 | counts | file and chunk counts against the frozen census | recomputing the count from the same run |

I9 exists because this repo's worst failure was two concurrent runs doubling a corpus, 11,764 rows
against a correct 5,882, depressing every depth by about 0.05 without erroring, **caught only by a
row count** (`docs/RESEARCH_PROTOCOL.md:49`).

---

## What falsifies this

- **P10 fails**: M1 proposes an edge on any transplanted fixture. Ship nothing above reviewing aid,
  regardless of every other number. Four documented failures, each one a refusal the rules arm got
  wrong; proposing any of them is worse than the regex it replaces.
- **P3 fails low** (M1 precision below R1 by more than 0.15): check P7 first. If the residuals are
  target resolution, the shared post filter is not reaching M1. That is an apparatus failure, not a
  result.
- **P8 exceeds 0.064**: **impossible by construction.** If observed, the arm is reading something
  other than the frozen input, or the gold denominator is wrong. Stop and fix the apparatus.
- ⚠️ **Upper falsifier: M1 precision at or above 0.95 on the 38 rows.** That would exceed the rate at
  which the adjudicators agreed with each other. Before believing it, in order: confirm I1 asserts on
  provider boundary bytes; run the unstripped positive control and confirm it scores **higher**,
  because if stripped and unstripped are equal the strip is inert; confirm the gold digest predates
  the first model call; run the inverted gold control. **Publish nothing until all four pass.**
- ⚠️ **Second upper falsifier: precision 1.00 at fewer than 10 proposals.** An arm proposing 3 and
  getting 3 right has not beaten a rules arm proposing 0; it has reproduced it with extra steps. That
  is the uninformative outcome, not a win, and P4's lower bound of 2 says I consider it live.

---

## Decision rule, fixed in advance

Keyed on the **Wilson lower bound** on precision (`recall/eval/metrics.py:187`), not the point
estimate. Wilson rather than bootstrap because a percentile bootstrap of a small degenerate sample
returns `[1.00, 1.00]`, which is the exact shape this experiment is at risk of producing.

| outcome | verdict | action |
|---|---|---|
| any transplanted fixture proposed (P10 fails) | **fails the public bridge** | reviewing aid at most, whatever else holds |
| any invariant I1 to I9 fails | **APPARATUS FAILURE** | publish no number. Precedence over every row below |
| fewer than 10 proposals in either arm, or Wilson half width above 0.30 | **UNDERPOWERED** | report "could not tell". Do **not** choose a tier. The runner prints this verdict itself |
| M1 proposes 0 | **VACUOUS ARM** | refuse. A non measurement, not a null |
| M1 precision at or above 0.95 and n at or above 20 | **SUSPICIOUS** | run the four upper falsifier checks in order first |
| precision below 0.50 | net cost to a reviewer | lint pointer only: file plus evidence sentence, no target, no `--apply` |
| 0.50 to 0.80 | **reviewing aid** | human in loop mandatory, `--apply` refused. This is `recall/fix.py:13`'s stated posture and it stays |
| 0.80 to 0.95, Wilson lower at or above 0.70 | **batch reviewable** | `--apply` permitted only with `--reviewer` and `--note` |
| at or above 0.95, Wilson lower at or above 0.90, n at or above 20 | **high confidence** | **still no auto apply.** The harm is asymmetric: a wrongly accepted edge declares the live memo stale and demotes it beneath the one it replaced, the exact failure the trust layer exists to prevent (`recall/fix.py:29`) |

**The uninformative outcome is declared live, not hypothetical.** The operative recall ceiling is 3,
the precision instrument is 38 rows, and P4's interval starts at 2. I expect a real chance that PEPs
returns UNDERPOWERED on precision as well, and in that case the honest output of this experiment is
the census finding plus the four fixture refusals, with the model's value left unsettled until the
private corpus arm runs.

---

## What this does not settle

- **The private memo corpus.** PEPs is a third corpus. A positive result supports the FORM of the
  claim, that a model resolves targets a regex cannot, and does not prove it on personal memos. The
  four transplanted fixtures are the only public evidence about that corpus.
- **Recall, at all, on PEPs.** Bounded at 3 of 47 by how PEP authors write.
- **The `contradicts`, `same_entity` and `status` claim kinds.** Neither `contradicts` nor
  `same_entity` has a frontmatter key, and none has a retrieval consumer. They are scored on parser
  accuracy alone and their numbers must never be aggregated with supersession.
- **Whether a reviewer accepts proposals at a sustainable rate.** That is a different measurement.

---

## Provenance

```
census:  python -m benchmarks.labelling.truth_extraction.census \
           --peps-dir <clone>/peps --peps-sha 5981b2a292610104eb30735423504c52fe454650 \
           --clone-date 2026-08-11
gold:    python -m benchmarks.labelling.truth_extraction.build_gold --peps-dir <clone>/peps ...
```

`python/peps` at `5981b2a292610104eb30735423504c52fe454650`, cloned 2026-08-11. Census generated
2026-08-11T14:09:10+00:00 against RE-call `7993b3e`. Gold manifest digest
`9e53dc7b20ee18cd1386eb25d2053bdaa4f026d42a621c8b581c25b113bad297`, 51 questions.

Both arms run against a pgvector container on a per worktree port bound to `127.0.0.1`. Do not use
5432: the default compose file binds it on all interfaces and it has previously resolved to a native
Postgres on this machine, producing false reds.

---

## Result

Scored 2026-08-15 against `arm_R1_rules.json`, `arm_M1_model.json` and `arm_P10_fixtures.json`,
generated 2026-08-15 at 14:07:39Z, 14:07:54Z and 14:08:14Z respectively, all three from `recall`
at `179e54e3` against `python/peps` at `5981b2a2`.

**The decision: the feature ships as a reviewing aid at most.** P10 failed. The model proposed an
edge on two of the four transplanted fixtures, and the decision table's top row is unconditional:
"any transplanted fixture proposed leads to fails the public bridge, reviewing aid at most,
whatever else holds". Nothing below changes that, and nothing below is allowed to.

**Neither arm may be given a tier.** R1 decided 8 proposals and M1 decided 2, against a registered
floor of 10 in either arm, so both artifacts read `UNDERPOWERED`. The honest report is "could not
tell", not "the model is bad". M1's Wilson half width is 0.329, which trips the second underpowered
clause on its own.

### Two corrections to how this was first written up

Both were found by reading this file rather than the summary of it, and both are the reason the
identifiers are now derived mechanically.

1. ⚠️ **P1 was reported as falsified, and it is not.** The first write up said "R1's upper bound of
   0.694 sits just below P1's predicted floor of 0.70". P1's predicted floor is **0.60**. The 0.70
   is the decision rule's gate on the Wilson **lower** bound for the batch reviewable tier, a
   different quantity on a different bound, and comparing R1's *upper* bound against it is a
   category error. R1's interval [0.137, 0.694] overlaps the predicted [0.60, 0.95] across
   [0.60, 0.694], so the data do not exclude the prediction. P1 is **wrong on the point estimate
   and not falsified**, which is the weaker and correct claim.
2. **The fixtures result was published as P7 and is P10.** P7 is a different registered prediction
   on a different instrument. The prose was right throughout and only the identifier was wrong,
   which is the shape that survives review. `tests/test_prereg_authority.py` now refuses an
   artifact whose published id does not resolve to the matching row here.

### The invariant row is not scored here, and that is an open gap

⚠️ The decision table's second row gives **APPARATUS FAILURE precedence over every row below it**:
if any of I1 to I9 fails, publish no number. This section does not report the status of I1, I2,
I3, I4, I6, I7, I8 or I9, and I am not going to assert a clean bill I have not audited one by one.
What is true today:

- **I5 is asserted for the first time**, by `tests/test_prereg_authority.py`, and it passes. Until
  the two branches were joined it was not even expressible.
- **I8 is partly unassertable here.** Its second half reads "the gold denominator equals
  `census['n_header_edges']`", and the 47 gold question arm was never run, which is the same fact
  P8 and P9 are unscored for. Its first half holds: both arms proposed at least one.
- The remaining invariants have tests in `tests/test_truth_extraction_*.py`, and the suite is
  green, but **nobody has walked I1 to I9 against those tests and confirmed the mapping**. That
  audit is outstanding and it is the next thing to do to this experiment.

**Why the headline stands anyway.** P10 does not depend on the precision apparatus. It is four
documents, four extractions and a refusal count, its validator refuses the one shape that would
inflate it (any batch failure at all, not merely all of them), and the artifact records an empty
`rejection_rungs` for every fixture, so no resolution rung stood in for a semantic refusal. The
precision numbers are the ones an unaudited invariant could move, and they are already published
as UNDERPOWERED, which is to say as no number at all.

### Score

| # | registered | measured | score |
|---|---|---|---|
| P1 | R1 precision 0.80, [0.60, 0.95] | 0.375, Wilson [0.137, 0.694], n=8 | **wrong on the point, NOT falsified**: intervals overlap on [0.60, 0.694] |
| P2 | M1 precision 0.80, [0.65, 0.92] | 0.00, Wilson [0.000, 0.658], n=2 | **wrong on the point, NOT falsified**, and only by 0.008 of overlap |
| P3 | abs Δ precision < 0.15 | 0.375 | **wrong**, and it fails low, which the file said to diagnose with P7 |
| P4 | R1 proposes 8, [2, 20] | 9 | **correct** |
| P5 | M1 proposes 22, [10, 34] | 2 | **falsified**, and by a count rather than an interval |
| P6 | M1 referral rate 0.15, [0.05, 0.40] | no extraction path emits a review required status | **unscored** |
| P7 | M1 out of corpus targets, exactly 0 | no counter exists | **unscored**, holds only by construction |
| P8 | M1 recall 0.043 | the 47 gold questions arm was never run | **unscored** |
| P9 | R1 recall 0.021 | as P8 | **unscored** |
| **P10** | **4 of 4 fixtures refused** | **2 of 4** | **FALSIFIED. This is the result** |
| P11, P12 | private 792 memo corpus | the host was not available | **not run**, explicitly, never null |
| O1 | M1 proposes more than R1 | 2 against 9 | **falsified, in the opposite direction** |
| O2 | M1's residual errors are majority hedged or partial scope, not target resolution | 2 of 2 partial scope | **held** |
| O3 | R1's residual errors are majority reported speech, via PEP 0 and PEP 1 | at most 1 of 5, and not by that mechanism | **falsified** |

**Six of twelve numbered predictions could not be scored at all**, half of them, and that is a
finding about the design rather than an inconvenience. P6 and P7 registered quantities nothing
counts, P8 and P9 registered an arm that was never built, and P11 and P12 registered a corpus that
was not reachable. Only P1 to P5 and P10 carry a score.
A prediction whose instrument does not exist is not a prediction; it reads as one until someone
tries to score it. Registering the instrument alongside the number is the change that follows.

⚠️ **P7 deserves its own line, because P3's diagnostic depended on it.** This file says that if P3
fails low, check P7 first, since target resolution residuals would mean the shared post filter is
not reaching M1, which is an apparatus failure and not a result. P3 did fail low. P7 cannot be
checked: the model never sees a target outside the corpus, because `extract_file_claims` is handed
`corpus_names` and the resolution rung drops the rest before anything is counted. So P7 holds by
construction, which is the vacuous form this file warned about in its own invariant table, and the
diagnostic it was supposed to power is unavailable. The evidence that this is not an apparatus
failure comes from elsewhere: both of M1's false positives resolved to real corpus files, and the
fixtures arm records an empty `rejection_rungs` for all four, so no rung did the work there either.

### Did the reasoning hold for the right reason?

**Mostly not, and the one place it did is the most useful thing here.**

**O1 is falsified in the opposite direction, and it was the load-bearing belief.** The registered
reasoning was "the model's value is coverage, not judgement; precision is held by the shared
deterministic post filters". M1 proposed **2** where R1 proposed **9**. The model was not more
generous and worse judged, it was far more conservative. Every downstream expectation rested on
this, including P5's floor of 10, which is why P5 missed by a factor of five rather than a little.

**O3 was specific and specifically wrong.** It predicted R1's residual false positives would be
majority reported speech, naming the mechanism: PEP 0 and PEP 1 narrate other PEPs' supersessions
and `_is_index` does not match the stem `pep-0000`. Of R1's five false positives, at most one reads
as reported speech, and it comes from PEP 3108, a module removal index, not from PEP 0 or PEP 1.
The named mechanism did not fire once. The actual residual is a mixture of five: one explicitly
partial (item 18, "This PEP has **partially** been superseded by :pep:`3137`"), one jointly
conditioned (item 28, "Combined with :pep:`345`, the current proposal supersedes :pep:`262`"), one
plain statement naming two targets (item 15, "superseded by :pep:`345` **and** :pep:`376`"), one
plain statement naming a single target (item 17), and the reported speech fragment (item 29).

🔑 **O2 held, and it held on two corpora independently.** It predicted that M1's residual false
positives would be majority partial scope rather than target resolution. Both of M1's false
positives on the 38 rows are partial scope:

- item 20, PEP 642 on PEP 634: "the `__match_args__ is None` handling **in this PEP replaces the
  special casing of** `bool`, ... **in** :pep:`634`". It replaces a mechanism *inside* PEP 634, not
  PEP 634.
- item 28, PEP 376 on PEP 262: "**Combined with** :pep:`345`, the current proposal supersedes
  :pep:`262`". PEP 376 alone does not supersede it.

And both of M1's fixture proposals are the two partial scope fixtures, `partial_scope_claim` and
`partial_scope_scope`. It refused `reported_speech` and `hedged`.

**So the model has one characteristic error: it proposes an edge broader than the sentence
supports.** All four of its proposals across both corpora are that error, and three of the four
are the same narrow form, a claim about something *inside* a document read as a claim about the
document (item 20, and both fixtures). The fourth, item 28, is the same error in a different
shape: the supersession is real but *jointly conditioned*, and the model dropped the condition.
Calling all four "inside the document" would overstate it, and the two shapes share the thing that
matters, which is that the proposal asserts more than the evidence sentence does.

This is the same error `recall/fix.py` recorded on the private corpus, where two of the four rule
based survivors were "superseding a claim or scope inside the target rather than the target". A
precision point estimate over 2 proposals says nothing. This says where the work is, and it is the
only thing here I would act on.

⚠️ **A registered piece of reasoning went unscored until review caught it, and it was wrong in the
model's favour.** Reasoning item 3 above predicted that hedging would be the residual and that
*the model does not fix it*, on the grounds that refusing is what models are measured worst at in
this repository. The model refused the hedged fixture. On one data point that is not a
vindication, but it is a registered prediction that failed in the direction nobody guards against,
and it belongs in the score rather than in the part of the write up that only counts the misses.

### What this does not settle, restated now that the numbers exist

- **The model's value, at all.** It was said in advance and it is still true: PEPs cite each other
  as ``:pep:`NNN```, a conventional form a regex handles, so the target resolution lever a model
  provides is largely absent. On the private corpus 56 of 60 markers never became proposals for
  exactly that reason. This corpus cannot measure the lever, and it did not.
- **Whether M1's conservatism is a virtue or a defect.** Proposing 2 where 9 were available is
  either good judgement or a broken proposer, and 2 decided proposals cannot tell the difference.
  O1's failure is the strongest reason to run the private arm.
- **Precision, to any useful width.** 8 and 2 decided proposals. The instrument was 38 rows and it
  yielded 10 scored proposals across both arms combined.
- **Anything about `contradicts`, `same_entity` or `status`.** Unchanged from the pre-registration.

### The uninformative outcome, declared live in advance, is what happened

This file predicted it: "with a precision instrument of 38 rows and a recall ceiling of 3, PEPs may
well return UNDERPOWERED, in which case the honest output is the census plus the four fixture
refusals." That is very nearly the outcome, with one substantive correction to it. The fixtures did
not all refuse. Two of four proposed, so the honest output is the census, **the failure of the
public bridge**, and the partial scope mechanism that O2 predicted and both corpora confirm.
