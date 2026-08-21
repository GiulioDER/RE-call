# Pre-registration: can a model backed extractor recover a supersession that rule based extraction could not?

Written 2026-08-14, **before either arm has been run and before `adjudication.csv` has been labelled**.
Branch `claude/truth-extraction-prereg` off `0a5da42`. Committed ahead of every result so the
prediction cannot be revised to fit it.

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

## Correction appended 2026-08-21: the instrument holds 37 distinct rows, not 38

**Nothing above this line is edited.** Every count, interval and citation above stands as written
on 2026-08-14, including the ones this section contradicts, because what they record is what was
believed before either arm ran. The correction goes underneath.

Two things moved under this document while it sat unmerged, both in
`benchmarks/labelling/truth_extraction/adjudication.csv`.

**A1 is effectively done.** The table above records "adjudication rows, currently unlabelled | 38".
37 of the 38 now carry a verdict: Y=10, N=27.

**The 38th is a duplicate, and it is the blank one.** Item 7 and item 29 carry a byte-identical
`evidence_sentence` and the same `candidate_target` (`pep-0324`). Item 29 is labelled `N`, item 7
is unlabelled. The file has 38 rows and **37 distinct (evidence, target) pairs**, so every distinct
row already has a verdict.

Consequence for scoring, which is not a rewrite of any prediction above: wherever this document
says the precision instrument is **38 rows**, the denominator actually available is **37**. P4's
interval, invariant I8's non vacuity check and the upper falsifier at 0.95 are all stated against
38 and should be **scored against 37 with the gap noted**, never by changing the number above.

One more property of the frozen input, recorded before anyone reads the evidence column literally:
4 of the 38 sentences (items 7, 14, 29, 30) begin with a stray `'+ ` or `'- `, which is diff
context that survived extraction. Two of those four are already labelled, so the labeller coped,
but the artefact is in the instrument rather than in a rendering of it.

**Item 7 is deliberately still blank.** It needs a human verdict, or a deliberate decision to drop
it as a duplicate of item 29. This instrument's whole value is that its negatives are not model
produced, so a verdict supplied by the thing under measurement would destroy what it measures.

---

## Result

*Appended after the arms run. Record correct / wrong / partially against the committed intervals,
**and whether the reasoning held for the right reason.** A prediction never compared against the
outcome is theatre.*
