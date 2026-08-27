# Pre-registration: the precision half of draft-time search

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_draft_precision.py`, committed with this record.

## The question

The direction screen (`2026-08-27-three-direction-screen.md`) measured that querying with the
agent's own draft retrieves the governing memo at top-5 for 14 of 14 recorded misses. It measured
**recall of a known-correct memo and nothing else**, and its own result section says no feature may
be built on it until precision is measured.

So: **if an agent searched with its draft before every write, what would it actually receive?**
Three sub-questions, each of which can kill the direction on its own:

1. **Does it work on the production path?** The 14/14 was the LEXICAL leg alone; production serves
   unweighted RRF fusion, where the same screen measured 11 of 14. The deployable number is the
   fused, calibrated, trust-gated one an agent really sees.
2. **What comes back beside the right memo?** Five slots per search, ~8 searches per session.
3. **What happens when there is NO relevant memo?** This is the dangerous case. A draft-time search
   fires on every write, and most writes have no hazard. If the layer returns five confident memos
   anyway, the direction is a noise generator regardless of its recall.

## Why this is affordable and what it costs to run it

Measured before writing this record, from the archive: draft payloads per session are **median 8,
mean 8.9, max 24**, 501 across all on-arm sessions. So the behaviour under test is roughly **8
searches and ~40 retrieved chunks per session**, against the ~2 searches the `check-memory-before-
acting` instruction produced for +107k median input tokens. The cost of this direction is
therefore ~4x that behaviour's query volume, and this record predicts the token consequence
explicitly rather than discovering it later.

## Design

- **Path: the production one.** The stdio MCP server against `probe2_control`, exactly as
  `agent_ab_probe_discoverability.py` drives it, so calibration, the strict trust policy,
  abstention and per-hit verdicts are all in force. Raw `HybridRetriever` legs are recorded
  alongside for comparison but are not the headline.
- **Positives: all 48 on-arm `memory_only` sessions across the 8 task families**, not just the 14
  misses. Every one has a governing memo in the corpus. This widens the screen's population and
  additionally answers whether draft-search BREAKS the four families that goal queries already
  handled.
- **Negatives: the `ctl-stage-by-pathspec` sessions** (locus `claude_md_only`, `governing_memo`
  None): their hazard lives in `CLAUDE.md`, not in the memory corpus, so a correct system finds
  nothing for them. Their draft payloads are the false-trigger population.
- **Queries:** one per recorded `Write`/`Edit`/`Bash` payload, as the screen did.
- **Hard labels, needing no judge:** every returned source is exactly one of (a) THIS session's
  governing memo, (b) ANOTHER family's governing memo, which is known-irrelevant to this draft by
  construction, or (c) other corpus content, unlabelled.
- **Judge, for actionable relevance:** `anthropic/claude-haiku-4.5`, temperature 0, fixed prompt
  committed in the script, shown the draft and one retrieved chunk and asked whether that note's
  failure could strike THIS code and change what the author should do. Actionable relevance, not
  topical similarity. Applied to the top-5 of the best draft query of each of the 14 misses and to
  every negative query's top-5. Every verdict is recorded verbatim.

## What I predict

Per `[[i-over-predict-effect-magnitudes]]`: benefits get the bottom of the band, costs get
multiplied and are predicted HIGH, and the fourth consecutive under-call on an information-side
change (14/14 against a registered 6 to 10) says the recall side may again beat me.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Production path recall@5 on the 14 misses: 9 to 13 of 14.** Fused scored 11 of 14 raw; the calibrated layer can only remove hits, never add them | outside the band |
| 2 | **Recall@5 across all 48 positives: 30 to 42.** The four already-passing families should not regress, but their drafts are shorter (`ts-false-zero-search` averages 2.2 payloads) | outside the band |
| 3 | **Judged precision@5 on the 14 misses: 0.20 to 0.45.** One slot is the governing memo; I expect one or two more to be genuinely applicable and the rest topical noise | outside the band |
| 4 | ⚠️ **THE DECISIVE ONE. Negatives: the layer abstains or returns no `ok` verdict on 0.10 to 0.35 of negative draft queries.** I predict it mostly does NOT abstain, because a 3,900-character draft overlaps lexically with something in any corpus about this repository | above 0.35, which would mean the trust layer already solves the false-trigger problem |
| 5 | **Hard noise: another family's governing memo appears in top-5 for 0.20 to 0.45 of all draft queries** | outside the band |
| 6 | **Cost: median 8 searches per session, and a predicted +150k to +400k input tokens per session** if every result is read. Stated high on purpose: the far cheaper 2-search behaviour cost +107k, and I have under-predicted every cost in this project | outside the band |

## Decision rule, full partition

- **BUILD, unconditionally**, if recall@5 on the misses is >= 9 AND negatives abstain on >= 0.35:
  the layer already separates signal from noise and draft-time search is a prompt-level change.
- **BUILD WITH A GATE** if recall is >= 9 but negatives abstain on < 0.35: the direction works and
  needs a trigger discipline (search only on writes touching unfamiliar APIs, or raise the
  threshold for draft queries). The next registered step is designing that gate, not shipping.
- **DEMOTE** if judged precision < 0.20 AND negatives abstain on < 0.20: an agent would receive
  mostly noise on most writes, and the recall result, however large, does not survive contact with
  a real session.
- **KILL** if production-path recall on the misses is <= 5: the 14/14 was a property of an
  unserved leg and not of anything deployable.
- Whatever the outcome, **the cost prediction is reported against measurement**, because a
  direction that works and costs 400k tokens a session is a different proposal from one that works.

## What I already know

- Recall of the governing memo from drafts: 14/14 lexical, 11/14 fused, 7/14 dense
  (`benchmarks/artifacts/agent_ab/direction-screen.json`).
- Goal queries on the same sessions: 1/14 fused at top-5, the memo ranked 127-142.
- The trust layer abstains when nothing clears the calibrated threshold, and `probe2_control` is
  certified, so abstention is live rather than theoretical.
- `[[search-with-the-draft-not-the-goal]]` states this precision gap as the blocker it is.

## Confounds I can name now

1. **Only 4 negative sessions exist**, ~18 draft payloads. That is a small denominator for
   prediction 4, the prediction the decision most depends on. The record will report the count
   beside the rate and will not quote a percentage of eighteen as though it were a rate over
   hundreds. If the result is borderline, the honest outcome is "needs a bigger negative set",
   which is itself a finding about the benchmark.
2. **The negatives are not neutral code.** `ctl-stage-by-pathspec` is about staging files with git,
   and this corpus is full of git memos. That biases prediction 4 toward false triggers, i.e.
   against the direction, which is the safe direction for a confound to point.
3. **A judge is a model.** Its verdicts are recorded verbatim and the prompt is fixed and
   committed; disagreement with it is auditable rather than arguable.
4. **Recorded drafts come from unwarned sessions.** Correct for this question, as before.
5. **This measures what is RETRIEVED, not what an agent does with it.** Reach and effect are
   different, measured at 0.674 reach against a null task-success delta
   (`[[agent-ab-task-success-result-2026-08-22]]`).

<!-- frozen_above -->

## 🔁 Correction appended 2026-08-27, BEFORE the measurement ran

**Nothing above is edited, including the prediction this corrects.** Prediction 4's rationale says
"a 3,900-character draft overlaps lexically with something in any corpus about this repository".
**That length is wrong.** 3,900 is the median TOTAL across all of a session's payloads; the probe
issues one query per payload, and per-payload length is **median 60 characters, mean 354, maximum
5,125** over the 501 recorded payloads.

The prediction's BAND (0.10 to 0.35) is left exactly as registered and will be scored as written.
Recording the correction here rather than restating the band, because the reasoning behind a
prediction is part of the evidence of what I believed, and a rationale quietly repaired after the
fact would make the prediction unfalsifiable in the way that matters. If the measurement lands
outside the band, the wrong premise is part of why, and that is the useful half.

**Direction of the error:** a 60-character query overlaps with far LESS of a corpus than a
3,900-character one, so the true premise argues for MORE abstention on negatives than I assumed,
i.e. the prediction is more likely to be falsified upward — against my own stated expectation and
in favour of the direction.

**A production constraint this record also did not anticipate:** the server refuses any query over
`MAX_QUERY_CHARS = 4096` rather than truncating it. Measured: **3 of 501** payloads exceed it, no
session loses all of its. Such a payload is recorded as `refused_too_long` and excluded from every
rate rather than scored as a miss; the count is reported in the result.

## Result (2026-08-27)

**Status: measured, positive control passed (memo at rank 1), 3 of 475 queries refused for
length.** 454 positive draft queries over 48 sessions, 18 negative draft queries over 4 sessions,
165 judged slots. Artifact: `benchmarks/artifacts/agent_ab/draft-precision.json`.

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | production recall@5 on the 14 misses: 9 to 13 | **7 of 14** | **falsified, below** |
| 2 | recall@5 across all 48 positives: 30 to 42 | **34 of 48** | confirmed |
| 3 | judged precision@5 on the misses: 0.20 to 0.45 | **0.253** (19 of 75) | confirmed |
| 4 | negatives abstain or return no `ok`: 0.10 to 0.35 | **0.000** (0 of 18, both measures) | **falsified, below** |
| 5 | another task's memo in top-5 for 0.20 to 0.45 of queries | **0.119** (54 of 454) | **falsified, below** |
| 6 | median 8 searches/session, +150k to +400k input tokens | median **10** searches, **~7,870 tokens** of retrieved text | searches confirmed, **token cost falsified by 20 to 50x** |

🔑 **The mechanism, which is worth more than any of the six numbers: the production path loses
the finding in TWO separable places, and neither is retrieval.** The lexical leg reached 14 of 14
and production reaches 7. Of the 7 losses:

- **5 are the TRUST LAYER rejecting a correctly-retrieved memo.** In five of the six
  `ts-lf-rewrite` sessions the governing memo IS in the top 5, every time, with verdict
  `low_confidence`. Retrieval did its job; the calibrated threshold threw the answer away. That
  calibration was fitted on a query set of goal-style questions, and a draft query has a different
  score distribution.
- **2 are FUSION**: in `ts-worktree-import#r1` and `#r2` the memo is never returned at all, because
  unweighted RRF over a leg that ranks it 1 and a leg that ranks it ~127 does not preserve it.

**Per-family production recall on the registered 14:** `ts-sample-covers-tail` 3/3,
`ts-worktree-import` 3/5, `ts-lf-rewrite` **1/6**. The family where the draft signal is strongest
lexically is the family production serves worst.

⚠️ **The false-trigger result is unambiguous and it is the bad news: 0 of 18.** The trust layer
never abstained and never withheld an `ok` verdict on a draft with no relevant memo in the corpus.
Every negative draft query returned five confident hits. With a median of 10 searches per session,
that is ~50 confident slots per session, most of them on writes with no hazard. The judge puts
actionable relevance on those negative slots at **0.056** (5 of 90), so ~94% of what a hazard-free
write retrieves is noise the agent must read and discard.

The consolation is prediction 5: hard noise is **lower** than I predicted (0.119, not 0.20 to
0.45). The five slots are mostly unlabelled corpus text rather than another task's hazard memo, so
the failure mode is dilution rather than active misdirection.

⚠️ **The token cost I predicted was wrong by 20 to 50x, and the correction does NOT make this
cheap.** Retrieved text is ~7,870 tokens per session, not the +150k to +400k I registered. But I
anchored that band on the skill A/B's +107k, which measured the FULL cost of a behaviour change
(extra turns, reasoning about each result), not the retrieved bytes. This run measures only the
bytes. The behavioural cost of ten searches per session is unmeasured and the +107k precedent for
two searches says it is the larger number. Do not quote 7,870 as the cost of this feature.

## ⛔ The decision rule has a GAP, and this result landed in it

Recall 7 of 14 with judged precision 0.253 satisfies **none** of the four registered branches:
BUILD needs recall >= 9, BUILD-WITH-A-GATE needs recall >= 9, DEMOTE needs precision < 0.20 **and**
abstention < 0.20 (precision is 0.253), KILL needs recall <= 5.

I am not inventing a fifth branch after seeing the number. The registered outcome is **no verdict**,
and the honest reading is that the rule was written as though recall and precision would move
together, when the measurement shows they did not.

**This is the SECOND time in this project that a registered partition has left a gap and a result
has fallen into it.** `[[query-side-expansion-reproduces-the-blind-spot]]` recorded exactly this
("3 landed in the unregistered gap between build at >= 5 and dead at <= 2; state the whole
partition next time"), and I read that record while designing this one. Writing "full partition"
above a rule does not make it total. The lesson is mechanical, not attitudinal: **enumerate the
branches over the CROSS PRODUCT of the endpoints, and check that every cell has a branch, before
committing.**

## What this licenses, and what it does not

**Not a build.** Nothing here clears a build bar, and the false-trigger rate alone would make
draft-time search a noise generator on the ~9 of 10 writes per session that carry no hazard.

**The next registered question is narrow and cheap, and it is not about authoring or retrieval:**
the draft signal survives retrieval and dies at the trust threshold in 5 of 7 failures. Fit a
calibration on draft-style queries (or a separate threshold for them) and re-measure recall AND the
0-of-18 false-trigger rate together — because a lower threshold that rescues the 5 will also admit
more of the negatives, and those two move in opposite directions. That trade is the whole question,
it is one calibration run, and it must be preregistered with a partition over both endpoints at
once rather than one after the other.
