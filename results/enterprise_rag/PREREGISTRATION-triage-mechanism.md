# Pre-registration: what is `ratio_8_over_1` actually reading?

**Date:** 2026-08-16   **Status:** predicted, not yet measured

## Registration

```yaml
registration_commit: 1a153cd3ae746786af3eff228d2fabbd5098fa9e  # the commit that ADDED this file
registration_note: a commit cannot carry its own hash, so this line was written by the next commit
label_source: the retrieval fixture's own gold membership (mechanical, no judge)
existing_fixture_digest: b6405b77a2d75472e03c651c2b51b9a62bde4a6d0da6f1c65597091e7492a774
supersedes_nothing: this is a NEW question, not a re-measurement of the triage sweep
```

## The question

[`FINDING-triage-signal.md`](FINDING-triage-signal.md) leaves exactly one protocol-clean number
standing: `ratio_8_over_1` reaches a held-out AUC of **0.642** on `missed_any`, and its own
correction says **"I do not currently have an explanation for why it predicts."**

One sentence: **is that feature reading the disagreement between the dense leg and the fused
ranking, and would the true RRF fused score, which is the actual ranking criterion, predict better
than the dense-score proxy that was measured by accident?**

Both halves are answerable by a number. Neither needs a judge, a generator, or any sampling.

## What is already established, and what is new here

Established, and not re-measured:

- The ranked lists are sorted by **RRF fused rank** ([`retriever.py:301`](../../recall/retriever.py)),
  while each hit's `score` is the **dense cosine**, reassigned at
  [`retriever.py:306`](../../recall/retriever.py) by `_rescored`. All 500 of 500 lists are
  non-monotonic in `score`.
- `ratio_8_over_1` has whole-set AUC **0.6375** on `missed_any`, and 10-seed test-half AUCs from
  0.593 to 0.681.
- The registered sweep's T4 was FALSIFIED, and the shipped `gap_warning` is chance at 0.5015.

New here, and the reason this is a fresh registration rather than more exploration: the previous
document searched **features**. This one tests a **mechanism**, and a mechanism makes a falsifiable
prediction about a quantity nobody has looked at yet.

**The mechanism I am proposing, stated before measuring.** `_rrf`
([`retriever.py:103`](../../recall/retriever.py)) gives each id `1 / (60 + rank + 1)` from every leg
it appears in. With three legs, the fused score is dominated by **how many legs found the chunk**:
a chunk in all three legs scores near `3/61`, and a chunk in one leg cannot exceed `1/61`. So fused
rank is mostly a leg-agreement count, and the dense score at fused rank 8 is largely a readout of
**how far down the fused list you get before you hit chunks the dense leg did not choose**.

If that is right, `ratio_8_over_1` is a disagreement statistic wearing a score-curve costume, and
it should be reproducible from leg agreement alone.

## What I predict

Denominators, because a rate is named by one. `n = 500` questions throughout except where stated;
`missed_any` positives are ~160 of 500. All AUCs are whole-set unless the row says held out.

| # | quantity | point | interval |
|---|---|---|---|
| M1 | mean per-query Spearman rho between fused position and dense score, over 500 queries | **−0.55** | −0.30 to −0.75 |
| M2 | fraction of queries where dense(fused rank 8) > dense(fused rank 1), i.e. `ratio_8_over_1 > 1` | **0.12** | 0.05 to 0.25 |
| M3 | AUC of a pure **disagreement** feature (count of inversions in the dense-score sequence over the top 8) on `missed_any` | **0.60** | 0.53 to 0.68 |
| M4 | AUC of the **true fused score** ratio, `fused(rank 8) / fused(rank 1)`, on `missed_any` | **0.55** | 0.50 to 0.63 |
| M5 | AUC of `legs_hit_at_8` (how many of the three legs contained the rank-8 chunk) on `missed_any` | **0.58** | 0.50 to 0.66 |
| M6 | AUC of `ratio_8_over_1` restricted to the non-inverted regime (`ratio <= 1`), on `missed_any` | **0.60** | 0.52 to 0.68 |

🔑 **M4 is the prediction I most expect to be called wrong, and it is deliberately the opposite of
the handoff's assumption.** The natural expectation is that the real ranking criterion beats the
accidental proxy. I predict it **loses**, because the fused score is quantised into a handful of
levels by leg count, and a feature with four effective values cannot separate as finely as a
continuous cosine. If M4 comes in above 0.6375 the mechanism story above is wrong in an
interesting way and the fused curve becomes the feature to carry forward.

**Ordering predictions:**

- **P1.** M3 and M6 both clear 0.55, i.e. the disagreement reading survives in both halves of the
  feature's range. If only the inverted regime carries the signal, the feature is a rare-event
  detector on ~12% of queries rather than a graded score, which changes how it would be deployed.
- **P2.** M4 < 0.6375, the dense-proxy number. Stated separately from M4's interval because it is
  the comparison, not the level, that decides which feature is carried forward.
- **P3.** The disagreement features and `ratio_8_over_1` are **substantially correlated** (absolute
  Spearman above 0.5 across the 500 queries). If they are near-independent, they are reading
  different things and the mechanism claim fails even if both predict.

## What would falsify this

- 🔑 **M3 at or below 0.53**: leg disagreement is **not** what the feature reads, the proposed
  mechanism is wrong, and `ratio_8_over_1` returns to being an unexplained empirical regularity. I
  report that as loudly as a confirmation, and the correct conclusion is that the feature must not
  be carried to another corpus on a mechanism argument.
- **P3 fails** (correlation below 0.5): same verdict as above, reached a different way.
- **M4 above 0.6375**: the quantisation argument is wrong; good news for the product, bad for this
  story, and the fused curve supersedes the proxy.
- **M2 near 0**: the non-monotonicity that started this whole line of inquiry is rarer than the
  2.726 example suggested, and the inverted regime cannot be carrying much.
- **M1 near −1**: fusion is effectively dense-dominated, there is little disagreement to read, and
  the mechanism has no room to operate.

## How it will be measured

Two passes, and only one of them needs the index.

**Pass A, offline, from the existing fixture `b6405b77…`.** Scores M1, M2, M3, M6 and P3. The dense
scores are stored *in fused order*, so every disagreement quantity is already observable. No
network, no model, no database.

```bash
python -m benchmarks.probe_triage_mechanism --retrieval triage_norerank.json \
  --questions questions.jsonl
```

**Pass B, one retrieval run on VPS2.** Scores M4 and M5, which need quantities the fixture never
recorded. `freeze_triage_retrieval.py` will be extended to write, per hit, the **fused RRF score**
and the **per-leg ranks** (dense, lexical, learned-sparse), alongside the dense score it already
writes. Same index (`ber_voy_lex_12k_full`), same fingerprint inputs, `--reranker none`, 500
questions. Previous identical run: 6h50m.

### Apparatus checks, with known answers, before any number is read

| # | check | known answer |
|---|---|---|
| A1 | digest of the existing fixture, recomputed | `b6405b77…`, unchanged |
| A2 | AUC of a seeded random feature; AUC of the label as a feature | ≈0.50; exactly 1.00 |
| A3 | 🔑 the re-run's ranked `doc_id` sequence vs the existing fixture's, per question | **identical on every question**. Retrieval has no sampling and the fingerprint is unchanged, so any difference means the index or the configuration moved under me and Pass B is not comparable to Pass A |
| A4 | `ratio_8_over_1` recomputed by the new probe vs the value the previous script produced | agrees to 1e-9 on all 500; otherwise the probe is measuring a different feature than the one with the 0.642 |
| A5 | per-leg ranks vs fused score, on the re-run | `fused = sum(1/(60+rank+1))` over present legs, exact to 1e-12. This checks the capture, not the retriever |

A3 and A5 are the two that matter. A3 decides whether Pass A and Pass B can be spoken about in one
sentence at all, and A5 is the difference between capturing the fusion and capturing something that
merely looks like it.

## What I already know

- Everything in [`FINDING-triage-signal.md`](FINDING-triage-signal.md) and
  [`PREREGISTRATION-retrieval-triage.md`](PREREGISTRATION-retrieval-triage.md), including both
  corrections to each.
- **T1 holds:** 176 of 277 missed gold documents (63.5%) were already in the pool, so the misses
  this feature predicts are mostly recoverable by depth.
- The project memory index carries no entry on RRF fusion or leg agreement, and the dogfood corpus
  that serves semantic memory search is not running, so this is not a re-measurement of anything
  recorded. The nearest relevant memo is
  [mixing-embedders-in-one-tenant-is-silent](../../../MEMORY.md), which is about index hygiene, not
  fusion.

## Confounds I can name now

1. ⚠️ **This is the same 500 questions and the same corpus that selected the feature.** Nothing here
   can undo that selection. A mechanism found on the selecting dataset is an *explanation* of an
   existing number, not independent confirmation of it, and this document must not be cited as
   confirmation.
2. **M3 and M6 are themselves chosen quantities.** I am registering three of them rather than
   sweeping, precisely so the count of looks stays small and stateable; the count is three, and any
   further disagreement feature I try afterwards is exploration and must be labelled so.
3. **No reranker in either pass.** The shipped path reranks, which reshapes the top-8 completely.
   Every number here describes the fused pool, not the shipped configuration, exactly as the
   previous run did.
4. **The fused score is bounded and discrete**, so its AUC may be depressed by ties rather than by
   lack of information. The tie-averaging in `auc()` handles this correctly, but M4 being low is
   evidence about the *feature as a feature*, not proof that fusion carries no signal. If M4 is low
   I will also report the number of distinct fused-ratio values, so the tie explanation is
   checkable rather than asserted.
5. **`ratio_8_over_1` has two known hazards** that are being fixed before this runs: a denominator
   that can be negative or near zero (5 of 500 lists contain a negative score), and an `at()` that
   returns 0.0 out of range. Both fixes change feature values on affected rows, so the probe records
   how many rows moved and A4 is checked against the **pre-fix** definition to keep the comparison
   honest.
