# CORRECTION: every earlier number compared raw metrics against conditioned ones

**Supersedes the figures in the first version of `RESULTS.md` and in the pre-registration scoring.**
Found by verifying the comparison rather than the result, after an implausible +0.0249 appeared.

**Prior work** (searched: `docs_search(source_type='memory', "MTRAG answerability label two shapes
capitalised lowercase key scorer idk conditioning metric definition")`):
🔴 [[reference-mtrag-withheld-metadata-two-shapes-2026-08-08]] — **written by me earlier the same
day, and it states the exact fact that predicts this bug**: the release files spell the label
`Answerability`, `sample_data/responses-10.jsonl` spells it `answerability`. Its own summary line
is *"A guard that knows one shape guards half the corpus."*

I recorded that the scorer's sample data uses the lower-case form, then ran that scorer against
release-shaped input and never connected the two. 🔑 **Having the fact, and having written it down,
is not the same as applying it.** The memo existed, was indexed, was searchable, and was mine — and
the search that would have surfaced it was one I only ran after the damage.

## The bug

`judge_wrapper.get_idk_underspec_score` reads the answerability label in **lower case**:

```python
answerability_vals = row.get("answerability", [])
```

The release files (`reference.jsonl`, `RAG.jsonl`) ship it **capitalised**, `Answerability`. Only
`scripts/evaluation/sample_data/responses-10.jsonl` uses the lower-case form. Our scoring input is
built from the release, so the key was never found:

```
Error: answerability is None   x2526   (842 rows x 3 metrics)
label matched                  x0
```

With `answerability = None`, every label branch is skipped and control falls through to
`elif idk_eval == 1: return 0` — which **penalises an abstention** instead of rewarding a correct
one. That is why our `_idk_underspecified` values came out BELOW the raw ones, which is impossible
if conditioning works, and which I noted as a curiosity instead of chasing.

## Why it invalidated the comparison

The published baselines in `RAG.json` carry the **conditioned** metric. Proven, not assumed:

| test | result |
|---|---|
| `rb_agg` exactly 1.0, by label | UNANSWERABLE 120/550, CONVERSATIONAL 100/100, **ANSWERABLE 0/7090, PARTIAL 0/680** |
| rows where `rb_agg`=1.0 also have `rl_f`=1.0 and `rb_llm`=1.0 | **220 of 220** |
| rows where `rb_agg`=0 on UNANS/CONV also zero on the other two | **430 of 430** |

Conditioning is applied identically to all three metrics, and only ever on UNANSWERABLE or
CONVERSATIONAL. So comparing our RAW metric to their CONDITIONED one compares two different
quantities, and understates us precisely where a correct abstention is worth a full point.

## What was checked and found CLEAN

- **Contexts**: our Task C benchmark run uses the release's own contexts — 842/842 identical
  document-id lists, 5 per task. Not a source of error.
- **`rb_agg` formula**: ours is the harmonic mean of `(BertscoreR+1)/2`, `RougeL`,
  `(BertKPrec+1)/2`, matching the documented "harmonic mean of Bert-Recall, Bert-K-Precision and
  Rouge-L". Value ranges agree on both sides.
- **Structure after the fix**: our corrected file reproduces the baselines' signature exactly —
  exact 1.0 only on UNANSWERABLE (16/55) and CONVERSATIONAL (5/10), never ANSWERABLE or PARTIAL,
  all three metrics moving together (21/21), and `RB_llm` reaching 0.

## The corrected numbers

`fix_idk_conditioning.py` recomputes the conditioning from fields already present — `idk_eval`,
`underspecified_eval`, the raw metrics and the capitalised label. **No re-generation, no GPU, no
API calls**: only the final combination step was broken.

| contexts | prompt | harmonic (conditioned) |
|---|---|---|
| gold (Task B) | abstain | 0.5913 |
| gold (Task B) | **official** | **0.6195** |
| benchmark | abstain | 0.5228 |
| RE-call | abstain | 0.5327 |
| benchmark | **official** | **0.5516** |
| RE-call | **official** | **0.5527** |

### What changed in the conclusions

1. **The harness is validated.** Task B official **0.6195** vs their gpt-4o **0.6208** — a gap of
   **0.0013** on identical inputs. The old +0.0249 anomaly, which had us implausibly beating them
   with the same contexts and model, is gone.
2. **RE-call's sign is now consistent.** Against the benchmark's own ELSER contexts: **+0.0011**
   (official prompt), **+0.0099** (abstain). Before the fix the sign flipped between prompts, which
   I had correctly called "not a real effect". It is now positive under both — small, but coherent.
3. ⛔ **We do NOT beat the baselines.** RE-call at 0.5527 against their gpt-4o's 0.5591 is
   **−0.0064**. The reading "we beat every baseline including llama-3.1-405b" was pure artifact and
   was one edit away from being published.
4. **Real abstention rate, by the official judge**: 16/55 = **29%** correct on UNANSWERABLE, inside
   the published 0–32.7% band. My earlier regex estimate of 43.6% was inflated, as suspected.

## Pre-registration, re-scored

| prediction | registered | actual (conditioned) | verdict |
|---|---|---|---|
| harmonic mean | 0.57 – 0.62 | **0.6195** | ✅ in band |
| RL_F | 0.75 – 0.79 | 0.7793 | ✅ in band |
| RB_alg | 0.43 – 0.46 | 0.4573 | ⚠️ in band, but see below |
| RB_llm | 0.66 – 0.70 | 0.7285 | ❌ above band |
| false abstentions | < 40 / 709 | 61 / 709 | ❌ |
| ordering: largest gain RL_F | RL_F > RB_llm > RB_alg | RB_llm **+0.0970** > RL_F +0.0269 > RB_alg **−0.0055** | ❌ |

⚠️ **RB_alg is worse than "in band" suggests.** On raw metrics it rose 0.4117 → 0.4432 and read as
a hit. On the correct metrics it started at **0.4628 — already above my band** — and **fell** to
0.4573. It entered the band by descending into it. I predicted a rise and got a decline.

⚠️ **Scoring caveat**: the predictions were derived from raw metrics, so they are being scored
against a definition they were not calibrated to. The harmonic mean is nearly identical either way
(0.6192 raw, 0.6195 conditioned) so the headline verdict is robust, but the per-metric bands are
not strictly like-for-like.

## 🔑 The lesson

The failure was **visible and I explained it away**. I grepped that judge log, counted exactly 2526
"Error" matches, and classified them as the word "error" appearing inside judge explanations —
quoting one as evidence. A second signal pointed the same way: conditioned values BELOW raw ones,
which cannot happen if conditioning rewards correct abstention.

🔑 **An anomaly you can explain is still an anomaly.** The +0.0249 was the third signal, and the
only one I chased — because it was the one that flattered us and therefore had to be wrong. The two
that would have caught it earlier were both dismissed with a plausible story.

🔑 **Verify the comparison, not just the result.** Both sides were individually correct: our raw
metrics were right, their conditioned metrics were right. The error lived entirely in the join.
