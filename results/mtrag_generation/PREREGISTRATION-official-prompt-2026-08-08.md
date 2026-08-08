# Pre-registration: does the official D.2 prompt close the Task B gap?

**Written before the run finished. Generation was at ~50/842 and no scored output existed.**
Committed ahead of the result so the prediction cannot be adjusted to fit it.

**Prior work** (searched before writing, per CLAUDE.md):
`docs_search(source_type='memory', "over-abstention false refusal prompt instruction says I don't
know on answerable questions cost, abstention prompt sensitivity generation")`. No prior work on
GENERATOR PROMPT WORDING and its false-abstention rate — this experiment is new. Two hits bear on
it directly and one is a trap avoided:

- 🔑 [[project-recall-mtrag-rbalg-probe-2026-08-05]] — *"My 'models are verbose' thesis was
  FALSIFIED."* Verbosity was already tested as an explanation on this exact benchmark and rejected.
  Predicting an RB_alg gain from the 150-word cap would have re-litigated a closed finding; the
  independent character-vs-word check below reached the same conclusion, which is the agreement
  worth having.
- [[project-recall-abstention-bounded-domain-2026-07-24]] — abstention works on far gaps, fails on
  near-miss, six signals measured and dead, *"don't re-litigate by tuning the threshold"*. ⚠️ That
  is RE-call's RETRIEVAL-side abstention layer, a different component from the generator's system
  prompt. This experiment does not touch it and must not be read as reopening it.
- [[project-recall-semeval-task8-abstention-convergence-2026-08-06]] — three independent top
  systems name abstention as the end-to-end bottleneck; none exceeds 25% UNANSWERABLE F1. Context
  for why the lever is worth getting right, not evidence about prompt wording.

## The claim being tested

RE-call's Task B harmonic mean was **0.5508** against the benchmark's own `gpt-4o` at **0.6208** —
same task, same gold contexts, same model. Task B holds retrieval perfect, so RE-call is not in
this number at all: the gap is our generation harness, and specifically the system prompt.

Diagnosis: my `abstain` prompt said *"say that you do not know rather than guessing"*, which fired
on **83 of 709 ANSWERABLE tasks (11.7%)**, and a false abstention scores near zero
(RL_F 0.0726 against 0.8901 when it answered).

The re-run uses the prompt the baselines actually got, quoted from the MTRAG paper
(arXiv 2501.03468, Appendix D.2 "Model invocation").

## Mechanism, and one correction made before predicting

D.2 differs from `abstain` in three ways. Their expected effects are NOT equal, and I got one of
them wrong an hour ago:

1. **Abstention trigger** — theirs names an exact string, `"I do not have specific information"`;
   mine was a vague instruction. **This is the dominant mechanism** and the only one with a
   measured cost attached.
2. **150-word cap** — ⚠️ I claimed this was "load-bearing for RB_alg". **It is not.** The scorer's
   `length()` returns `len(text)`, i.e. CHARACTERS, so our mean "Length 323" is ~52 words, not 323.
   Measured: mean 51.6 words, and only **18/842 (2.1%)** answers exceed 150 words. The cap barely
   binds. I conflated characters with words; any RB_alg movement must come from fixing abstentions,
   not from brevity.
3. **Layout** (`PASSAGE 1..M` then turns) — no mechanism I can argue for either direction. Treated
   as noise.

## Predictions

Derived from the abstained-vs-answered split, assuming the false abstentions are largely fixed:

| quantity | before | if ALL 83 fixed | **predicted** |
|---|---|---|---|
| false abstentions on ANSWERABLE | 83 / 709 (11.7%) | 0 | **< 40 (< 5.6%)** |
| RL_F | 0.7011 | 0.7797 | **0.75 – 0.79** |
| RB_llm | 0.6283 | 0.6854 | **0.66 – 0.70** |
| RB_alg | 0.4117 | 0.4555 | **0.43 – 0.46** |
| **harmonic mean** | **0.5508** | 0.6077 | **0.57 – 0.62**, point **0.595** |

**Ordering prediction** (independent of the levels, and harder to hit by luck): the ABSOLUTE gain
is largest for RL_F, smallest for RB_alg. RB_llm gains more than RB_alg.

⚠️ The "if ALL 83 fixed" column is an upper bound, not an expectation. It assumes the abstained
tasks would score like the average answered task, and the model may have abstained precisely on the
harder ones. The predicted band sits below it deliberately.

## What falsifies this

- **Harmonic mean ≤ 0.56** ⇒ the prompt is not the cause. The diagnosis is wrong and something I
  have not looked at is responsible. Do NOT re-run Task C until that is found.
- **False abstentions stay above ~60/709** ⇒ the exact-string trigger is not the mechanism; the
  model over-abstains for a reason the wording does not control.
- **Harmonic mean > 0.65** ⇒ suspiciously good. That would exceed the recomputed `llama-3.1-405b`
  (0.6277) and approach the human reference band, and I would look for a leak or a scoring
  mismatch before believing it.

## What it does NOT test

Nothing here measures RE-call. Task B is generation over gold contexts. Task C is the only run in
which retrieval participates, and it inherits whichever prompt wins here.

## Provenance

Comparator baselines are recomputed from `mtrag-human/evaluations/reference.json` with the same
aggregation used for our row, per [[project-recall-mtrag-rbalg-probe-2026-08-05]]: recomputing the
published table runs +0.018 to +0.043 high, so only an anchored lift is meaningful and both sides
must be computed identically.

Run: `--prompt official`, `openai/gpt-4o` via OpenRouter, `max_tokens 512`, 842 tasks, manifest at
`taskb_official.predictions.jsonl.manifest.json`.
