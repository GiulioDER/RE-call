# Pre-registration — memory abstention benchmark (RE-call vs Mem0)

Committed **before** any re-judged or full-run result was inspected. The git history of this file
is the evidence for that claim; check it against the timestamps of the artifacts in
`benchmarks/results/`.

The point of this file is narrow and specific: several of the analysis choices below move the
headline number **in our own favour**, and every one of them was fixed here before we could see
what it would do to the result. Anything decided after the numbers are known is disclosed as a
post-hoc choice, not smuggled in as if it had been planned.

## 1. The judge

- **As-published judge: `openai/gpt-4o-mini`** (temperature 0), via OpenRouter. This is the model
  the incumbents' published LOCOMO evaluations use, which is why it is the primary.
- **Stronger judge, pre-registered here: `openai/gpt-4o`** (temperature 0), via OpenRouter.

Reason for a second judge: the independent audit at <https://github.com/dial481/locomo-audit>
scored deliberately wrong-but-on-topic answers with the standard setup and found `gpt-4o-mini`
accepted **62.81%** of them — it rewards vagueness. Since our two arms produce different answer
*styles* (RE-call returns raw conversation turns, Mem0 returns LLM-compressed facts), a
vagueness-rewarding judge is a potential **systematic** bias between arms, not merely noise.

**Commitments:**
- The strong judge model is named above before any re-judge has been run. It will not be swapped
  for another model because the first one gave an unflattering answer.
- **Both** judges' results are published, whichever way they fall.
- The **full disagreement list** (every question where the two judges differ, with gold, answer and
  both verdicts) is published, not a summary statistic.

## 2. Corrupted answer keys

The same audit finds **99 of 1,540** answerable questions have wrong gold answers (plus 57
citation-only issues whose gold answers are correct). Theoretical ceiling ≈ **93.6%**, not 100%.

**Commitments:**
- Accuracy is reported **as a pair**: with all questions (the as-published number) and with the 99
  excluded. Never the excluded-only figure on its own — on the pilot conversation, excluding them
  moved us **+3.7 points in our own favour**, so publishing it alone would be indefensible.
- A **sensitivity check** is also reported dropping only the unambiguous error categories, because
  13 `AMBIGUOUS` + 3 `INCOMPLETE` entries are contestable and a reviewer could fairly argue they
  belong in the denominator.
- The id mapping from the audit's `locomo_{c}_qa{n}` to ours is **verified per entry** against the
  question text and gold answer in `locomo10.json`; a mismatch raises rather than silently
  excluding the wrong questions.

## 3. The statistical test

Both arms answer the **identical** question set, so the comparison is **paired**. Comparing two
independent confidence intervals is the weakest available analysis and will not be used as the
basis for any claim.

**Commitments:**
- Primary test: **McNemar** on per-question outcomes — exact binomial when discordant pairs < 25,
  chi-square with continuity correction otherwise.
- Two families of tests are run (accuracy and abstention), so reported p-values carry a
  **multiple-comparison correction**, stated explicitly.
- A non-significant abstention result will be reported as *"too few adversarial questions to
  detect a difference"*, **not** as *"the systems are equivalent"*.
- **Discrimination J** (adversarial abstention − answerable false-abstain) is **descriptive only**.
  No p-value will be attached to it.

## 4. Retrieval budget

The two systems return different amounts of text per retrieved item, so matching on `k` does
**not** match context budget: at k=10 the pilot gave RE-call ~537 context tokens and Mem0 ~255.

**Commitments:**
- The headline comparison is reported against **context tokens**, not `k` — as a curve for both
  systems — so no single operating point can be accused of being chosen to flatter us.
- Mean and median context length per arm is published alongside every accuracy figure.

## 5. Things we already know cut against us

Recorded here so they cannot look like later concessions:

- **Abstention is generator-driven.** RE-call's trust layer abstained on **0 of 199** pilot
  questions; every refusal came from the shared generator. The original thesis — that RE-call
  uniquely knows when it doesn't know — is **not supported**.
- **Mem0 scored higher on adversarial abstention** in the pilot (0.936 vs 0.809). It also
  false-abstained far more (0.362 vs 0.211); discrimination was effectively tied (0.574 vs 0.598).
- **Supersession/expiry never fire on this corpus.** LOCOMO turn documents carry no `supersedes:`
  and no validity window, so the trust layer's demotion logic is untested here. No claim will be
  made that demotion contributed to any result.
- **Prompts were fixed once and frozen** (commit `db9e20e`) after the pilot showed the generator
  was penalised by unstated answer conventions. They will not be tuned again; iterating prompts
  until the numbers improve is overfitting to the benchmark.

## 6. Data licence

LOCOMO and the audit are both **CC BY-NC 4.0** — attribution required, non-commercial.
