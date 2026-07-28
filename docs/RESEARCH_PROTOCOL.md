# Research protocol

How a measurement in this repo becomes a claim. Four parts, in order. Skipping one is what produced
every wrong number this project has had to retract.

## When this applies

Trigger it when a result **will be reported as a finding**, **will change a default**, or **will
land in `FINDINGS.md` / `RESULTS.md` / `README.md`**.

Do **not** trigger it for a refactor, a lint fix, or a run whose output nobody will quote. A
protocol applied to everything gets skipped on the thing that mattered.

---

## 1. Predict the outcome, in writing, before running

A prediction made after the data is a story. Made before, it is a test of the reasoning.

Commit it — file or commit message, either is fine, but it must be **timestamped before the result
exists** so it cannot be quietly revised. State:

- **the number**, with an interval you would defend
- **the reasoning**, in the order the arguments actually carry weight
- **what would falsify it**, and roughly how likely that is

The point is not to be right. A wrong prediction with explicit reasoning is worth more than a
correct one with none, because it says exactly which belief to update.

**Evidence this works:** the embedder-gap study was pre-registered, so `oov_rate` failing at
Holm p = 0.65 landed as a clean negative result. The LOCOMO rerank arm was not, and a contaminated
number was reported three times with confidence intervals before anyone checked the corpus.

## 2. Predict the INVARIANTS, not just the outcome

The outcome prediction tests the **hypothesis**. Invariants test the **apparatus** — and they are
what catch a broken experiment, which no outcome prediction can do.

Write down, before running, what must be true if the machinery is working:

| example invariant | what it proves |
|---|---|
| BM25 scores byte-identical across two embedder arms | the pipeline varied *only* the embedder |
| hit@20 barely moves when a reranker is added | the gain came from reordering, not a config change |
| the indexed table holds exactly N rows | the corpus was indexed once |
| a predictor's null correlates ≈0 with corpus size | the null is clean, not an artefact |

**Assert them in code where you can.** An invariant a human remembers to check is a comment; one
the runner enforces is a guard. `recall/eval/locomo.py` refuses to index over an existing tenant
for exactly this reason, after two concurrent runs doubled a corpus (11,764 rows against a correct
5,882) and depressed every depth by ~0.05 without erroring.

⚠️ **This is the part that does the heavy lifting, and it is not optional.** The worst failure in
this repo's history was invisible to outcome-prediction: the corrupted numbers were plausible,
internally consistent, and produced a *believable finding* about multi-hop retrieval that was
entirely manufactured. Only a row count caught it.

**Verify the artifact, not the process.** "Did the job finish?" and "does the output have the
properties it must have?" are different questions, and only the second one is evidence. Exit code 0
is not a measurement.

## 3. Fix the decision rule before the data

State what each possible outcome will *mean*, and what you will *do*, before you can see which one
arrived:

> switch the default only on ≥0.02 with disjoint CIs; below that keep the shipped model.

Without this, every result gets a story. With it, the recommendation cannot be reverse-engineered
from the number.

Declare the **uninformative** outcome too — the one that means "could not tell". The gap study
pre-declared that below n=12 a null means *underpowered*, not *no effect*, and the runner prints
that verdict itself rather than leaving it to be noticed.

## 4. Score the prediction afterwards

A prediction never compared against the outcome is theatre. Append a `## Result` section to the
prediction file with:

- **correct / wrong / partially**, against the interval you committed to
- **was it right for the right reason?** You can be right by luck and wrong for an instructive
  reason, and those are the two cases worth writing down.

Example worth keeping: before the rerank arm, §10b was used to argue reranking would probably fail
— that measurement showed the same cross-encoder scoring *below* plain cosine. It succeeded
enormously. Logging "wrong" teaches nothing. Logging *"wrong because an abstention-**signal** result
was transferred to a **ranking** task — a category error"* transfers.

---

## Template

```markdown
# Prediction: <what is being tested>
Written <date>, before <artifact> exists.

## Already measured
<the numbers this builds on>

## Prediction
<number + interval per metric>

## Reasoning
<arguments, heaviest first>

## Invariants — must hold if the apparatus works
- [ ] <invariant>   ← assert in code where possible

## Where I could be wrong
<falsifiers + rough probability>

## Decision rule, fixed in advance
| outcome | action |

## Result   ← appended after
<verdict, and whether the reasoning held for the right reason>
```

## The limit, stated honestly

**Anchoring.** Having predicted 0.80, it is tempting to see 0.79 and stop looking. The protection is
the two mechanical parts — the decision rule and the invariant assertions — because neither depends
on judgement exercised at the moment the number appears.

And the sharper one: **this protocol disciplines reasoning; it does not verify data.** Part 2 is the
only part that does, which is why it is the part to skip last.
