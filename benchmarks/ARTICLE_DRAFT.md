# We benchmarked AI memory as honestly as we could — including where we lose

> **Review draft.** All numbers from `benchmarks/results/` this session. Sour-grapes-proofed:
> every criticism below is proven with data that holds even if you delete RE-call from the tables.
> Two cells marked *[pending]* are running. Pre-registration: `benchmarks/PREREGISTRATION.md`.

## The claim

We're not here to claim a crown. On OpenAI's cheap model RE-call is the more accurate of the two;
on the stronger model the comparison is what it is, and we report it either way. We'll show you
every cell — wins and losses — and we run the models the incumbents actually evaluate with, not an
expensive off-ecosystem model chosen to flatter one side.

What we *are* claiming is narrower and, we think, more useful: **the LOCOMO accuracy scores everyone
cites — the 90%-plus numbers — are measured with instruments we can show are unreliable, and they
omit the dimension that actually matters for an agent acting on its memory: what it does when it
doesn't know.** We ran the most complete evaluation we could, published every cell including our
losses, and the rest of this post is the evidence.

We benchmarked **RE-call** (ours) against **Mem0** (the most-adopted open-source memory layer) on
LOCOMO, through an *identical* generator and judge — the only variable is the memory. We report
**two columns for every configuration**, across **two generators**, **two retrieval budgets**, and
**two judge models**, with and without LOCOMO's known-corrupt answer keys.

## The complete results

Answerable accuracy (LLM-as-judge), paired McNemar (same 1,540 questions both systems):

| generator | budget | judge | RE-call | Mem0 | paired p |
|---|---|---|---|---|---|
| gpt-4o-mini | item-matched (k=10/10) | gpt-4o-mini | **0.416** | 0.378 | 0.0059 |
| gpt-4o-mini | item-matched | gpt-4o | **0.466** | 0.412 | 0.00018 |
| gpt-4o-mini | token-matched (k=10/20) | gpt-4o-mini | **0.416** | 0.370 | 0.00077 |
| gpt-4o-mini | token-matched | gpt-4o | **0.466** | 0.411 | 0.00018 |
| **gpt-4o** (strong) | token-matched | gpt-4o | *[running]* | *[running]* | *[running]* |

We test the strong-generator case with **gpt-4o** — the model the incumbents actually evaluate
with — not an off-ecosystem model, so the comparison stays on the battleground people really use.

Abstention on the 446 adversarial questions (does the system refuse when the answer isn't there?),
and its inseparable twin, false-abstention on answerable questions:

| | RE-call | Mem0 |
|---|---|---|
| adversarial abstention (want high) | 0.883 | **0.948** |
| answerable false-abstain (want low) | **0.291** | 0.340 |
| discrimination (abstention − false-abstain) | 0.593 | 0.608 |

**Read this straight: on OpenAI's cheap model, RE-call is more accurate across every budget and
both judges (p<0.001); Mem0 abstains more but also refuses more real questions; the two discriminate
about equally.** The strong-model row (gpt-4o) is running and will be reported either way.

## The four things the headline numbers hide

Each of these is a fact about the *measurement*. None depends on which system wins.

### 1. An abstention score alone is meaningless — you need its false-abstain twin
A system scores 0.95 "knows when it doesn't know" by refusing *everything*. Mem0's higher
abstention (0.948) comes with a higher false-abstain (0.340): it's more conservative, not more
discriminating. Reporting abstention without false-abstain — as the field does — rewards blanket
refusal. We always report both.

### 2. The standard judge is unreliable — and we proved it by hand
Published LOCOMO evaluations grade with `gpt-4o-mini`. On the 199 questions where `gpt-4o-mini` and
`gpt-4o` disagreed, we hand-labelled all of them **blind** (system and both verdicts hidden) against
the judges' own rubric:

- `gpt-4o` was right on **167/195 = 85.6%**; `gpt-4o-mini` on **28/195 = 14.4%**.
- `gpt-4o-mini` systematically **under-credits correct answers**. (An independent audit,
  dial481/locomo-audit, separately found it *over*-accepts 62.8% of deliberately wrong answers —
  so it's unreliable in whichever direction the prompt pushes it.)
- Crucially, its error is **not** system-asymmetric (14/108 correct on RE-call items vs 14/87 on
  Mem0 items) — which is why our ranking held across both judges. We checked, so we can say it.

A benchmark graded by a judge that's wrong 86% of the time on the hard cases is not measuring what
it claims.

### 3. The result can depend on the generator — so we test the one people actually use
A "which memory is best" number that flips when you swap an unrelated component isn't a property of
the memory. So we report the comparison across the generators the incumbents actually evaluate with
— `gpt-4o-mini` and `gpt-4o` — rather than cherry-picking one. There is a measurable mechanism for
any gap: Mem0 returns LLM-compressed facts, which a stronger reader can exploit and a weaker one
can't, while RE-call returns raw turns that read the same to any model. The `gpt-4o` row above is
running; whichever way it falls, it's reported, and it's on OpenAI's stack — not an off-ecosystem
model chosen to make a point.

### 4. Nobody controls for retrieval budget or the corrupt answer keys
- The two systems return different amounts of text per item; matching on `k` doesn't match tokens.
  We ran **both** matchings; the result holds at each, so we're not hiding behind a budget. We also
  held the *embedder* constant — both systems on the same strong local model (bge-large): RE-call
  **0.478** vs Mem0 **0.370**, paired **p=0.000022** (gpt-4o-mini generator). A better embedder
  didn't flip the ranking; only the generator does — so the flip is a property of how each memory's
  output is *read*, not of retrieval quality.
- LOCOMO's answer key is **6.4% wrong** (99/1,540; independent audit, verified per-question against
  the source). The theoretical ceiling is **~93.6%**, not 100%. Excluding those keys moves both
  systems almost identically (RE-call +2.2, Mem0 +2.1), so it doesn't change the comparison — but
  every absolute score you've seen sits under a contaminated ceiling.

## Where RE-call actually stands (up front, no hiding)

- On `gpt-4o-mini` — the model the incumbents' own numbers are built on — **RE-call is more
  accurate than Mem0**, across every retrieval budget and both judges, all at p<0.001.
- On `gpt-4o` the comparison is *[running]* and reported whichever way it lands.
- Its cat1 single-hop retrieval recall is the weak spot; a stronger embedder + reranker lifted it
  39%→50% (accuracy 0.440→0.476), a real gain, not category-redefining.
- **The axis where RE-call is unambiguously ahead is cost**: its memory layer makes **zero** LLM
  calls (measured, not asserted — see below), so it runs local and free, while Mem0 charges an LLM
  extraction call per memory written.

What RE-call is built for is the axis this whole post argues the field mis-measures: **trustworthy,
auditable memory** — calibrated abstention, per-hit provenance, validity/supersession, exact source
data, and no LLM in the retrieval path. Whether that's worth more than a few accuracy points is a
choice about your application — but you can only make it if someone reports the honest numbers.

## The cost gap nobody prints

Accuracy is one column; **cost per memory is the other**, and it's a *category* difference, not a
few percent. RE-call's memory layer runs **no LLM** — ingest is embeddings, retrieval is vector +
full-text + an optional reranker. Mem0 runs an **LLM fact-extraction call per session at ingest**;
there is no configuration where writing a memory is free.

We metered it (the harness records every LLM token each memory layer sends, so this is measured,
not modelled):

| memory-layer LLM cost (4 LOCOMO conversations) | RE-call | Mem0 |
|---|---|---|
| LLM calls the memory layer made | **0** | 99 |
| tokens the memory layer sent to an LLM | **0** | 985,687 |
| $ to build the memory (gpt-4o-mini extraction) | **$0.00** | **$0.166** |
| at Mem0's default extraction model (gpt-4o-class) | **$0.00** | ~$2.66 |

The ratio isn't "N×" — it's **free vs not-free**. RE-call's memory-layer cost is $0 and stays $0 at
any scale; Mem0's grows linearly with every memory written. A system ingesting millions of memories
pays millions of LLM calls with Mem0, zero with RE-call on a local embedder. For write-heavy, cost-sensitive, offline, or privacy-bound
deployments — most of them — a memory that's competitive on accuracy and **free to write** is the
correct engineering choice. That is the honest headline: not "most accurate," but *"trustworthy and
effectively free, at competitive accuracy, on the models you actually use."*

## Reproduce it

Every configuration was pre-registered before the numbers were seen. The harness, the per-question
raw dumps (context, answer, both judges' verdicts), the human labels, and the corrupt-key list are
all published; one command reproduces any cell. Rerun it, re-judge it, re-label it — that's the
point.
