# We benchmarked AI memory as honestly as we could — including where we lose

> **Review draft.** All numbers from `benchmarks/results/` this session. Sour-grapes-proofed:
> every criticism below is proven with data that holds even if you delete RE-call from the tables.
> Every cell is filled and measured. Pre-registration: `benchmarks/PREREGISTRATION.md`.

## The claim

On the two OpenAI readers the incumbents actually evaluate with — `gpt-4o-mini` and `gpt-4o` —
RE-call is the more accurate of the two, and it builds memory at **zero marginal API cost** while
Mem0 pays an LLM call per memory written. That accuracy lead is real but **reader-conditional** —
~4 points on those two readers, narrowing as the reader strengthens and reversing on a third
generator (Claude Sonnet) we ran *after* pre-registration; we show that row too. We'll show you
every cell and every methodological reason the headline scores you've seen don't measure what they
claim.

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

Answerable accuracy (LLM-as-judge), paired McNemar (each system answers the identical question set):

| generator | budget | judge | n | RE-call | Mem0 | paired p |
|---|---|---|---|---|---|---|
| gpt-4o-mini | item-matched (k=10/10) | gpt-4o-mini | 1,540 | **0.416** | 0.378 | 0.0059 |
| gpt-4o-mini | item-matched | gpt-4o | 1,540 | **0.466** | 0.412 | 0.00018 |
| gpt-4o-mini | token-matched (k=10/20) | gpt-4o-mini | 1,540 | **0.416** | 0.370 | 0.00077 |
| gpt-4o-mini | token-matched | gpt-4o | 1,540 | **0.466** | 0.411 | 0.00018 |
| **gpt-4o** (strong) | token-matched | gpt-4o | 1,540 | **0.484** | 0.444 | 0.0065 |

**Every row is the full benchmark** — all 1,540 answerable questions across all 10 conversations,
each system answering the identical set. RE-call is the more accurate of the two in every one, and
the margin survives Holm–Bonferroni correction across all five rows (every row holds; largest
adjusted p = 0.012).

**RE-call is the more accurate of the two across *both* OpenAI generators** — the models the
incumbents actually evaluate with — at every retrieval budget and both judges. But watch the trend
*within* the table: at the gpt-4o judge the lead is +0.054 under the gpt-4o-mini generator and
+0.040 under gpt-4o (both n=1,540) — it shrinks as the reader strengthens, and a third,
stronger-still generator reverses it (§3). This is a property of the reader tier, not a
reader-independent win.

Abstention on the 446 adversarial questions (does the system refuse when the answer isn't there?),
and its inseparable twin, false-abstention on answerable questions — **full benchmark, gpt-4o-mini
generator (n=446 adversarial, 1,540 answerable)**:

| | RE-call | Mem0 |
|---|---|---|
| adversarial abstention (want high) | 0.883 | **0.948** |
| answerable false-abstain (want low) | **0.291** | 0.340 |
| discrimination (abstention − false-abstain) | 0.593 | 0.608 |

(Under the gpt-4o generator, full n, these shift to 0.924/0.955 abstention and 0.294/0.333
false-abstain — abstention is partly a generator behaviour, so we label which one.)

**Read this straight: on both OpenAI readers — the cheap one and the strong one — RE-call is the more
accurate of the two at full n=1,540 (p = 0.0002 to 0.0065, Holm-corrected), at every budget and both
judges. Mem0 abstains slightly more but also refuses more real questions; the two discriminate about
equally.** The one configuration where Mem0 came out ahead — Claude Sonnet as generator — was added
*after* pre-registration and is not in this headline table; we report it in §3 rather than bury it,
because the trend it belongs to (our lead narrowing as the reader strengthens) is itself the finding.

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

### 3. The result depends on the generator — and our lead is reader-conditional
A "which memory is best" number that flips when you swap an unrelated component isn't a property of
the memory — so a benchmark that reports one score without naming the generator is hiding a
variable. We report across the generators the incumbents actually evaluate with, `gpt-4o-mini` and
`gpt-4o`, and RE-call is ahead on both. We then ran a third generator, **Claude Sonnet 4.5, which was
not in our pre-registration**, and Mem0 won it: on the 4-conversation subset (n=584, gpt-4o judge)
RE-call scored **0.565 vs Mem0 0.608**. Put the three side by side and the pattern is monotone —
RE-call's lead is **+0.054** under gpt-4o-mini and **+0.040** under gpt-4o (both n=1,540), and
**−0.043** under Sonnet (n=584); all at the gpt-4o judge. The honest reading is not "Sonnet doesn't
count" but "our accuracy lead is
a property of the reader tier": the stronger the generator, the smaller our edge, until it reverses.
The mechanism is measurable and predicts exactly this — Mem0 returns LLM-compressed facts a stronger
reader can exploit, while RE-call returns raw turns a weaker reader handles better. We keep the claim
to the readers where we hold it, and disclose the one where we don't.

### 4. Nobody controls for retrieval budget or the corrupt answer keys
- The two systems return different amounts of text per item; matching on `k` doesn't match tokens.
  We ran **both** matchings; the result holds at each, so we're not hiding behind a budget. We also
  held the *embedder* constant — both systems on the same strong local model (bge-large): RE-call
  **0.478** vs Mem0 **0.370**, paired **p=0.000022** (gpt-4o-mini generator, 4-conversation subset
  n=584). A better embedder didn't flip the ranking — so any gap is a property of how each memory's
  output is *read*, not of retrieval quality. We also ran Mem0 on **its own documented default
  embedder** — OpenAI `text-embedding-3-small` — at full n=1,540, both arms otherwise identical:
  RE-call **0.42** vs Mem0 **0.366**, paired **p ≤ 0.0014**. Mem0's shipped embedder didn't close
  the gap; it scored *below* its bge-small (0.366 vs 0.378). "You didn't show Mem0 at its best"
  doesn't hold — and that best still cost 272 extraction calls to RE-call's zero.
- LOCOMO's answer key is **6.4% wrong** (99/1,540; independent audit, verified per-question against
  the source). The theoretical ceiling is **~93.6%**, not 100%. Excluding those keys moves both
  systems almost identically (RE-call +2.2, Mem0 +2.1), so it doesn't change the comparison — but
  every absolute score you've seen sits under a contaminated ceiling.

## Where RE-call actually stands (up front, no hiding)

- On **both** OpenAI generators — `gpt-4o-mini` and `gpt-4o`, the models the incumbents' own numbers
  are built on — **RE-call is more accurate than Mem0**, at full n=1,540, across every retrieval
  budget and both judges (p = 0.0002 to 0.0065, Holm-corrected). That lead is **reader-conditional**:
  it narrows as the generator strengthens and reverses on Claude Sonnet (a post-hoc generator, §3),
  so we claim it for the OpenAI reader tier the field benchmarks with, not as a universal result.
- Its cat1 single-hop retrieval recall is the weak spot; a stronger embedder + reranker lifted it
  39%→50% (accuracy 0.440→0.476, 4-conversation subset), a real gain, not category-redefining.
- **The axis where RE-call is unambiguously ahead is cost**: its memory layer makes **zero** LLM
  calls (measured, not asserted — see below), so building and querying memory costs **zero marginal
  API tokens** — you run the embedder locally, and the generator that reads the results costs the
  same for both systems — while Mem0 charges an LLM extraction call per memory written.

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
| $ to build the memory — gpt-4o-mini extraction | **$0.00** | **$0.166** |
| $ to build the memory — gpt-4o extraction (measured) | **$0.00** | **$2.65** |

Both Mem0 figures are metered, not modelled. Two things follow. First, the ratio isn't "N×" — it's
**zero-marginal-cost vs linear**: RE-call's memory layer spends $0 in API tokens and *stays* $0 at
any scale and any model, because it uses no LLM; Mem0's grows linearly with every memory written.
Second, Mem0's cost is *unbounded in quality* — a 16× jump from mini to gpt-4o extraction — while
RE-call never leaves zero. And the kicker: at the gpt-4o tier — where building the full benchmark's
memory cost Mem0 **$7.29** (272 metered extraction calls, 2.62M tokens) and RE-call **$0** — Mem0
still scored **0.444**, below RE-call's **0.484**. You pay more for less. For write-heavy,
cost-sensitive, offline, or privacy-bound deployments — most of them — a memory that's competitive on
accuracy and **free of per-write API cost** is the correct engineering choice. That is the honest
headline: not "most accurate," but *"trustworthy, zero marginal API cost, at competitive accuracy on
the OpenAI readers you actually use."*

## Reproduce it

Every headline configuration was pre-registered before the numbers were seen; the one post-hoc
addition (the Claude Sonnet generator) is labelled as such. The harness, the human labels, and the
corrupt-key list are published, and one command regenerates the per-question dumps (context, answer,
both judges' verdicts) for any cell — those dumps are run-scratch (`benchmarks/results/` is
gitignored), so you reproduce them rather than read ours. Rerun it, re-judge it, re-label it —
that's the point.
