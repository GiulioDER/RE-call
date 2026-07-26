# When is a cloud embedder worth it? — study design

**Date:** 2026-07-26 · **Branch:** `research/vocab-gap-predictor` · **Status:** design approved, implementation started

## Why this exists

FINDINGS §8 established a *conditional* rule:

| hit@5 | bge-small (local) | voyage-3 (cloud) | Δ |
|---|---|---|---|
| private memory corpus, 794 docs | 0.348 | 0.630 | **+0.282** |
| PEPs, 746 docs (public) | 0.705 | 0.727 | **+0.022** |

The rule reads: *pay for a cloud embedder when your corpus vocabulary is unusual — internal
codenames, project shorthand, identifiers absent from any pretraining set.*

That is true and it is **not operational**. A reader cannot apply it to their own corpus, and a
reviewer cannot critique it. This study tries to turn "unusual" into a number you can compute
before you spend anything.

There is a second motive, stated so it is not mistaken for a hidden one: this result is the
anchor for a technical-review request to the Voyage team and to the local-embedding community
(bge / sentence-transformers / fastembed). The study is designed to be worth publishing whether
or not anyone replies.

## The question

> What property of a corpus predicts how much a cloud embedder will beat a local one — and is
> that gap closable by adapting the local model instead?

## The design

**Unit of observation:** one corpus. **Response:** `gap = hit@5(voyage-3) − hit@5(bge-small)`,
both measured through the existing `recall.eval.labelled` harness so nothing but the embedder
changes.

**Candidate predictors** (all computable without touching the cloud API):

| predictor | intuition |
|---|---|
| `oov_rate` | share of word *types* the local model's tokenizer shatters into ≥k subwords |
| `vocab_novelty` | share of types absent from a reference technical-English vocabulary |
| `query_overlap` | lexical overlap between questions and their gold documents — low overlap = paraphrase pressure |
| `crowding` | mean nearest-neighbour cosine among documents in local embedding space |
| **`bge_hit@5`** | **the null model — the local embedder's own score** |

### The null model is the point

`gap = voyage − bge`, so the gap is mechanically anti-correlated with `bge` through the ceiling
alone: a corpus where the local model already scores 0.9 cannot show a +0.28 gap. Any vocabulary
metric will therefore correlate with the gap *without explaining anything*.

**The finding is not "predictor X correlates with the gap." It is whether X explains variance
beyond `bge_hit@5`** — partial correlation, local score held fixed.

Both outcomes are publishable, and that is deliberate:

- **X beats the null** → a rule you can apply to a corpus you have never retrieved from.
- **X does not beat the null** → *"skip the vocabulary analysis; just measure your local embedder
  on 30 labelled questions."* Cheaper advice, equally useful, and an honest negative result of the
  same kind as §2 and §3.

A design with only one publishable outcome is a design that will find one.

### How many corpora, and why not eight

§7 and §8 give two corpora. Any metric separates two points perfectly — that is a line through
two dots, and it is the same in-sample defect that §2b already retracted a number for.

Eight is not enough either, which the power calculation says plainly. Partialling out one
covariate leaves `n - 3` degrees of freedom:

| n corpora | df | min detectable \|partial r\| (α=.05) | first Holm test of 3 |
|---|---|---|---|
| 8 | 5 | 0.75 | **0.85** |
| 12 | 9 | 0.60 | 0.70 |
| **20** | 17 | 0.46 | **0.54** |

At n=8 nothing short of a partial correlation of **0.85** survives correction. A null result there
would not mean "no effect", it would mean "underpowered" — a night of compute spent to learn
nothing, and a negative finding we could not honestly publish.

**Corpora (target n≈20):** CQADupStack's ~12 subforums (tex · mathematica · physics · programmers ·
gis · unix · android · english · gaming · stats · webmasters · wordpress) — each its own corpus
with its own qrels, spanning TeX/Mathematica jargon through to plain English, which is precisely
the axis under test — plus NFCorpus (medical) · SciFact · SciDocs · FiQA (financial) · ArguAna
(plain argumentative English, the low-jargon anchor) · PEPs · the private memory corpus.

BEIR ships qrels for all of these, which convert to the harness's `relevant_files` format.

**Subsampling:** corpora are capped (qrels-relevant documents plus random negatives) so the local
embedding runs finish in a night. A smaller haystack is an easier haystack, so absolute scores are
**not** comparable to published BEIR numbers — only the within-corpus gap is, and only because both
embedders see the identical subsample. This must be stated wherever the numbers appear.

## PREREGISTRATION — frozen 2026-07-26, before any gap was measured

Everything below is fixed now, while no `gap` figure exists for any BEIR corpus. Each of these is
a knob that could be turned after seeing results to make a finding appear, which is precisely the
in-sample defect §2b retracted a number for. Changing any of them later is permitted, but only as
a **restatement**: the pre-registered result gets published alongside it.

### The analysis set excludes the corpora that generated the hypothesis

§7's finding — *the cloud embedder wins on idiosyncratic vocabulary* — was **discovered by looking
at the private memory corpus**. A predictor designed with the knowledge that this corpus is both
high-OOV and high-gap is guaranteed to fit it, so scoring it as evidence is circular.

- **Primary analysis: BEIR + CQADupStack only.** Genuinely held out.
- **Reported but excluded:** the private memory corpus (discovery, §7) and PEPs (first
  replication, §8). Both appear in the writeup as context, neither enters the correlation.

At n≈20 the two excluded points cost almost nothing, and their exclusion is what makes the rest
out-of-sample.

### Frozen parameters

| parameter | value | why this value, decided before seeing any gap |
|---|---|---|
| `min_pieces` | **2** | Definitional: 2 means "the tokenizer has no whole-word entry for this", which is exactly the claim being made. 3 is arbitrary — and the smoke test showed 3 gives a *larger* separation (5.2× vs 3.5×), which is the reason it must not be chosen now. |
| `token_budget` | smallest corpus's token count, floor 2 000 | Rank ordering is stable from 16k down to 2k (ρ = 1.000; 0.964 at 1k) across 7 local text corpora. The analysis is rank-based, so the magnitude drift does not propagate — only the floor matters. |
| `crowding` sample | fixed across corpora, set at ingest | Sampling changes what "nearest" means; a sparser sample has more distant neighbours. |
| primary response | `gap = cloud − local` | What is built and tested. |
| secondary response | `headroom_capture` | Ceiling-corrected structurally rather than statistically. Reported always, not only if the primary disappoints. |
| correction | Holm over the 3 predictors | Reported alongside the inter-predictor correlation matrix, because Holm assumes independence and is over-conservative if the predictors turn out to be near-copies. |
| code handling | `strip_code` before `oov_rate`; `code_density` reported | Identifiers shatter like codenames. Scope limit: this removes code *marked up inside prose* (≈ −0.03 on a markdown corpus at ~11% density), not files that are wholly code (−0.002 on `.py`). The study's corpora are prose. |

### RESTATEMENT 2026-07-26 — `MAX_DOCS` 5 000 → 20 000

Recorded as a restatement rather than an edit, and legitimate only because **no gap has been
measured for any corpus**. A larger machine became available, and the honest way to spend capacity
is on a better study rather than a faster one.

What it does *not* change: the cap stays **common across corpora**, so haystack size remains
controlled. That control is the reason a cap exists at all, not the cost saving. Full BEIR corpora
range 3.6k (nfcorpus) → 68k (cqadupstack-tex), a **19× spread**, and a bigger haystack is a harder
haystack that can move the gap by itself. A 20k common cap compresses that to roughly **5.5×**.

What it does change: five corpora (nfcorpus, scifact, arguana, cqadupstack-mathematica,
cqadupstack-webmasters) now fall under the cap and are used whole. Haystack size therefore still
varies, so `analyse_records` reports `haystack_confound` — the correlation between `n_documents`
and the gap, raw and partialled on the local score. Measured and printed, not carried in prose as
a caveat no reader can check.

### Declared in advance: what each outcome means

- **A predictor beats the null (Holm-adjusted p < 0.05):** a rule applicable to a corpus you have
  never retrieved from.
- **No predictor beats the null:** *"skip the vocabulary analysis, just measure your local
  embedder on 30 labelled questions."* Published in the register of §3's null result.
- **Primary and secondary responses disagree:** published as the finding it is — the predictor was
  tracking headroom, or tracking the gap, and which one is now known.

## Compute

A rented box for one night. **Not VPS2 or VPS3** — §7's fine-tune died at 44/96 steps from 629% CPU
beside live systems, and `nice` lowers priority without capping thread count. Every BEIR corpus is
public; the private memory corpus is embedded locally and never leaves this machine.

## Non-goals

- No partnership, logo, sponsorship, credits or compute request in the resulting outreach.
- pgvector and Anthropic are **out of this round**. Both have real give-first material and neither
  has an open question attached; firing at four organisations at once is what makes outreach look
  scattershot.
- No sixth clever metric. The analysis is worth more than another predictor.

## Risks

**The only reproducible number makes the cloud embedder look unnecessary.** +0.022 on the PEPs is
public; the +0.282 sits on a corpus nobody else can see. This is walked into deliberately: lead with
the conditional rule, never either number alone. The cloud model wins exactly where its makers would
predict it wins, and that is a usable result for them.

**A predictor that fails to beat the null may read as a null study.** Mitigated by writing the
negative outcome up as the finding it is, in the register §3 already established.

**Subsample size confounds the gap.** Haystack size has measurable effects elsewhere in this repo
(§10b: false-abstain *rises* as the haystack narrows). If the gap correlates with subsample size
across corpora, that is a confound and must be reported as one, not tuned away.

## Success criteria

Not "we found a predictor." The study succeeds if it returns a defensible answer to *does a
vocabulary metric beat simply measuring your local model* — in either direction — with the
confound above either ruled out or reported.
