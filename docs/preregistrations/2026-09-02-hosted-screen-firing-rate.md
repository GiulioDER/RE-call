# Pre-registration: how often the hosted screen refuses a real memo

**Date:** 2026-09-02   **Status:** predicted, not yet measured

## The question

Over the memory corpora on this machine, what fraction of files does
`recall_hooks.screening.screen` withhold, and of those, how many hold a real credential rather
than something the screen misjudged?

## What I predict

Two rates, each over its own denominator.

| Corpus | Files | Withheld (predicted) | Of those, TRUE positives |
|---|---:|---:|---:|
| `C--Users-gde00-Documents-recall` | 366 | **0**, range 0 to 1 | **0** |
| `C--Users-gde00-Documents-progetto-sentimental` | 933 | **2**, range 0 to 8 | **1**, range 0 to 3 |
| everything else combined | 84 | **0**, range 0 to 1 | 0 |
| **all corpora** | **1383** | **2**, i.e. **0.14%** | 1 |

The direction that matters more than the count: **I predict the screen fires on well under 1% of
files.** A gate that fires routinely is not a cautious gate, it is an ignored one.

I am deliberately predicting low. [[i-over-predict-effect-magnitudes]] records eleven of twelve
predictions on this project falsified by being two to four times too high, so the honest correction
is to take the number I first thought of and quarter it. My first instinct was "about ten across
both corpora"; the recorded prediction is two.

Why the sentiment-agent corpus is where I expect anything at all: the user-level `CLAUDE.md`
records that a session recorded in that project carries *"VPS addresses, the SSH key name, funder
and signer key paths and the full strategy"*, measured at 58 host-address hits and 53 SSH-key hits
in one transcript. But note what those are: **names and paths, not key material**, and this screen
matches key material only. That is precisely why I predict a low number there despite it being the
riskier corpus, and it is the prediction most likely to be wrong.

## What would falsify this

- **More than 2% withheld in either corpus.** That is a screen too noisy to ship: people would
  turn it off or learn to ignore it, and the design brief says the rule set is small on purpose.
- **Any false positive on prose that a reasonable person would call ordinary.** The whole argument
  for prefix-anchored rules is that they do not fire on engineering writing.
- **Zero findings across 1,383 files** would not falsify the screen, but it would falsify the
  claim that it is worth its complexity, and I should say so rather than quietly keeping it.

## How it will be measured

```bash
python scripts/screen_corpora.py          # written for this, read-only, prints counts only
```

- **n = 1,383 files**, every `*.md` under `~/.claude/projects/*/memory/`.
- **Metric: withheld files / files scanned**, per corpus. The denominator is FILES, not findings:
  one memo with three keys is one refused upload, and the upload is the unit the gate acts on.
- **Secondary: findings / withheld file**, and the rule name distribution.
- Each withheld file is then read BY HAND to classify it true or false positive. There is no
  automatic ground truth here and I am not going to pretend there is.

⛔ **The matched text is never printed, logged, or written into this record.** The screen is built
not to return it; the harness must not reintroduce it. A measurement that copies the corpus's
secrets into a markdown file in a public repository would be the exact disclosure the feature
exists to prevent.

**Apparatus check, before trusting any number:** the harness runs first over
`tests/test_hosted_screening.py`'s assembled fixtures, where the answer is known (10 rules, 10
distinct findings). If that does not come back 10, the harness is broken and the corpus numbers
mean nothing.

## What I already know

- No prior measurement of secrets in these corpora exists. `docs/preregistrations/` has none, and
  the two related memos ([[two-stores-for-one-credential-must-share-a-key]],
  [[fixes-leak-outside-the-file-you-edited]]) are about different subjects.
- `docs/SECURITY_MODEL.md:21` states RE-call does not redact or classify content, and that content
  you would not want a co-tenant to see *"should not be indexed in the first place"*. This screen
  enforces that sentence rather than contradicting it.
- 42 unit tests pass, and the wiring is mutation-tested: bypassing `_screened` makes the payload
  assertion fail.

## Confounds I can name now

1. **This measures FIRING, not RECALL.** There is no labelled set of secrets in these corpora, so
   a false NEGATIVE is invisible to this measurement. Whatever comes back, it cannot support a
   claim that the corpus is clean, only that the screen is or is not noisy. The module docstring
   already states the gap; the result must not be written up as though it closed it.
2. **The JWT rule is the likeliest false-positive source.** Base64 of any JSON object begins `eyJ`,
   so two dotted base64 blobs of JSON match without being a token. An engineering corpus that
   quotes an encoded payload would trip it.
3. **I wrote both the rules and the corpora.** Much of this memory is my own prose, written by
   sessions that knew the house rule against pasting secrets. A low number may measure the
   author's habits rather than the screen's precision, and would not transfer to a stranger's
   corpus. That is the single biggest reason not to generalise this result to users.
4. **Placeholder filtering could hide a true positive.** A real key that happens to contain
   `example` or four X's is discarded before counting. I judge this unlikely and cannot rule it
   out from the counts alone.

---

## Result (2026-09-02)

**Status:** measured

| Corpus | Files | Withheld | Rate |
|---|---:|---:|---:|
| `progetto-sentimental` | 933 | **0** | 0.00% |
| `recall` | 366 | **0** | 0.00% |
| `agent-memory-bench` | 68 | **0** | 0.00% |
| the five smaller corpora | 16 | **0** | 0.00% |
| **all** | **1383** | **0** | **0.00%** |

Measured: **0 of 1,383 files**, 0 findings, no rule fired once.
Predicted: 2 files (0.14%), of which 1 a true positive.

**Gap: I over-predicted again, and this time after deliberately correcting for over-predicting.**
My first instinct was about ten across both large corpora; [[i-over-predict-effect-magnitudes]]
says to quarter that, so the record says two; the answer is zero. The correction was in the right
direction and still not far enough. The useful generalisation is narrower than "predict less": for
a question of the form *does my own artefact contain a thing I have a standing rule against*, the
base rate is not small, it is **zero**, and the prior should be zero rather than a small number.

### The apparatus, verified rather than assumed

A zero is exactly where a broken harness hides, so this was checked two ways before the number was
believed.

1. **Known-answer check, in the harness itself:** 10/10 fixtures detected, and a known-clean line
   (`postgresql://user:pw@localhost/recall`) did not fire.
2. **Positive control on real content:** all 1,383 files read back as **6,731,733 characters over
   91,204 lines, 0 unreadable**; then a known credential was planted into a temporary COPY of 12
   randomly chosen real memos, one rule each, and **12/12 were caught**. Each copy was confirmed
   clean before the plant, so the catch is the plant and not the memo. Nothing under any memory
   directory was modified.

### What this does and does not license

**It measures noise, and noise was the design risk.** The stated worry was that a scanner over
prose ABOUT engineering would fire constantly, be ignored, and bury a real finding. Over 91,204
lines of exactly that input, containing example DSNs, host names, git shas, uuids, base64 blobs
and whole paragraphs discussing credentials, it fired **zero times**. That is the strongest
available evidence for the small prefix-anchored rule set, and it is a real result.

**It does not measure whether the screen catches anything**, and confound 3 predicted precisely
this: I wrote both the rules and the corpora, under a house rule against pasting secrets. A number
from my own corpus measures my habits. It does not transfer to a stranger's, which is the only
corpus the feature exists for.

**It cannot support "these corpora are clean."** Confound 1 stands untouched: there is no labelled
set here, so a false negative is invisible to this design. The correct statement is "no
structurally distinctive credential was found", not "there are no secrets".

### On the falsification criterion I set for myself

I wrote that zero findings *"would falsify the claim that it is worth its complexity, and I should
say so rather than quietly keeping it."* Holding to that: **the claim that this catches real
secrets is unevidenced, and I am not going to assert it.** What survives is narrower and worth
keeping on its own: 160 lines, no runtime dependency, a measured false-positive rate of zero over
91,204 lines, and a gate that must exist before a stranger's memory is uploaded to a server I run.
It is kept as an unproven floor, described as one, and the honest place to prove it is a corpus I
did not write.
