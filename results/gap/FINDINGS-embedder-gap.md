# A cloud embedder wins almost everywhere — and the gap grows with your haystack

**Run:** 2026-07-27 · 17 BEIR / CQADupStack corpora · `bge-small-en-v1.5` vs `voyage-3` · hit@5
**Preregistered:** 2026-07-26, before any gap was measured —
`docs/superpowers/specs/2026-07-26-embedder-gap-predictor-design.md`
**Artifacts:** one JSON per corpus in this directory · `analysis.json` · reproduces from them exactly
on a different machine.

The two corpora that *generated* the hypothesis (the private memory corpus of FINDINGS §7, and the
PEPs corpus of §8) are deliberately **excluded**. Everything below is held out.

---

## 1. The headline: 16 wins out of 17

| | hybrid arm | dense arm (embedder isolated) |
|---|---|---|
| corpora where the cloud model wins | **16 / 17** | 15 / 17 |
| median gap | **+0.059** | **+0.105** |
| mean gap | +0.053 | — |
| range | −0.033 … +0.101 | −0.025 … +0.198 |

Sign test **p = 0.00027**. Bootstrap 95% CI on the mean hybrid gap **[+0.038, +0.068]** — nowhere
near zero.

The dense arm matters because it isolates the embedder: BM25 is embedder-independent and scored
**byte-identically** in both runs of every corpus, which is the cleanest available evidence that the
pipeline varies only what it should.

## 2. The gap grows with the size of the haystack

| corpus size | n | median gap (hybrid) |
|---|---|---|
| < 10 000 docs | 3 | **+0.013** |
| ≥ 17 000 docs | 13 | **+0.062** |

Gap vs document count: Spearman **+0.509**, and **+0.436** after partialling out the local model's
own score. Nearly five times the effect at realistic corpus sizes.

### This restates §8 rather than refuting it

[FINDINGS §8](../FINDINGS.md) measured **+0.022** on the PEP corpus and concluded that on ordinary
technical English a cloud embedder "buys nothing measurable here." That corpus is **746 documents** —
smaller than anything in this study, and its result sits exactly where the small-corpus regime
predicts (`nfcorpus`, 3 633 docs: **+0.019**; `scifact`, 5 183 docs: **+0.013**).

§8 was not wrong. Its **scope was one small corpus**, and the conclusion does not extend to corpora
of the size people actually run. The honest restatement:

> A cloud embedder buys little on a corpus of a few hundred documents and roughly **+0.06 hit@5**
> at twenty thousand. Size, not subject matter, is what moved in this data.

## 3. What we could NOT find: any way to predict *where* it helps more

The preregistered question was whether a computable corpus statistic beats simply measuring the
local embedder. **None does.**

| predictor | partial r (control = local score) | permutation p | Holm p | |
|---|---|---|---|---|
| `oov_rate` | +0.265 | 0.327 | 0.653 | ✗ |
| `query_overlap` | +0.229 | 0.392 | 0.653 | ✗ |
| `crowding` | −0.655 | 0.006 | **0.019** | ✓ then ✗ — §4 |

n = 17, above the preregistered power floor of 12, so this is a null result and not an underpowered
one. The null model is itself substantial: the local score alone correlates **−0.512** with the gap.

**The vocabulary hypothesis fails cleanly.** §7 attributed the effect to out-of-vocabulary jargon.
`oov_rate` does not predict the gap here, and the failure is not hiding anything: it correlates
**−0.015** with corpus size. Excluding the corpus that generated that hypothesis is what made this
checkable at all.

## 4. `crowding` passes the significance test and fails the confound test

`crowding` — mean cosine to the nearest other document in local embedding space — beats the null
under the preregistered analysis. It is also **−0.613 correlated with corpus size**:

```
crowding -> gap | local score  = -0.655   p = 0.006   ✓
crowding -> gap | n_documents  = -0.413   p = 0.112   ✗
n_docs   -> gap | crowding     = +0.229   p = 0.388   ✗
```

Neither survives once the other is held fixed. In this data **"documents are hard to tell apart" and
"the haystack is bigger" are the same axis**, which is unsurprising: more documents means more near
neighbours.

A mechanism is plausible — a larger corpus packs more confusable candidates into the same space, and
a stronger embedder earns more by separating them — but it is a **hypothesis, not a result**. This
design cannot test it. Doing so needs haystack size varied *within* a corpus, content held fixed.
Reported as a confound, not tuned away; the `haystack_confound` diagnostic was preregistered for
exactly this case and caught it.

## 5. What to actually do

**Measure your local embedder on ~30 labelled questions.** It is cheaper than every statistic tested
here and a better estimator than any of them — the local score alone carries −0.512 of the signal.
Then, if your corpus is large, expect roughly +0.06 hit@5 from a cloud embedder, and weigh that
against ~5× query latency, an API dependency, and your documents leaving your infrastructure.

## 6. Per-corpus results

Sorted by size, which is the axis that moved.

| corpus | docs | local | cloud | gap (hybrid) | gap (dense) | oov | crowding |
|---|---|---|---|---|---|---|---|
| `nfcorpus` | 3,633 | 0.696 | 0.714 | **+0.019** | -0.025 | 0.220 | 0.877 |
| `scifact` | 5,183 | 0.780 | 0.793 | **+0.013** | +0.027 | 0.255 | 0.796 |
| `arguana` | 8,674 | 0.567 | 0.534 | **-0.033** | -0.007 | 0.081 | 0.854 |
| `cqadupstack-mathematica` | 16,705 | 0.284 | 0.353 | **+0.070** | +0.107 | 0.260 | 0.760 |
| `cqadupstack-webmasters` | 17,405 | 0.451 | 0.510 | **+0.059** | +0.130 | 0.192 | 0.759 |
| `cqadupstack-android` | 20,000 | 0.533 | 0.625 | **+0.092** | +0.192 | 0.202 | 0.754 |
| `cqadupstack-english` | 20,000 | 0.559 | 0.625 | **+0.066** | +0.105 | 0.122 | 0.738 |
| `cqadupstack-gaming` | 20,000 | 0.688 | 0.730 | **+0.043** | +0.083 | 0.133 | 0.739 |
| `cqadupstack-gis` | 20,000 | 0.468 | 0.543 | **+0.075** | +0.158 | 0.297 | 0.758 |
| `cqadupstack-physics` | 20,000 | 0.518 | 0.576 | **+0.058** | +0.110 | 0.170 | 0.748 |
| `cqadupstack-programmers` | 20,000 | 0.473 | 0.534 | **+0.062** | +0.096 | 0.136 | 0.756 |
| `cqadupstack-stats` | 20,000 | 0.371 | 0.436 | **+0.064** | +0.095 | 0.197 | 0.761 |
| `cqadupstack-tex` | 20,000 | 0.413 | 0.464 | **+0.051** | +0.105 | 0.334 | 0.759 |
| `cqadupstack-unix` | 20,000 | 0.492 | 0.593 | **+0.101** | +0.198 | 0.328 | 0.730 |
| `cqadupstack-wordpress` | 20,000 | 0.367 | 0.456 | **+0.089** | +0.167 | 0.283 | 0.779 |
| `fiqa` | 20,000 | 0.664 | 0.716 | **+0.052** | +0.117 | 0.096 | 0.775 |
| `scidocs` | 20,000 | 0.468 | 0.496 | **+0.028** | +0.040 | 0.138 | 0.795 |

`arguana` is the single loss and is published as such. It is also the outlier on structure — its
"documents" are counter-arguments and its queries are whole arguments, so query and document are the
same length and genre, unlike every other corpus here.

## 7. Scope and limits

- **Subsampling.** Corpora are capped at 20 000 documents, so absolute scores are **not** comparable
  to published BEIR numbers. Only the within-corpus gap is, and only because both embedders saw the
  identical subsample. The cap is a *control* — it compresses a 19× size range to 5.5× — and §2's
  size finding lives inside what remains.
- **The size effect is measured across corpora, not within one.** Corpora differ in subject, query
  style and difficulty as well as size. §2 is a correlation across 17 points, not a controlled
  manipulation.
- **One local model, one cloud model.** `bge-small-en-v1.5` and `voyage-3`. Nothing here says another
  pair behaves the same.
- **Half the questions.** The harness fits calibration on half and scores the other half, so per-corpus
  n is half what the corpus offers.
- **Discovery corpora excluded on purpose.** §7's private corpus generated the hypothesis and §8's
  PEPs was its first replication; scoring either as evidence would be circular.

## 8. Reproduce

```bash
python -c "
import json, glob
from pathlib import Path
from recall.eval.gap_study import analyse_records
from recall.eval.gap_run import write_json
recs=[json.load(open(p)) for p in sorted(glob.glob('results/gap/*.json'))
      if 'summary' not in p and 'analysis' not in p]
out = analyse_records(recs, arm='hybrid')
write_json(Path('results/gap/analysis.json'), out)   # NaN -> null; a bare NaN is not valid JSON
print(json.dumps(out, indent=2, default=str))"
```

Regenerating the corpora and scores from scratch:
`docs/superpowers/specs/2026-07-26-embedder-gap-RUNBOOK.md`.
