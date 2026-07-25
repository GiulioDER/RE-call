# Adversarial review — `ARTICLE_DRAFT.md` (RE-call vs Mem0 on LOCOMO)

**Stance:** hostile. The brief was to *disqualify* the two headline edges — "zero API tokens" and
"more accurate than Mem0 on a public benchmark" — the way a Mem0 maintainer or an HN skeptic would.
**Scope:** desk review, **no API spend**. Every number below was recomputed **independently** from
the published raw dumps in `benchmarks/results/` (own scripts, not `analyze.py`/`locomo_audit.py`).
**Date:** 2026-07-25.

## Verdict

I attacked both edges across nine vectors and **could not disqualify the core numbers**. All five
accuracy cells and all five p-values reproduce exactly from the raw per-question dumps, survive the
multiple-comparison correction the pre-registration promised, and hold on the subset of questions
where both judges agree. The cost edge is genuinely metered (RE-call's memory layer makes **zero**
LLM calls; Mem0's `$7.29` reconciles to the cent).

**One claim must be narrowed:** "RE-call is the more accurate of the two" is **reader-tier
conditional** — the lead shrinks as the generator strengthens and *reverses* on Claude Sonnet, a
config that was run post-hoc, lost, and is excluded from the headline. Plus one overstated word
("free") and one omitted-but-harmless correction.

| # | Attack | Verdict | Evidence |
|---|--------|---------|----------|
| 8 | Point estimates cooked | **SURVIVES** | 5/5 cells reproduce: 0.4162 / 0.3779 / 0.3695 / 0.4844 / 0.4435 |
| 8 | p-values inflated | **SURVIVES** | McNemar (χ²+cc) reproduces: 0.0059 / 0.00018 / 0.00077 / 0.00018 / 0.0065 |
| 8 | Won't survive the promised correction | **SURVIVES** (procedural gap) | Holm–Bonferroni over 5 rows: all hold, max adj p = 0.012. Article never *states* the correction §3 pre-registered |
| 3 | "Accurate" config ≠ "free" config | **SURVIVES** | Headline arm = `fastembed bge-small` (local). Voyage only in the labelled cat1 ablation |
| 5 | Wins by feeding more context | **SURVIVES** | Item-matched context ≈ 486 tok (RE) vs 466 (Mem0); token-matched gives Mem0 *more* and it still loses |
| 4 | Margin is judge noise | **SURVIVES** | On the 1,369 both-judges-agree questions: 0.440 vs 0.399, +0.041, p=0.006 |
| 6/7 | "Zero tokens / $7.29" asserted not measured | **SURVIVES** | Metered: RE memory-layer = 0 calls / 0 tok; Mem0 = 272 calls / 2.62M tok → $7.29 |
| 9 | Corrupt-key exclusion favours RE-call | **SURVIVES** | 99/99 matched; RE +0.0224 vs Mem0 +0.0211 — symmetric |
| 1 | **Accuracy claim's generality** | **⚠️ WOUNDED** | Sonnet flip, reader-tier trend — below |
| 2/ext | Mem0 not run as-shipped; single benchmark | **RESIDUAL** | Open caveats — below |

---

## Reproduction evidence

### Accuracy table (n = 1,540 answerable, all 10 conversations)

Recomputed by pairing per-`question_id` on `is_adversarial=false`; McNemar with continuity
correction, matching the article's method.

| Row | Generator | Budget | Judge | RE-call (mine / article) | Mem0 (mine / article) | p (mine / article) | Holm adj p |
|---|---|---|---|---|---|---|---|
| R1 | gpt-4o-mini | item k=10/10 | gpt-4o-mini | 0.4162 / 0.416 | 0.3779 / 0.378 | 0.00586 / 0.0059 | 0.0116 |
| R2 | gpt-4o-mini | item k=10/10 | gpt-4o | 0.4662 / 0.466 | 0.4117 / 0.412 | 0.000183 / 0.00018 | 0.00085 |
| R3 | gpt-4o-mini | token k=10/20 | gpt-4o-mini | 0.4162 / 0.416 | 0.3695 / 0.370 | 0.000774 / 0.00077 | 0.00227 |
| R4 | gpt-4o-mini | token k=10/20 | gpt-4o | 0.4662 / 0.466 | 0.4110 / 0.411 | 0.000175 / 0.00018 | 0.00085 |
| R5 | **gpt-4o** | token | gpt-4o | 0.4844 / 0.484 | 0.4435 / 0.444 | 0.00650 / 0.0065 | 0.0116 |

Every cell and every p-value matches. Under Holm–Bonferroni across the five (the correction §3 of
the pre-reg promised but the article omits), **all five still clear α=0.05**.

### Judge robustness (mini generator)

Partitioning the 1,540 answerable questions by whether the two judges agree on each system's answer:

| Subset | n | RE-call | Mem0 | margin | McNemar p |
|---|---|---|---|---|---|
| Both judges agree (clean) | 1,369 | 0.4397 | 0.3988 | **+0.041** | 0.0056 |
| Contested, gpt-4o-mini judge | 171 | 0.2281 | 0.2105 | +0.018 | 0.78 (ns) |
| Contested, gpt-4o judge | 171 | 0.6784 | 0.5146 | +0.164 | 0.0064 |

The win lives on the 89% of questions where the verdict is judge-independent, at the same +0.04
margin. On the contested tail only the *stronger* judge separates the systems — which corroborates
§2 ("gpt-4o-mini under-credits"), it doesn't undermine it.

### Cost (metered `usage.memory_layer`, full 10-conv gpt-4o run)

| | RE-call | Mem0 |
|---|---|---|
| memory-layer LLM calls | **0** | 272 |
| memory-layer tokens | **0** | 2,617,362 in + 74,484 out |
| $ @ gpt-4o (2.50 / 10.00 per 1M) | **$0.00** | 2.617M·2.50 + 74.5k·10 = **$7.29** |

`$7.29` is metered, not modelled — it is the recorded `memory_layer` usage priced at gpt-4o rates.
RE-call's `memory_layer` block is literally `{calls: 0, prompt_tokens: 0, completion_tokens: 0}`.

### Corrupt-key exclusion (audit, 99 score-corrupting = 156 − 57 WRONG_CITATION)

Matched independently by `(question, gold)` text: **99/99**. Excluding them:

| | full | excl-99 | Δ | accuracy *on* the 99 |
|---|---|---|---|---|
| RE-call | 0.4162 | 0.4386 | **+0.0224** | 0.091 |
| Mem0 | 0.3779 | 0.3990 | **+0.0211** | 0.071 |

Symmetric (+2.2 / +2.1), and both systems are penalised near-equally by the mis-keyed questions.
The exclusion is not a lever in RE-call's favour.

---

## The one crack: "the more accurate of the two" is reader-conditional

`PREREGISTRATION.md` fixes the *judges* (gpt-4o-mini primary, gpt-4o strong) but says **nothing
about the generator dimension**. Claude Sonnet is not in it. Sonnet was therefore run post-hoc,
Mem0 won, and it is excluded from the headline table as "off-ecosystem, nobody uses it." It *is*
disclosed in prose — so this is not hidden data — but two problems remain:

1. **The exclusion is post-hoc.** The pre-reg's own rule (intro) is that post-hoc choices are
   "disclosed *as* a post-hoc choice, not smuggled in as if it had been planned." Framing Sonnet as
   "nobody uses it" reads as a principled a-priori exclusion, not "we chose to drop it after seeing
   we lost."

2. **The trend is the finding.** Recomputed at the gpt-4o judge:

   | Generator | n | RE-call − Mem0 margin |
   |---|---|---|
   | gpt-4o-mini | 1,540 | **+0.055** |
   | gpt-4o | 1,540 | **+0.041** |
   | Claude Sonnet 4.5 | 584 | **−0.043** (RE 0.565 vs Mem0 0.608) |

   Monotone: the stronger the reader, the smaller the lead, until it inverts — exactly what the
   article's own mechanism predicts ("Mem0 returns LLM-compressed facts a stronger reader can
   exploit"). A skeptic reprints the Sonnet row and the unqualified headline is finished.

   *(The Sonnet row is 4-conversation, n=584 — not fully comparable to the 10-conv rows. It
   establishes the flip's existence and direction, not a precise magnitude at n=1,540.)*

**Recommended restatement:** *"On the gpt-4o-mini and gpt-4o readers the field benchmarks with,
RE-call leads Mem0 by ~4 points (p<0.01, Holm-corrected); the lead narrows as the reader strengthens
and reverses on Claude Sonnet — so this is a property of the reader tier, not a reader-independent
win."* That keeps the win where it's real and pre-empts the counter-example.

## Secondary

- **"Free" overstates.** Zero *API* tokens at the memory layer is true and metered — but "free"
  hides local embed compute/latency, and the claim is scoped to the memory layer (an application
  still pays a generator LLM to read the retrieved context). Prefer **"zero marginal API cost to
  build and query memory."**
- **State the correction.** §3 pre-registered a multiple-comparison correction; the accuracy table
  prints five raw p-values without one. It survives Holm (above) — so add the sentence, don't
  change the numbers.

## Residual (not closeable on desk / no-API)

- **Mem0 is not benchmarked as-shipped.** The headline runs Mem0 on the local `bge-small` embedder;
  Mem0's documented default is OpenAI `text-embedding-3-small`. `systems.py` supports that arm but
  no result file for it exists. The bge-large control (RE-call still wins by more) makes a flip
  unlikely, but "was Mem0 shown at its best?" is the fair question a maintainer will ask.
- **External validity.** The accuracy edge rests on one benchmark, and the article's own mechanism
  ties it to LOCOMO's "answer sits verbatim in a turn" structure. A synthesis-heavy benchmark
  (where fact-compression helps) could read differently. State that the accuracy claim is
  LOCOMO-scoped.

---

## Appendix — file → cell map (for reproduction)

| Cell | RE-call file | Mem0 file |
|---|---|---|
| R1/R3 RE (mini judge) | `recall_openai-gpt-4o-mini_10conv_20260724T110019Z.partial.jsonl` | — |
| R1 Mem0 | — | `mem0_openai-gpt-4o-mini_10conv_20260724T120442Z.partial.jsonl` |
| R3 Mem0 (k=20) | — | `mem0_openai-gpt-4o-mini_10conv_20260724T151540Z.partial.jsonl` |
| R2/R4 RE (4o judge) | `recall_10conv_rejudged_gpt4o.json` | — |
| R2 Mem0 / R4 Mem0 | — | `mem0_10conv_rejudged_gpt4o.json` / `mem0_k20_rejudged_gpt4o.json` |
| R5 (4o gen) | `recall_openai-gpt-4o_10conv_20260725T172349Z.partial.jsonl` | `mem0_openai-gpt-4o_10conv_20260725T172547Z.partial.jsonl` |
| Sonnet | `recall_sonnet_4conv_rejudged_gpt4o.json` | `mem0_sonnet_4conv_rejudged_gpt4o.json` |
| Cost | `usage.memory_layer` in the R5 files | same |
| Corrupt keys | `audit_data/locomo_errors.json` (error_type ≠ WRONG_CITATION) | |

Answerable accuracy = mean(`correct`) over `is_adversarial=false`. McNemar discordant pairs:
b = RE-correct ∧ Mem0-wrong, c = RE-wrong ∧ Mem0-correct; exact two-sided binomial ≈ χ²+cc for
these n. Scripts used for this review are external to the repo (independent recompute).
