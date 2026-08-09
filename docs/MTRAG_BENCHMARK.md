# MTRAG: multi-turn RAG, scored by someone else's judge

**Prior work** (searched before writing, per the project protocol):
`docs_search(source_type='memory', "MTRAG Task A B C generation retrieval abstention harmonic
baselines conditioned")`. Binding results:
[[project-recall-splade-learned-sparse-measured-2026-08-06]] (the SPLADE lever and the *scope* of
the closed reranker decision), [[project-recall-mtrag-rbalg-probe-2026-08-05]] (the anchored-lift
constraint), [[incident-mtrag-raw-vs-conditioned-metric-comparison-2026-08-09]] (the metric
definition bug that invalidated the first version of every number here).

> **The one-line summary: we do not top this benchmark.** RE-call's retrieval beats the benchmark's
> own retrieval by a hair, our end-to-end score sits just below the published `gpt-4o` baseline, and
> the largest single lever on the whole task turned out not to be ours. All of that is below, with
> the numbers.

---

## 1. What MTRAG is, and why it is worth running

[MTRAG](https://github.com/IBM/mt-rag-benchmark) (IBM, TACL 2025) is a human-generated **multi-turn**
conversational RAG benchmark: 842 tasks over four document corpora (ClapNQ, Cloud, FiQA, Govt),
with published baselines for nine systems and an official evaluation harness.

Three properties make it the right second opinion for a memory layer:

1. **Multi-turn.** The query is a conversation, not a sentence. That is the shape a memory layer
   actually faces, and it is not what a single-shot IR benchmark measures.
2. **It scores abstention as a first-class outcome.** Tasks are labelled ANSWERABLE, PARTIAL,
   UNANSWERABLE or CONVERSATIONAL, and a correct "I do not have that information" on an unanswerable
   task earns a **full 1.0 on all three metrics**. Nearly every other benchmark treats a refusal as a
   miss. This project has claimed a calibrated abstention layer for a year, measured on its own
   harness. MTRAG is the first place that claim faced **someone else's judge**.
3. **Three separable tasks.** A (retrieval only), B (generation from gold contexts), C (full RAG).
   Because B holds retrieval perfect, it isolates *our harness* from *our retrieval*, which is what
   makes the C number interpretable at all.

Metrics are `RL_F` (RAGAS faithfulness), `RB_llm` (RADBench LLM judge) and `RB_alg` (harmonic mean of
Bert-Recall, Bert-K-Precision and RougeL). The leaderboard figure is the harmonic mean of the three.
The official judge is `gpt-4o-mini-2024-07-18`, hard-coded.

⚠️ **Every baseline in this document was recomputed by us from the release**, not copied from the
paper. Recomputing runs +0.018 to +0.043 high against the published table
([the anchored-lift constraint](../results/FINDINGS.md)): the aggregation is right, the instance set
is not identical to the published one. So each comparison below is an **anchored lift**, our row
against their row through the same code, and **none of these numbers may be quoted against the
published leaderboard directly.**

---

## 2. Task A — the retrieval ladder, and what each rung costs

777 judged dev queries, four domains, `candidate_k=100`, embedder `BAAI/bge-small-en-v1.5`, learned
sparse `prithivida/Splade_PP_en_v1` (Apache-2.0). Every figure recomputed by a scorer written
separately from the harness; all six arms agree to **±0.000000**.

| arm | sparse leg | rerank | nDCG@5 | R@100 | cost |
|---|---|---|---|---|---|
| `hybrid_lexical` | Postgres FTS | — | 0.2930 | 0.6865 | free, local |
| `dense_only` | — | — | 0.3024 | 0.6736 | free, local, fastest |
| `splade_only` | SPLADE | — | 0.3286 | 0.7074 | free, local |
| `hybrid_both` | FTS + SPLADE | — | 0.3348 | 0.7347 | free, local |
| **`hybrid_splade`** | SPLADE | — | **0.3573** | **0.7377** | **free, local** |
| **`hybrid_splade_voyage`** | SPLADE | Voyage rerank-2.5 | **0.4342** | **0.7668** | paid API |

**The span is the point.** One engine, same 777 queries, nDCG@5 from **0.2930 to 0.4342**. That is a
**48% relative spread** across configurations of the same system, and every rung is a named flag.

Deltas that are readable straight off this table, all against `hybrid_splade`, the free default:

| against | R@100 delta | reading |
|---|---|---|
| `hybrid_lexical` | **+0.0512** | what the learned sparse leg buys over Postgres FTS |
| `splade_only` | **+0.0303** | what the **dense** leg still adds on top of SPLADE |
| `dense_only` | **+0.0642** | what the sparse leg adds to dense alone |

⚠️ **Name the comparison before quoting the delta.** An earlier draft of this file carried
"+0.0303" as *the SPLADE lever*. On this table +0.0303 is `splade_only → hybrid_splade`, which is the
**dense** contribution; the SPLADE lever against the lexical leg is **+0.0512**. Same number, wrong
arrow. The independently-run paired study that shipped SPLADE
([[project-recall-splade-learned-sparse-measured-2026-08-06]], PR #222) reports its own significance
on its own query population, and that population is not this one, so its p-value is not restated
here.

The one paired test run on exactly these 777 queries:

- **Voyage rerank over `hybrid_splade`**: nDCG@5 **+0.0769**, bootstrap 95% CI
  **[+0.0575, +0.0968]**, permutation **p = 0.00010**.

  Re-derived deterministically from the two arms' committed per-query scores (fixed seed, 10,000
  resamples) so the artifact holds the same interval on every rebuild. The original run reported
  [+0.0571, +0.0964]; the two agree to within bootstrap noise, and the point estimate, the
  permutation p and the 302/162/313 split are identical.

⚠️ **The rerank helps on average and hurts 162 queries.** Better on 302, worse on **162**, unchanged
on 313. An average lift is not a promise per query, which is exactly why the reranker is opt-in and
off by default.

⚠️ **SPLADE is not free in latency**, only in money. It runs a transformer encode per query, and on
CPU that dominates the query. The p50 figures from this run were taken on a heavily loaded shared
host and are not representative enough to publish; treat SPLADE as a GPU-or-patience option and
measure it on your own hardware.

---

## 3. Task B — validating our own harness before trusting anything else

Task B supplies gold contexts, so **RE-call is not in this number at all**. It measures only whether
our generation and scoring pipeline behaves like theirs. If it does not, the Task C number is
meaningless.

| system | RL_F | RB_llm | RB_alg | harmonic |
|---|---|---|---|---|
| *target (human reference, not a system)* | 0.8690 | 0.9519 | 0.8772 | *0.8978* |
| llama-3.1-405b-instruct | 0.7543 | 0.7416 | 0.4751 | 0.6277 |
| gpt-4o | 0.7576 | 0.7616 | 0.4547 | 0.6208 |
| **ours, `gpt-4o`, official prompt** | 0.7793 | 0.7285 | 0.4573 | **0.6195** |
| qwen-2.5-72b-instruct | 0.7239 | 0.7473 | 0.4435 | 0.6031 |
| c4ai-command-r-plus | 0.7595 | 0.6938 | 0.4435 | 0.5985 |
| gpt-4o-mini | 0.7156 | 0.7499 | 0.4331 | 0.5953 |
| ours, `gpt-4o`, our own prompt | 0.7524 | 0.6315 | 0.4628 | 0.5913 |

**0.6195 against their 0.6208 on identical inputs is a gap of 0.0013.** Same model, same contexts,
same tasks, same judge. The harness is sound, so what Task C measures is retrieval. Our two rows
place **3rd** and **6th** of ten.

⚠️ Every cell above is the **conditioned** metric, ours and theirs alike. An earlier draft put our
abstain row in at **0.5508**, which is its *raw* harmonic, in a column where every other number was
conditioned. Same error as §6, inside the table meant to demonstrate it.

⚠️ The `abstain` row also **mixes denominators**: twelve RAGAS `TimeoutError`s leave `RL_F`
averaging **832** rows against 842 for the other two. Complete-case is **0.5902**. Every other run
here is 842 on all three metrics.

---

## 4. Task C — how RE-call actually compares

The only row where RE-call is the variable is the fixed-prompt pair: same harness, same prompt, same
generator, **only the contexts differ**.

| contexts | prompt | harmonic |
|---|---|---|
| **RE-call** | official | **0.5527** |
| benchmark's own (ELSER) | official | 0.5516 |
| RE-call | our own | 0.5327 |
| benchmark's own (ELSER) | our own | 0.5228 |

**RE-call's contexts beat the benchmark's own retrieval by +0.0011** with the official prompt, and
**+0.0099** with the other one. Small, but the sign is **consistent across both prompts**, which it
was not before a scoring bug was fixed (§6).

Against the published baselines, recomputed:

| system | harmonic |
|---|---|
| llama-3.1-405b-instruct | 0.5691 |
| qwen-2.5-72b-instruct | 0.5625 |
| gpt-4o | 0.5591 |
| **ours, RE-call contexts** | **0.5527** |
| c4ai-command-r-plus | 0.5502 |
| gpt-4o-mini | 0.5437 |

⛔ **We do not beat the baselines.** 0.5527 against their `gpt-4o` at 0.5591 is **−0.0064**. An
earlier version of this analysis read as "RE-call beats every baseline including llama-3.1-405b";
that was an artifact and it was one edit from publication. See §6.

---

## 5. Abstention, judged by someone else

The claim this project has made for a year is that abstention is calibrated rather than guessed.
Here is what an independent judge says.

**16 of 55 UNANSWERABLE tasks correctly abstained: 29.1%.** Recomputed per system from `RAG.json`,
scoring an exact 1.0 on `rb_agg` for an UNANSWERABLE task as a correct refusal:

| system | correct refusals | end-to-end harmonic |
|---|---|---|
| *target (human reference, not a system)* | *48/55 · 87.3%* | *0.7947* |
| llama-3.1-8b-instruct | 18/55 · 32.7% | 0.4710 |
| **ours** | **16/55 · 29.1%** | **0.5527** |
| llama-3.1-70b-instruct | 16/55 · 29.1% | 0.5378 |
| gpt-4o-mini | 13/55 · 23.6% | 0.5437 |
| c4ai-command-r-plus | 11/55 · 20.0% | 0.5502 |
| gpt-4o | 7/55 · 12.7% | 0.5591 |
| llama-3.1-405b-instruct | 3/55 · 5.5% | 0.5691 |
| qwen-2.5-7b-instruct | 3/55 · 5.5% | 0.5447 |
| qwen-2.5-72b-instruct | 1/55 · 1.8% | 0.5625 |
| mixtral_8x22b_instruct | 0/55 · 0.0% | 0.5230 |

**Second of ten, tied with `llama-3.1-70b`, and the correlation runs the wrong way for the leaderboard.** The two systems that
beat us end to end refuse **12.7%** and **5.5%** of what they cannot answer. The system that refuses
best, `llama-3.1-8b`, is last on answer score. A single harmonic mean cannot express that trade, and
this table is the reason we publish both columns rather than the one that flatters us.

⚠️ Do not read this as a retrieval result. See the second bullet below: the abstention rate is set
by the generator prompt.

Two things this measurement corrected in our own understanding:

- ⛔ **A regex cannot measure an abstention rate.** String-matching the refusal produced 43.6%,
  inflated, and it misled this analysis **three separate times**. Every abstention figure here comes
  from the official judge instead.
- **The prompt, not the retriever, sets the abstention rate.** Switching the generator prompt moved
  correct abstentions from 89% to 64% and false abstentions from 12.6% to 8.6%. The trade was
  strongly positive overall, but it is a *generator* property, and no retrieval change moved it
  comparably.

---

## 6. What we got wrong

**Every number in the first version of this analysis compared our RAW metrics against the
baselines' IDK-CONDITIONED ones.**

The official scorer reads the answerability label in lower case, `row.get("answerability", [])`,
while the release files ship `Answerability`. The key was never found, conditioning silently never
ran, and the fall-through branch **penalised** a correct abstention instead of rewarding it. The
scorer printed `Error: answerability is None` 2,526 times and completed successfully.

The failure was visible three times and explained away twice. The 2,526 error lines were grepped,
counted exactly, and dismissed as the word "error" inside judge explanations. Conditioned values
came out *below* raw ones, which is impossible if conditioning rewards correct abstention, and that
was noted as a curiosity. The third signal, an implausible **+0.0249 in our favour**, was the only
one chased, because it was the one that flattered us.

🔑 **An anomaly you can explain is still an anomaly, and the direction of the flattery should not
decide which ones get investigated.**
🔑 **Verify the comparison, not just the result.** Both sides were individually correct. The error
lived entirely in the join.

Full account, including the proof that the published baselines are conditioned:
[`results/mtrag_generation/CORRECTION-idk-conditioning-2026-08-09.md`](../results/mtrag_generation/CORRECTION-idk-conditioning-2026-08-09.md).

**Reported upstream** as [IBM/mt-rag-benchmark#23](https://github.com/IBM/mt-rag-benchmark/issues/23),
verified live at HEAD `cc5b1d4` with no prior report in the tracker.

---

## 7. The finding that matters most, and it is not about retrieval

Switching the generator prompt from ours to the paper's own (arXiv 2501.03468, Appendix D.2) moved
Task B from **0.5913 to 0.6195**, a lift of **+0.0282**, which is **6th to 3rd of ten** in the
recomputed table. Cause, measured: our prompt produced **83 false abstentions on 709 ANSWERABLE
tasks**, and a false abstention scores near zero (RL_F 0.0726 against 0.8901 when it answered).

⚠️ **An earlier draft of this section said +0.0687, from 0.5508.** That subtracted the **raw**
Task B abstain figure from the **conditioned** official one, which is the very raw-vs-conditioned
mix-up documented in §6, committed again inside the document explaining it. The lift on a single
consistent definition is +0.0282. Caught by generating `mtrag_summary.json` from the artifacts
instead of subtracting two numbers by hand.
🔑 **Knowing the failure mode does not stop you repeating it; computing the number does.**

Compare that against RE-call's whole retrieval stack **on the same metric**: our contexts against
the benchmark's own move the end-to-end harmonic **+0.0011**. The prompt was worth roughly
**twenty-five times** the retriever. (The retrieval-side levers are larger in their own units,
SPLADE +0.0512 R@100 and Voyage rerank +0.0769 nDCG@5, but those are different metrics on a
different task and do not belong in the same subtraction.)

🔑 **On this benchmark the retriever was not the cap.** A single prompt line outweighed the entire
retrieval stack. We publish that because the alternative is letting a reader assume a better
retriever is always the next thing to buy. **Measure where your own cap is before paying to move
it**, which is the same conclusion this project reached independently on corpus size and on
reranking.

⚠️ One honest caveat on transfer: the decision to change the prompt was *motivated* by a
label-stratified analysis. The prompt adopted is the paper's own and justified independently, but no
gain that depended on label access should be claimed to transfer to a sealed evaluation set.

---

## 8. Compliance

MTRAGEval withholds question type, answerability and multi-turn type at evaluation time; only the
corpus domain is provided. Audited across all 842 tasks:

| path | status |
|---|---|
| prompt sent to the model | leak-free, verified on all 842 |
| submission rows | `task_id`, `Collection`, `input`, `contexts`, `predictions` only |
| the 65-task retrieval backfill | selected by **set difference on task ids**, never on labels |
| domain routing | uses `Collection`, which is provided |

The withheld labels *are* used post-hoc for stratification and diagnosis, which is what they are for
once a run is over.

---

## 9. Reproduce

Every artifact from all six generation runs is committed, 68 MB gzipped from ~430 MB, each file
sha256-verified: [`results/mtrag_generation/runs/`](../results/mtrag_generation/runs/README.md).
That includes the judge logs carrying the 2,526-error evidence.

```bash
# the headline number, straight from the committed artifact
cd results/mtrag_generation/runs
python -c "
import gzip,json,statistics
rows=[json.loads(l) for l in gzip.open('taskb_official.fixed.jsonl.gz','rt',encoding='utf-8') if l.strip()]
f=lambda v: v[0] if isinstance(v,list) else v
m=[statistics.fmean([f(r['metrics'][k+'_idk_underspecified']) for r in rows]) for k in ('RL_F','RB_llm','RB_agg')]
print(len(rows), [round(x,4) for x in m], 'harmonic', round(len(m)/sum(1/x for x in m),4))
"
```

Expect `842 [0.7793, 0.7285, 0.4573] harmonic 0.6195`.

Generator `openai/gpt-4o` via OpenRouter on all six runs, zero task failures in all six. Judge
`gpt-4o-mini-2024-07-18` on the `openai` path via an OpenAI-compatible endpoint; the Azure and local
vLLM judge paths were not exercised.

MTRAG is used under Apache-2.0; attribution and citation in
[`results/mtrag_generation/runs/README.md`](../results/mtrag_generation/runs/README.md).
