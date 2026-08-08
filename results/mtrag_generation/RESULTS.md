# MTRAG Tasks B and C — results as of 2026-08-08

**Prior work** (searched this session, per CLAUDE.md; recorded rather than repeated):
`docs_search(source_type='memory', "MTRAG Task B Task C generation RAG answer quality GPT-4o judge
RL_F RB_llm RB_alg harmonic mean")` and
`docs_search(source_type='memory', "over-abstention false refusal prompt instruction says I don't
know on answerable questions cost")`. Binding results:
[[project-recall-mtrag-rbalg-probe-2026-08-05]] (the anchored-lift constraint, and a "models are
verbose" thesis already FALSIFIED — which is why no RB_alg gain is claimed from the 150-word cap),
[[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]], and
[[project-recall-abstention-bounded-domain-2026-07-24]] (⚠️ that one is RE-call's RETRIEVAL-side
abstention layer, a different component from the generator prompt studied here).

Six generation runs, 842 tasks each, `openai/gpt-4o` via OpenRouter, zero failures in all six.
A 2x2 by design: **{gold, benchmark-retrieved, RE-call-retrieved} contexts × {abstain, official}
prompt**, so the prompt effect and the retrieval effect can be separated instead of confounded.

⚠️ **Read the comparison caveat first.** Recomputing the published table from the release runs
+0.018 to +0.043 HIGH on every model ([[project-recall-mtrag-rbalg-probe-2026-08-05]]): the
aggregation formula is right, the instance set is not the published one. Every baseline below is
therefore **recomputed here with the same code that computes our row**, and the comparison is an
ANCHORED LIFT. None of these numbers may be quoted against the published leaderboard directly.

## Scored so far

| run | contexts | prompt | RL_F | RB_llm | RB_alg | **harmonic** |
|---|---|---|---|---|---|---|
| Task B | gold | `abstain` | 0.7011 | 0.6283 | 0.4117 | **0.5508** |
| Task C | benchmark | `abstain` | 0.7369 | 0.6024 | 0.3999 | **0.5437** |

Task B's RL_F is over 830 rows, not 842: 12 RAGAS `TimeoutError`s. Complete-case (n=830) gives
0.5502 against 0.5508 for all rows, a difference of 0.0006, so they are missing at random and are
not moving the number.

### Recomputed baselines — Task B (gold contexts)

| model | RL_F | RB_llm | RB_alg | harmonic |
|---|---|---|---|---|
| *target (human reference, not a system)* | 0.8690 | 0.9519 | 0.8772 | *0.8978* |
| llama-3.1-405b-instruct | 0.7543 | 0.7416 | 0.4751 | 0.6277 |
| gpt-4o | 0.7576 | 0.7616 | 0.4547 | 0.6208 |
| qwen-2.5-72b-instruct | 0.7239 | 0.7473 | 0.4435 | 0.6031 |
| c4ai-command-r-plus | 0.7595 | 0.6938 | 0.4435 | 0.5985 |
| gpt-4o-mini | 0.7156 | 0.7499 | 0.4331 | 0.5953 |
| qwen-2.5-7b-instruct | 0.6787 | 0.7189 | 0.4358 | 0.5815 |
| llama-3.1-70b-instruct | 0.6962 | 0.6632 | 0.4422 | 0.5763 |
| mixtral_8x22b_instruct | 0.6174 | 0.6978 | 0.4202 | 0.5522 |
| **ours (`abstain`)** | 0.7011 | 0.6283 | 0.4117 | **0.5508** |
| llama-3.1-8b-instruct | 0.5539 | 0.5936 | 0.3706 | 0.4848 |

### Recomputed baselines — Task C (retrieved contexts)

| model | RL_F | RB_llm | RB_alg | harmonic |
|---|---|---|---|---|
| *target (human reference, not a system)* | 0.6478 | 0.9467 | 0.8513 | *0.7947* |
| llama-3.1-405b-instruct | 0.7156 | 0.6826 | 0.4151 | 0.5691 |
| qwen-2.5-72b-instruct | 0.7119 | 0.6995 | 0.4002 | 0.5625 |
| gpt-4o | 0.7076 | 0.7006 | 0.3960 | 0.5591 |
| c4ai-command-r-plus | 0.7246 | 0.6370 | 0.3996 | 0.5502 |
| qwen-2.5-7b-instruct | 0.6715 | 0.6805 | 0.3924 | 0.5447 |
| **ours, benchmark ctx (`abstain`)** | 0.7369 | 0.6024 | 0.3999 | **0.5437** |
| gpt-4o-mini | 0.6801 | 0.6869 | 0.3859 | 0.5437 |
| llama-3.1-70b-instruct | 0.6764 | 0.6304 | 0.3978 | 0.5378 |
| mixtral_8x22b_instruct | 0.6157 | 0.6560 | 0.3865 | 0.5230 |
| llama-3.1-8b-instruct | 0.5587 | 0.5873 | 0.3475 | 0.4710 |

## 🔑 The finding so far: my prompt, not RE-call

Task B holds retrieval perfect, so **RE-call is not in that number at all**. Ours vs the
benchmark's own `gpt-4o` — same task, same gold contexts, same model — is therefore a clean read on
our harness:

| task | ours | their gpt-4o | gap |
|---|---|---|---|
| B (gold) | 0.5508 | 0.6208 | **−0.070** |
| C (benchmark ctx) | 0.5437 | 0.5591 | **−0.015** |

Cause, measured: `abstain` produced **83 false abstentions on 709 ANSWERABLE tasks**, and a false
abstention scores near zero (RL_F 0.0726 against 0.8901 when it answered).

🔑 **The handicap costs 4.6x more on Task B than on Task C**, from the same prompt. With gold
contexts the answer is nearly always present, so every false abstention is pure loss; with
retrieved contexts the passages genuinely often fall short, so abstaining is right more often.

### The prompt was published, and I said it was not

I asserted "MTRAG publishes no generation script, so the baselines' prompt is unknown" after one
grep. It is in the paper, arXiv 2501.03468 **Appendix D.2 "Model invocation"**. `abstain` differs
from it in three ways, not the one I first diagnosed:

1. no **150-word limit**;
2. `"say that you do not know rather than guessing"` instead of the exact string
   `"I do not have specific information"`;
3. different ordering and passage framing.

⛔ Also NOT the prompt in `conversations/conversations.json` — that one built the conversation
dataset with mixtral-8x7b and has no length limit. Taking the first prompt found in the repo would
have been the same error one level quieter.

### Pre-registration: the mechanism prediction FAILED

Registered in `PREREGISTRATION-official-prompt-2026-08-08.md` before any scored output existed.
Predicted false abstentions would fall from 83 to **under 40**. Measured, with a single consistent
detector across both files:

| | false-abstain on ANSWERABLE | UNANSWERABLE abstain | mean words |
|---|---|---|---|
| `abstain` | 89 / 709 (12.6%) | 49 / 55 (89.1%) | 51.6 |
| `official` | **61 / 709 (8.6%)** | **35 / 55 (63.6%)** | **79.9** |

Direction right, magnitude wrong, and an effect I did not predict cuts the other way: **correct
abstentions collapsed too**, 89.1% → 63.6%. The official prompt abstains less *everywhere*, and a
correct IDK scores exactly 1.0 on all three metrics, so losing 14 of them is expensive. Answers also
got longer, because `abstain` said "concisely" and D.2 does not.

⚠️ The detector here also matches "I do not have specific information", so it reads the old file as
89 where the pre-registered figure said 83. The old-vs-new comparison is internally consistent; the
registered threshold was anchored to a narrower detector. That is a measurement-definition slip on
my side and the third time today that string-matching abstention has misled.

## MTRAGEval compliance

MTRAGEval withholds question type, answerability and multi-turn type at evaluation; only the
corpus domain is provided. Audited:

| path | status |
|---|---|
| prompt sent to the model | leak-free, verified across all 842 |
| submission rows | `task_id`, `Collection`, `input`, `contexts`, `predictions` only |
| the 65-task retrieval fix | selected by **set difference on task ids**, not labels |
| domain routing | uses `Collection`, which is provided |

The withheld labels ARE used post-hoc for stratification and diagnosis, which is what they are for
once a run is over. ⚠️ One honest caveat: the decision to change the prompt was *motivated* by a
label-stratified analysis. The prompt adopted is the paper's own, justified independently, and the
`neutral` variant that was derived from label stratification is not used — but no dev-set gain that
depended on label access should be claimed to transfer to the sealed set.

## Pending

Awaiting one GPU session (three algorithmic passes, ~8 min each) then the LLM judge on each:
`taskb_official` (judge running), `taskc_recall`, `taskc_benchmark_official`,
`taskc_recall_official`.

🔑 **The comparison that actually measures RE-call** is the fixed-prompt pair
`taskc_recall_official` vs `taskc_benchmark_official`: same harness, same prompt, same model, only
the contexts differ. It needs no baseline table to be meaningful, and it is the only row in this
document where RE-call is the variable.
