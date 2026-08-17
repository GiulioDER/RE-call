# Pre registration: cheap OpenRouter retrieval reasoning

**Date:** 2026-08-17   **Status:** predicted, not yet measured

## The question

Does cheap OpenRouter retrieval reasoning improve expected document recall on the official 20 question EnterpriseRAG calibration fixture without reducing repeated capture stability?

## What I predict

The closed loop arm will improve document recall by at least 0.03 over the library baseline, with the largest gain on questions whose initial retrieval has a gap warning or misses the expected document. The cheap arm will produce useful expansions on at least 25 percent of questions. Repeated capture document Jaccard will remain at or above 0.90.

## What would falsify this

No absolute document recall improvement of 0.03, no useful expansion on at least 25 percent of questions, or repeated capture mean document Jaccard below 0.90 falsifies the prediction. Any provider validation failure also falsifies the safety part of the prediction.

## How it will be measured

The fixed fixture is `.benchdata/enterprise-rag-v1.0.0/calibration_20_questions.jsonl` with its paired `calibration_20_docs.zip`, n equals 20. I will run the same indexed corpus and retrieval settings for four arms: `none`, `depth`, `cheap`, and `closed_loop`. Each arm will use three retrieval captures per question, the same local fastembed profile, k equals 8, candidate k equals 80, and extractive answers. The independent judge will not run. The primary metric is `retrieval.document_recall` over expected document ids. Secondary metrics are reasoning expansion rate, added query count, fallback rate, validation failures, and repeated capture mean document Jaccard.

The cheap model is `openai/gpt-5-nano` through OpenRouter. The OpenRouter model page lists it at 0.05 dollars per million input tokens and 0.40 dollars per million output tokens, and describes it as the smallest and fastest GPT 5 variant for cost sensitive interactions.

## What I already know

The reasoning provider is opt in and separate from the answer provider. Expansion outputs are untrusted until retrieval binding and trust validation pass. The current implementation is in `recall/reasoning_expansion.py`, `recall/reasoning.py`, and `benchmarks/enterprise_rag.py`.

## Confounds I can name now

The calibration fixture is small. Local fastembed and Postgres execution are deterministic enough for this capture, but reranker and learned sparse arms are excluded. The cheap model can change output if the provider routes differently, so the model id, revision, prompt digest, cache, and usage metadata will be retained. This run measures retrieval value only, not answer correctness.

## Retrieval-only result

Measured on 2026-08-17 with three captures per question and no answer judge:

| Arm | Document recall | Exact document coverage | Capture stability |
| --- | ---: | ---: | ---: |
| `none` | 0.9310 | 0.8750 | 1.00 |
| `depth` | 1.0000 | 1.0000 | 1.00 |
| `cheap` | 1.0000 | 1.0000 | 0.80 |
| `closed_loop` | 1.0000 | 1.0000 | 0.95 |

Depth expansion recovered the two missed expected documents from the baseline. The cheap and closed loop arms reached the same retrieval ceiling, so no additional retrieval gain over depth was observed on this fixture. Among generated queries for questions with expected documents, useful query precision was 39 of 39 for the cheap cache and 20 of 20 for the closed loop cache. The cheap arm produced 33 retrieval queries and the closed loop arm produced 40. The provider was OpenRouter `openai/gpt-5-nano`, revision `openrouter-2026-08-17`, with minimal reasoning effort.

The result is a retrieval signal only. It does not establish answer correctness, abstention safety, citation validity, or promotion eligibility. The expensive model remains disabled and the promotion gate remains pending. The complete machine-readable record is `results/real-reasoning-20260817/retrieval-value-summary.json`.

The first cheap-model attempt was discarded as an apparatus failure because the activation variable was wrong and a direct probe showed the model consuming the output budget in hidden reasoning before returning JSON. The measured reruns used the corrected activation variable and `reasoning_effort=minimal`.
