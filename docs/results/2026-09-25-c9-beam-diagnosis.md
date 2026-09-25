# C9 on BEAM: why the AML Textual smoke scored 0 on B2 and F1, and what did not fix it

**Dates:** 2026-09-24 to 2026-09-25. **Service:** the official C9 on VPS2 (port 18015, served
`385c6074` for ingest and retrieval). **Harness:** `benchmarks/beam/aml_c9_probe.py` and
`benchmarks/beam/aml_c9_prep.py`.

This page is the summary. The four pre-registrations below each hold their predictions, committed
before measuring, and the full results appended under them. The commits that timestamp each
prediction are on branch `claude/c9-beam-probe`.

## Why this was run

The official AML Textual smoke scored 0 on **B2** (causal chain and intermediate-step recovery),
0 on **F1** (summarization and long-history synthesis) and 20 on **C1** (dates, relative time and
intervals), on roughly 1 to 3 questions each. AML forbids analysing its evaluation data, so the
diagnosis used the public **BEAM 100K** split instead. It has 20 conversations and 400 questions,
40 per ability type. Summarization maps to F1; multi-session reasoning and event ordering to B2;
temporal reasoning to C1.

## Method

- **Ingest and retrieval:** each BEAM session is one Add to the official C9, and each question is
  one Search with `top_k` 100.
- **Answering and judging:** AML's public BEAM pipeline (`AML-memory/agent-memory-leaderboard` at
  `1b8142b`): its answer prompt, its batch rubric judge, and its event-ordering score (alignment F1
  times Kendall tau).
- **Model:** Qwen3-14B for both answering and judging, with thinking off.
- **Assumed:** the platform passes `data[].content` to the answer model in returned order.
- **Offline passes:** every pass after the first re-uses the stored retrieval, so none of them
  sent traffic to C9.

## Results

| pass | pre-registration | question | result |
| --- | --- | --- | --- |
| 1. ability probe | `docs/preregistrations/2026-09-24-c9-beam-ability-probe.md` | Where does C9 lose these types? | Retrieval returns the evidence; the answer model does not use it |
| 2. fewer items | `docs/preregistrations/2026-09-24-c9-beam-topk-reader.md` | Does top 10, 20 or 40 help? | **Null.** The 10-type mean is 0.456 for all four sizes |
| 3. quote-verified coverage | `docs/preregistrations/2026-09-25-c9-beam-quoted-coverage.md` | Is the evidence really there? | **Yes.** It is 71% to 81% present even when each point needs a verbatim quote |
| 4. session summaries | `docs/preregistrations/2026-09-25-c9-beam-session-summaries.md` | Do Add-time summaries help? | **No.** The 10-type mean is -0.016, and summarization -0.025. Not built |

Per type, with top 100 in returned order (pass 1), against the other passes:

| type (AML leaf) | answer score | lenient coverage | quote-verified coverage | time order | session summaries |
| --- | --- | --- | --- | --- | --- |
| summarization (F1) | 0.304 | 0.950 | 0.764 | -0.027 | -0.025 |
| event_ordering (B2) | 0.224 | 0.911 | 0.714 | **+0.101** | **+0.103** |
| multi_session_reasoning (B2) | 0.550 | 0.923 | 0.805 | -0.070 | -0.064 |
| temporal_reasoning (C1) | 0.289 | 0.529 | 0.618 | -0.132 | -0.105 |
| information_extraction (control) | 0.831 | 0.838 | 0.745 | -0.167 | -0.109 |
| contradiction_resolution | 0.062 | | | | -0.013 |

## What it means

1. **For F1 and B2, the loss is in the answer model, not in retrieval.** The 100 items C9 returns
   contain most of the evidence. The strict, quote-verified measure puts it at 71% to 81%, and the
   true figure lies between that and the lenient judge's 91% to 95%.
2. **Three changes on the memory side do not fix it.** Fewer items is a null. Time order and
   session summaries each help event ordering by about +0.10, but they hurt temporal reasoning and
   information extraction by about 0.1 each. So neither is a general change.
3. **AML's answer format caps several types, whatever memory returns.** Under "Be direct and
   concise ... Only output the answer", Qwen3-14B's median answer is one word for
   contradiction_resolution (34 of 40 are a bare "Yes." or "No."), knowledge_update and
   temporal_reasoning. Contradiction rubrics check about 4 points, so that type sits near 0.06
   regardless of retrieval. In summarization, answers under 60 words score 0.17 and answers of 60
   or more score 0.38.
4. **C1 (dates)** is the one retrieval-limited type here (lenient coverage 0.53). Its lever,
   timestamped windows, belongs to separate work (`docs/preregistrations/2026-09-24-aml-c9-reader-dates.md`),
   which measured +18 temporal points on LoCoMo.
5. **The one open idea is time order for ordering questions only.** It could give event ordering
   its +0.10 without the losses elsewhere, but it is a query-shape rule that could overfit BEAM's
   wording. It needs a second dataset before it is worth building.

## Side findings

- **Bare anchor ids, fixed.** During ingest, 11 of 88 anchored compiles kept 0 records, because
  gpt-4o-mini cited `a162` for `a162_<hash>`. PR 757 resolves a bare index of a sent anchor, and it
  is deployed as `33260782`. Replaying the same model answers kept 78 records instead of 23 on the
  failing batches.
- **Empty answers depend on the OpenRouter provider.** Some providers ignore
  `reasoning.enabled=false` for Qwen3, spend the 512-token answer budget thinking, and return an
  empty answer. 8% to 16% of answers per pass were empty, and the rate varied between passes.
  Re-analysed on the questions where both answers are non-empty, **no decision changed**. Future
  runs must pin a provider (`provider: {"order": ["alibaba"], "allow_fallbacks": false}`) and count
  empty answers per arm.
- **The first summary prompt was broken.** It put the instruction above a very long session, and
  gpt-4o-mini continued the dialogue in 71 of 90 outputs. It was caught before any answer was
  generated and fixed with a system message and fenced data. Always shape-check generated context
  before spending on answers.

## Reproduce

Data prep runs where pyarrow imports; the probe itself is stdlib-only:

```bash
python -m benchmarks.beam.aml_c9_prep --data 100K-00000-of-00001.parquet --out beam100k.jsonl
python benchmarks/beam/aml_c9_probe.py ingest   --data beam100k.jsonl --out out --base-url http://127.0.0.1:18015 --workers 6
python benchmarks/beam/aml_c9_probe.py retrieve --data beam100k.jsonl --out out --base-url http://127.0.0.1:18015 --workers 6
python benchmarks/beam/aml_c9_probe.py answer   --data beam100k.jsonl --out out --arm returned
python benchmarks/beam/aml_c9_probe.py judge    --data beam100k.jsonl --out out --arm returned
python benchmarks/beam/aml_c9_probe.py report   --data beam100k.jsonl --out out
python benchmarks/beam/aml_c9_probe.py cleanup  --data beam100k.jsonl --out out --base-url http://127.0.0.1:18015
```

The environment needs `RECALL_AML_API_KEY` and `OPENROUTER_API_KEY`; the other arms are listed in
`ARMS`. Raw outputs from these runs are on VPS2 at `/root/c9-beam-probe/out/` and
`/root/c9-beam-probe-out-20260924.tgz`. Model spend was about $8.00 across all passes, plus about
$2 of ingest.
