# ATM Bench RE call retrieval preregistration

Date: 2026-08-19

## Purpose

I will test whether RE call improves evidence retrieval on the official ATM Bench benchmark while keeping retrieval quality separate from answer length and answer completeness. The first measurement is retrieval only. It does not call an answer model and it does not call the ATM Bench judge.

## Frozen data

I will use the official `ATM-Bench-Hard` question file with all 31 questions. The memory corpus will contain the official text artifacts available locally:

1. `atm-bench-hard.json`, SHA256 `ACD35F2A172A9741D970D2CF21184FF0AF8D79A8BF59967FC8AA33D619F6AF4A`
2. Processed image descriptions from `image_batch_results.json`, SHA256 `7204E8B4AB1A0FEA97F2003058213742D47589E09636941F8D83B2E77C33E0A1`
3. Processed video descriptions from `video_batch_results.json`, SHA256 `88ED14CE32AFDE1A4D54D5B77A0E7F84305B543186307BE6C9BC34119342727A`
4. Email memory from `emails.json`, SHA256 `5C82C38E7F18923A9EBA3AF321663A8A5BD70E94E0D5E59E987141C71219A8AF`

Raw media will not be used in this retrieval measurement because the official text baseline also supports processed text artifacts, and the raw release is not present locally. Every memory item is indexed once, with its official evidence ID preserved as the RE call chunk ID. Duplicate IDs will fail the run rather than being silently merged.

## Frozen representation

The text representation follows the official MMRag `SGM` configuration:

1. Image and video items include ID, type, timestamp, location, short caption, caption, OCR, and tags.
2. Email items include ID, timestamp, short summary, and detail.
3. No question, answer, or gold evidence ID is added to a memory item.
4. No raw media is added to the embedding text.

## Frozen retrieval arms

I will run these arms over the same index and query set:

1. RE call dense retrieval, using `HybridRetriever` with `use_dense=True`, `use_sparse=False`.
2. RE call hybrid retrieval, using `HybridRetriever` with dense retrieval and lexical full text retrieval.

Both arms use `st:sentence-transformers/all-MiniLM-L6-v2`, candidate pool size 200, no reranker, and the same PostgreSQL and pgvector store. The official baseline comparison will be labeled as an engineering comparison unless the exact official runner is also executed, because implementation details outside the frozen representation can affect rank order.

## Frozen metrics

For each question I will record the retrieved evidence IDs and calculate:

1. Per item evidence recall at `k`, the fraction of gold evidence IDs present in the first `k` retrieved IDs.
2. Question level `Recall@k`, the official ATM Bench metric, where at least one gold evidence ID is present in the first `k` retrieved IDs.
3. Complete evidence recall at `k`, also called `Recall@kGT`, where every gold evidence ID is present in the first `k` retrieved IDs.
4. Mean and p95 retrieval latency.
5. Corpus size, index time, embedding profile, candidate pool size, and run environment.

The frozen `k` values are `1`, `5`, `10`, `25`, `50`, and `100`. The primary comparison is question level `Recall@10` and complete evidence recall at `10`. The secondary comparisons are all other frozen `k` values and latency.

## RE call selection prediction

The first measurement does not apply answer slot selection because ATM Bench supplies evidence IDs, not serving time answer slots. Applying gold evidence IDs as slots would leak the target and would not test RE call. I predict that retrieval remains the main loss before answer selection on the multi evidence questions, and that any later selector experiment must report candidate pool coverage separately from selector retention.

## Exclusions

This preregistration excludes answer generation, ATM Bench answer quality, the `gpt-5-mini` judge, answer token counts, and leaderboard submission. A later answer quality run will use the official prompt and judge configuration, with any fallback judge explicitly reported.

## Reproduction command

After the runner is implemented, the exact command will be recorded in the appended results section and in the result JSON manifest. The runner will be `python -m benchmarks.atm_bench` from this repository root.

## Official references

1. ATM Bench project page: https://atmbench.github.io/index.html
2. ATM Bench repository: https://github.com/JingbiaoMei/ATM-Bench
3. ATM Bench metrics: https://github.com/JingbiaoMei/ATM-Bench/blob/main/docs/metrics.md
4. ATM Bench reproducibility: https://github.com/JingbiaoMei/ATM-Bench/blob/main/docs/reproducibility.md
5. ATM Bench judge configuration: https://github.com/JingbiaoMei/ATM-Bench/blob/main/memqa/utils/evaluator/config.py

## Protocol clarification recorded before measurement

The official metrics document describes `Recall@k` as a question level hit metric. The official MMRag implementation currently writes `retrieval_recall` as the fraction of gold evidence items found at each `k`, and the official comprehensive evaluator also computes the mean item level recall. I will preserve both interpretations in the result: the official implementation compatible item level recall, question level hit rate, and complete evidence recall. I will not silently rename one into another.

## Results appended after measurement

Measured on 2026-08-19. The exact command was:

```text
python -m benchmarks.atm_bench --qa-file C:\Users\gde00\Documents\atm-bench-official\data\atm-bench\atm-bench-hard.json --image-file C:\Users\gde00\Documents\atm-bench-official\output\image\qwen3vl2b\batch_results.json --video-file C:\Users\gde00\Documents\atm-bench-official\output\video\qwen3vl2b\batch_results.json --email-file C:\Users\gde00\Documents\atm-bench-official\data\raw_memory\email\emails.json --out results/atm_bench_hard_retrieval_20260819_fastembed.json --embedder fastembed:sentence-transformers/all-MiniLM-L6-v2 --arms dense hybrid
```

The run used 11,034 memory items and 31 ATM Bench Hard questions. It used the local FastEmbed ONNX conversion of the official `sentence-transformers/all-MiniLM-L6-v2` backbone because the native sentence transformers wrapper could not import on this Windows runtime. No answer model, judge, or API credit was used. This is a local engineering result, not an official leaderboard claim for the native MMRag implementation.

The saved artifact is `results/atm_bench_hard_retrieval_20260819_fastembed.json`. The official comprehensive evaluator was also run on both saved retrieval detail files.

| Arm | Official item R@10 | Question Recall@10 | Complete Recall@10GT | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.2904 | 0.6452 | 0.0323 | 243.5 ms | 519.3 ms |
| Hybrid | 0.3352 | 0.7097 | 0.0645 | 300.8 ms | 432.4 ms |

The result supports hybrid retrieval for this corpus, with a gain of 6.45 percentage points in question Recall@10 and 3.23 percentage points in complete evidence Recall@10GT. Complete evidence recall remains low, confirming that answer selection must be evaluated as candidate retention after retrieval, not as a substitute for retrieving missing evidence.

## Preregistration for deterministic list answer retention

I will now measure the ATM list answer path without an LLM. I will select the first `k` retrieved evidence IDs as the predicted answer, preserving retrieval order. I will not filter those IDs using the gold answer or gold evidence IDs.

The scored subset is the 12 `list_recall` questions in ATM Bench Hard. The gold list comes from each row's official `answer` field, split on commas and newlines, with surrounding whitespace and terminal punctuation removed. The `evidence_ids` field is retained only for the separate retrieval metrics because it can contain more items than the answer list.

The primary metrics are mean Jaccard answer score and mean gold answer containment at `k=5` and `k=10`. Secondary metrics are question level answer hit rate, the fraction of questions with at least one gold answer ID in the selected output, and scores at `k=1`, `25`, `50`, and `100`. The dense and hybrid retrieval outputs from the committed result artifact are the only arms. No answer model, judge, or gold conditioned filtering is allowed.

I predict that hybrid will improve mean Jaccard and gold answer containment relative to dense retrieval, but that top five output will remain incomplete on the long list questions. This test measures the retrieval to final ID list path directly; it does not claim to test AnswerSlot policy selection, which ATM does not annotate.

## List answer retention results appended after measurement

Measured on 2026-08-19 with `python -m benchmarks.atm_list_selection_probe --ground-truth C:\Users\gde00\Documents\atm-bench-official\data\atm-bench\atm-bench-hard.json --retrieval-result results/atm_bench_hard_retrieval_20260819_fastembed.json --out results/atm_bench_hard_list_selection_20260819.json`.

| Arm | Jaccard at 5 | Containment at 5 | Jaccard at 10 | Containment at 10 | Containment at 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.1661 | 0.2013 | 0.1527 | 0.2669 | 0.5895 |
| Hybrid | 0.2472 | 0.2913 | 0.2050 | 0.3381 | 0.6592 |

Hybrid improves both top five Jaccard and containment over dense retrieval. Increasing the output from five to ten improves containment from 0.2913 to 0.3381 but reduces Jaccard from 0.2472 to 0.2050. At 100 items, containment reaches 0.6592 while Jaccard falls to 0.0481. This is direct evidence that answer selection needs a precision aware stopping or selection rule. Retrieval depth alone is not a valid final answer policy.

The Jaccard values were independently checked against the official ATM Bench `list_jaccard_score` implementation and matched at all reported cutoffs.

## Preregistration for the full ATM Bench split

Before measuring the larger split, I will repeat the same retrieval and deterministic list answer retention protocol on the official `atm-bench.json` file with all 1,013 questions, including its 139 `list_recall` questions. The memory corpus, text representation, embedding backbone, dense arm, hybrid arm, candidate pool, and cutoffs remain unchanged. The only changed input is the official question split.

The primary outcomes are question level `Recall@10`, complete evidence `Recall@10GT`, list answer Jaccard at five, and list answer containment at five. Secondary outcomes are the complete `k` curves, latency, and qtype strata. The output selection remains exactly the first `k` retrieved IDs. I predict that hybrid will remain above dense on retrieval and list Jaccard, while containment will continue to rise with `k` and Jaccard will peak at a smaller output than containment.

## Full split score correction appended after measurement

The first full split list artifact used a narrower answer parser than the official scorer. The retrieval measurement is unaffected. I corrected the parser to match ATM Bench's official list splitting and normalization, reran the deterministic scoring only, and saved the corrected result as `results/atm_bench_full_list_selection_20260819_corrected.json`. The earlier full list artifact remains preserved as evidence of the correction and must not be used for claims.

Corrected full split results:

| Arm | Jaccard at 1 | Jaccard at 5 | Containment at 5 | Jaccard at 10 | Containment at 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.2068 | 0.1161 | 0.4222 | 0.0755 | 0.7601 |
| Hybrid | 0.2716 | 0.1550 | 0.5263 | 0.1060 | 0.8582 |

The corrected values match the official ATM Bench scorer. Hybrid improves retrieval and list containment, but the best Jaccard cutoff is one item, while containment continues increasing through 100 items. This confirms that a final answer selector must optimize precision and recall jointly.

The cutoff difference is explained by answer cardinality. The 12 hard split list answers average 7.58 IDs, while the 139 full split list answers average 1.69 IDs. I will not promote one fixed cutoff from these results. The next selection experiment must test a serving time cardinality policy derived from the question, with no access to the gold answer.
