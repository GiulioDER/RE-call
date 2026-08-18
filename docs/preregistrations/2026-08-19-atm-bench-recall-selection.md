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
