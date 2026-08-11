# Visibility Benchmarks

This is the submission track for getting RE-call onto public RAG leaderboards. It is intentionally
separate from the internal evidence record: internal benchmarks can support claims, but visibility
requires a result that a third party hosts, scores, or verifies.

## Priority

| rank | benchmark | why it is first | first deliverable |
|---|---|---|---|
| 1 | EnterpriseRAG-Bench | Best product fit: messy multi-source enterprise memory, absent information, citations, and cross-document reasoning. | Reproducible answer JSONL plus manifest submitted to the leaderboard maintainers. |
| 2 | Kaggle AgentEval Part I: Grounded RAG Benchmark | Kaggle has the largest broad ML audience, and the task measures grounded answers, citations, and hallucination control, but the active competition path was closed when checked. | Reopen only if there is a live submission path or a public notebook track worth using. |
| 3 | LiveRAG Challenge assets | Strong research credibility, especially for correctness plus faithfulness. | Post-challenge reproducible run if the organizers keep an evaluation path open. |
| 4 | CRAG | Highest-prestige RAG lineage because of Meta plus KDD Cup, but older and less product-shaped. | Reproducible CRAG adapter, only promoted if it can still land on a visible board. |

## Non-negotiables

1. A public result must name the exact RE-call version, git commit, model stack, corpus, prompt,
   retrieval budget, and scorer.
2. No internal recomputed baseline may be described as a leaderboard result.
3. If a submission uses hosted models or rerankers, the artifact must record provider, model id,
   revision if available, endpoint family, and cost-bearing calls.
4. If a benchmark requires a generated answer, the prompt is part of the measured system and must be
   versioned beside the retrieval configuration.
5. The public README only gets a new leaderboard claim after the external URL, screenshot, or
   maintainer verification is archived.

## Kaggle AgentEval

Goal: produce the first broad-audience public result.

Entry conditions:

1. Accept the Kaggle competition rules and download the data through a logged-in Kaggle account.
2. Record whether external internet access, local databases, package installs, and model calls are
   permitted in the notebook environment.
3. Freeze one honest primary configuration before looking at private scores.

Primary RE-call configuration:

```bash
python -m recall.cli setup
```

Use local embeddings first. If the Kaggle runtime cannot run PostgreSQL plus pgvector, use the
smallest compatible adapter that preserves RE-call's ranking, provenance, and abstention contract,
then state that portability layer in the artifact. Do not silently replace the system with plain
vector search.

Submission artifact:

| field | required value |
|---|---|
| external_url | Kaggle notebook, competition submission, or leaderboard row |
| recall_commit | `git rev-parse HEAD` |
| recall_version | installed `recall-rag` version |
| data_release | Kaggle dataset version or downloaded file digests |
| retrieval_config | embedder, reranker, `k`, candidate pool, sparse leg, trust mode |
| generation_config | model, endpoint, prompt digest, citation formatting |
| score | public and private leaderboard score when available |
| limitations | anything the Kaggle environment forced that differs from normal RE-call |

## EnterpriseRAG-Bench

Goal: land on a product-relevant leaderboard that rewards multi-source enterprise memory.

Entry conditions:

1. Download the released corpus and question set from the EnterpriseRAG-Bench repository or Hugging
   Face dataset.
2. Build a source-preserving importer. Source type should survive into metadata so citations can be
   audited by connector family.
3. Run one local primary configuration and one best-quality configuration only if cost and data
   movement are acceptable.

Implemented runner:

```bash
python -m benchmarks.enterprise_rag \
  --questions /path/to/questions.jsonl \
  --documents /path/to/all_documents.zip \
  --out results/enterprise_rag/recall.answers.jsonl \
  --reset-index --overwrite
```

The runner lives at `benchmarks/enterprise_rag.py`. It reads the official `questions.jsonl` plus
one or more local document release files or zip archives. The official release is a ZIP of `.txt`
files, so the importer derives stable `dsid_...` identifiers from filenames and preserves source
type metadata for audit.

The default `--answer-mode extractive` spends no LLM calls. Use it first to validate indexing,
retrieval, citation IDs, and output shape.

Current best launch preset:

```bash
python -m benchmarks.enterprise_rag \
  --questions /path/to/questions.jsonl \
  --documents /path/to/all_documents.zip \
  --out results/enterprise_rag/recall.top.answers.jsonl \
  --pool-size 4 \
  --reset-index --overwrite \
  --top-config
```

`--top-config` applies:

1. Voyage `voyage-4-large` embeddings,
2. Postgres lexical search plus SPLADE learned sparse search,
3. Voyage `rerank-2.5`,
4. `openai/gpt-4o` answer generation through OpenRouter,
5. `candidate_k=200`,
6. `k=8`,
7. `gap_threshold=0.5`,
8. `max_context_chars=12000`.

The SPLADE arm should run on a rented GPU for the full corpus. Local Windows CPU was validated, but
it took about 20 minutes to encode 1,227 chunks in the calibration smoke, which is not acceptable
for the full 500K document release. Use [ENTERPRISE_RAG_VAST.md](ENTERPRISE_RAG_VAST.md) for the
GPU runbook.

Calibration command:

```bash
python -m benchmarks.enterprise_rag \
  --questions /path/to/calibration_questions.jsonl \
  --documents /path/to/calibration_docs.zip \
  --out results/enterprise_rag/calibration.placeholder.answers.jsonl \
  --pool-size 4 \
  --reset-index --overwrite \
  --top-config \
  --calibrate-retrieval-out results/enterprise_rag/calibration.retrieval.json
```

The first 20-question calibration smoke found `k=8` to be the best launch setting in that sample:
it matched `k=10+` on document recall and exact gold-document coverage while returning fewer extra
documents.

The final submission package should include:

1. deterministic ingestion,
2. resumable question execution,
3. JSONL output compatible with the benchmark submission format,
4. a manifest with file digests,
5. streaming JSONL writes plus `--resume` so paid answer generation can survive restarts.

## Promotion Rule

A visibility run is ready to advertise only when all of these are true:

1. The result is externally visible or externally verified.
2. The exact code and configuration are reproducible from this repository.
3. The artifact records every prompt and model that can change the score.
4. A short limit statement has been written before the number is added to the README.
5. The claim gate either covers the number or the artifact is linked from `results/ARTIFACTS.md`.

## Immediate Work Queue

1. Kaggle AgentEval reconnaissance: capture data schema, runtime limits, submission format, and
   scoring columns.
2. Kaggle adapter: create a minimal end-to-end runner that indexes the provided context, answers
   with citations, and writes the required submission file.
3. EnterpriseRAG-Bench importer: map released sources into RE-call documents with stable ids and
   source metadata.
4. EnterpriseRAG-Bench runner: write answers, citations, and run manifest.
5. Public claim update: add a README row only after the first external result exists.
