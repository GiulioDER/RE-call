# ATM Bench full run package

This run package is prepared for VPS2. It does not start provider calls by itself.

## Inputs

The official full split contains 1,013 questions. The local files are:

```text
C:\Users\gde00\Documents\atm-bench-official\data\atm-bench\atm-bench.json
C:\Users\gde00\Documents\atm-bench-official\output\image\qwen3vl2b\batch_results.json
C:\Users\gde00\Documents\atm-bench-official\output\video\qwen3vl2b\batch_results.json
C:\Users\gde00\Documents\atm-bench-official\data\raw_memory\email\emails.json
```

The latest source bundle is:

```text
.tmp_atm_vps2_bundle\recall-source-a691ae23.tar.gz
```

Its SHA256 is recorded by the transfer step and must be checked on VPS2 before extraction.
The current bundle SHA256 is `63342662F64C1AE75515273C363D509900917F68BAB150B90C79F036FADBC443`.

## RE call configuration

The launch configuration is fixed in
`docs/preregistrations/2026-08-19-atm-full-voyage4-deepseek.md`:

```text
embedder: voyage:voyage-4-large
retrieval: dense plus lexical hybrid
reranker: voyage:rerank-2.5
candidate pool: 25
answer retrieval cutoff: 10
evidence budget: 8,192 characters, approximately 2,048 tokens
answer model: deepseek/deepseek-v4-pro through OpenRouter
reasoning request: medium
answer output ceiling: 1,024 tokens
```

The `candidate pool` is 25 because the previously tested larger Voyage pools exceeded the
observed 2,000,000 rerank token per minute project limit. The index uses a new isolated table and
tenant, so the existing production RE call index is not modified.

## VPS2 preparation

Run these commands from the local repository. They transfer only the source bundle and public
benchmark data. They do not transfer environment files or credentials.

```powershell
$bundle = Resolve-Path .tmp_atm_vps2_bundle\recall-source-a691ae23.tar.gz
scp $bundle vps2:/home/sentiment/atm-bench-run/recall-source-a691ae23.tar.gz
scp C:\Users\gde00\Documents\atm-bench-official\data\atm-bench\atm-bench.json vps2:/home/sentiment/atm-bench-run/data/atm-bench/atm-bench.json
scp C:\Users\gde00\Documents\atm-bench-official\output\image\qwen3vl2b\batch_results.json vps2:/home/sentiment/atm-bench-run/data/image/batch_results.json
scp C:\Users\gde00\Documents\atm-bench-official\output\video\qwen3vl2b\batch_results.json vps2:/home/sentiment/atm-bench-run/data/video/batch_results.json
scp C:\Users\gde00\Documents\atm-bench-official\data\raw_memory\email\emails.json vps2:/home/sentiment/atm-bench-run/data/emails.json
```

On VPS2, verify the bundle digest and extract it into the isolated source directory:

```bash
cd /home/sentiment/atm-bench-run
sha256sum recall-source-a691ae23.tar.gz
rm -rf source
mkdir source
tar -xzf recall-source-a691ae23.tar.gz -C source
```

The `rm` above targets only the explicitly named isolated source directory under the benchmark
run root. It must not be changed to the production repository path.

## No cost preflight

This validates the dataset counts and the selected configuration without constructing an index and
without calling Voyage or OpenRouter:

```bash
cd /home/sentiment/atm-bench-run
/home/sentiment/recall-repos/.venv/bin/python source/benchmarks/atm_full_run.py \
  --qa-file data/atm-bench/atm-bench.json \
  --image-file data/image/batch_results.json \
  --video-file data/video/batch_results.json \
  --email-file data/emails.json \
  --out-dir results/voyage4-deepseek-full \
  --dry-run
```

## Full run after approval

Load the two existing environment files through filtered assignment lines. Do not print the
environment or inspect secret values:

```bash
set -a
source <(grep -E "^[A-Za-z_][A-Za-z0-9_]*=" /opt/sentiment_agent/.env)
source <(grep -E "^[A-Za-z_][A-Za-z0-9_]*=" /home/sentiment/recall-repos/.env)
set +a
cd /home/sentiment/atm-bench-run
/home/sentiment/recall-repos/.venv/bin/python source/benchmarks/atm_full_run.py \
  --qa-file data/atm-bench/atm-bench.json \
  --image-file data/image/batch_results.json \
  --video-file data/video/batch_results.json \
  --email-file data/emails.json \
  --out-dir results/voyage4-deepseek-full
```

The runner checkpoints `retrieval.jsonl` and `answers.jsonl` after every question. Repeating the
same command resumes completed questions and does not regenerate their answer calls.

## Official judge after answer completion

The official evaluator remains the ATM Bench evaluator with `gpt-5-mini` and its checked in
official prompt. Run it from the official ATM Bench repository, using the answer JSONL created on
VPS2:

```bash
cd /path/to/atm-bench-official
PYTHONPATH=. python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth /home/sentiment/atm-bench-run/data/atm-bench/atm-bench.json \
  --predictions /home/sentiment/atm-bench-run/results/voyage4-deepseek-full/answers.jsonl \
  --output-dir /home/sentiment/atm-bench-run/results/voyage4-deepseek-full/official_eval \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --judge-fallback-model gpt-4o-mini \
  --judge-fallback-after-retries 3 \
  --request-delay 1 \
  --judge-max-retries 3 \
  --max-workers 4 \
  --metrics llm atm
```

The official evaluator accepts prediction rows in the form `{"id": "...", "answer": "..."}`.
The runner writes exactly that form.

## Cost gate

The full run is not started in this preparation step. I recommend an eight dollar reserve for the
answerer and official judge, with Voyage free credits assumed available. If those credits are
exhausted, the Voyage embedding and reranking charges add a separate amount. The answer output
ceiling is explicit so the estimate cannot drift through an unbounded provider default.
