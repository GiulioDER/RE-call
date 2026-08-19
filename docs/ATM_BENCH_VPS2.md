# ATM-Bench VPS2 preparation

This bundle runs beside the existing VPS2 checkout. It does not copy over the
dirty RE-call tree and it does not restart any live service.

## Files to transfer

Transfer these files from the local workspace:

* `benchmarks/atm_bench.py`
* `benchmarks/atm_pipeline_calibration.py`
* `benchmarks/atm_list_selection_probe.py`
* `docs/preregistrations/2026-08-19-atm-pipeline-calibration.md`
* `docs/preregistrations/2026-08-19-atm-vps2-voyage-reasoning.md`
* `results/atm_bench_full_pipeline_calibration_20260819.json`
* the official ATM files listed below

The runner also needs the RE-call Python package and its dependencies. The
transfer must therefore use a clean source snapshot of this branch, not only
the three benchmark files. Put it under `/home/sentiment/atm-bench-run/source`
and execute it with its own virtual environment or the branch’s locked
environment. Do not touch `/home/sentiment/recall-repos/engine`.

Official data to transfer:

* `data/atm-bench/atm-bench.json`
* `data/atm-bench/atm-bench-hard.json`
* `output/image/qwen3vl2b/batch_results.json`
* `output/video/qwen3vl2b/batch_results.json`
* `data/raw_memory/email/emails.json`

Raw images and raw videos are not needed for this text-description run. Do not
transfer `.env`, API keys, Hugging Face caches, or unrelated benchmark outputs.

## Isolated environment

The command must source the existing VPS2 secret file without printing it:

```bash
set -a
. /opt/sentiment_agent/.env
set +a
export RECALL_REASONING=1
export RECALL_REASONING_BASE_URL=https://openrouter.ai/api/v1
export RECALL_REASONING_API_KEY="$OPENROUTER_API_KEY"
export RECALL_REASONING_MODEL=openai/gpt-5-mini
```

The benchmark process receives `VOYAGE_API_KEY` from the existing secret file.
The reasoning key is used only by the isolated process and is never written to
the transfer bundle.

## VPS2 commands

Run the MiniLM control first:

```bash
cd /home/sentiment/atm-bench-run/source
python -m benchmarks.atm_bench \
  --qa-file /home/sentiment/atm-bench-run/data/atm-bench/atm-bench.json \
  --image-file /home/sentiment/atm-bench-run/data/image/batch_results.json \
  --video-file /home/sentiment/atm-bench-run/data/video/batch_results.json \
  --email-file /home/sentiment/atm-bench-run/data/emails.json \
  --out /home/sentiment/atm-bench-run/results/atm_vps2_minilm_voyage_rerank.json \
  --table bench_atm_vps2_minilm_chunks \
  --tenant atm-vps2-minilm-20260819 \
  --embedder fastembed:sentence-transformers/all-MiniLM-L6-v2 \
  --candidate-k 200 \
  --reranker voyage:rerank-2.5 \
  --sparse-backend lexical \
  --question-split all \
  --arms dense hybrid
```

Run the Voyage embedding comparison in a separate table and tenant:

```bash
python -m benchmarks.atm_bench \
  --qa-file /home/sentiment/atm-bench-run/data/atm-bench/atm-bench.json \
  --image-file /home/sentiment/atm-bench-run/data/image/batch_results.json \
  --video-file /home/sentiment/atm-bench-run/data/video/batch_results.json \
  --email-file /home/sentiment/atm-bench-run/data/emails.json \
  --out /home/sentiment/atm-bench-run/results/atm_vps2_voyage4_voyage_rerank.json \
  --table bench_atm_vps2_voyage4_chunks \
  --tenant atm-vps2-voyage4-20260819 \
  --embedder voyage:voyage-4-large \
  --candidate-k 200 \
  --reranker voyage:rerank-2.5 \
  --sparse-backend lexical \
  --question-split all \
  --arms dense hybrid
```

The reasoning flag is captured in the process environment and must also be
recorded in a run manifest. The current retrieval runner does not turn
reasoning into an answer score, so the reasoning audit and retrieval artifact
must remain separate.

## Transfer back

Transfer only the JSON result files, per-arm detail files, calibration output,
and the process log. Verify SHA-256 hashes before reading the results. Do not
export database dumps or sparse vectors until the later SPLADE run requires
them.
