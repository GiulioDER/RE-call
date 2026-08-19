# ATM-Bench GPU arm

This runbook describes the no-API-credit SPLADE arm for ATM-Bench. It is
intended for a rented Vast.ai GPU because the local Windows host has no CUDA
device and CPU SPLADE would not give a useful latency measurement.

## Frozen configuration

Use the same official ATM descriptions and question files as the local run.
Keep the embedding profile, sparse profile, candidate pool, and question split
in the output manifest.

The first GPU comparison should be:

* `fastembed:BAAI/bge-small-en-v1.5` with `--sparse-backend lexical`
* the same embedder with `--sparse-backend splade`

Both use `candidate-k=200`, the pinned local MiniLM reranker only in a separate
arm, and the development split for selection. SPLADE replaces the lexical leg
in the `splade` arm. Do not combine lexical and SPLADE in the first comparison.

## Host setup

Use a reliable RTX 4090, RTX 5090, or equivalent with at least 16 GB VRAM and
the PyTorch image. Install the repository and its benchmark dependencies, then
start the local Postgres service:

```bash
git clone <repository-url> RE-call
cd RE-call
pip install -e '.[fastembed,rerank]'
docker compose up -d db
```

Copy the official ATM description files to the instance. Do not copy raw media
when the description files are sufficient for the retrieval experiment.

## SPLADE development run

```bash
python -m benchmarks.atm_bench \
  --qa-file /data/atm-bench.json \
  --image-file /data/image/batch_results.json \
  --video-file /data/video/batch_results.json \
  --email-file /data/emails.json \
  --out results/atm_bench_dev_bge_small_splade.json \
  --table bench_atm_bge_small_splade_chunks \
  --tenant atm-bench-bge-small-splade-20260819 \
  --embedder fastembed:BAAI/bge-small-en-v1.5 \
  --candidate-k 200 \
  --sparse-backend splade \
  --sparse-model prithivida/Splade_PP_en_v1 \
  --sparse-device cuda \
  --sparse-batch-size 64 \
  --question-split development \
  --arms dense hybrid
```

Run the lexical control with a different table and tenant. The dense arm is
unchanged in both runs, while the hybrid arm differs only in its sparse leg.

## Holdout and hard checks

After the development comparison is frozen, run the selected arm with
`--question-split holdout`. Run the 31-question hard file as a separate
external check. Do not change the threshold, embedder, candidate pool, or
sparse model after reading those results.

## Transfer and cost boundary

Copy back the JSON result files and their per-arm retrieval detail files. The
SPLADE sidecar is only needed when the index is reused on the GPU instance. The
local score claim is the JSON artifact, not a claim that the Windows host can
serve SPLADE at the same latency.

Do not start a Vast.ai instance or call Voyage APIs without explicit credit
authorization. The SPLADE model itself is local and does not require an API
call.

