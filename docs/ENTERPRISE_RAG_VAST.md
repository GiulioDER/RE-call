# EnterpriseRAG Vast.ai Runbook

This is the GPU runbook for the EnterpriseRAG-Bench SPLADE arm. VPS2 can run the dense plus lexical
arm, but the full SPLADE backfill should run on a GPU instance.

## Instance Requirements

Use an Ubuntu or CUDA PyTorch image with:

1. NVIDIA GPU visible to `torch.cuda`.
2. Python 3.11 or newer.
3. Git.
4. Either Docker access for `pgvector/pgvector:pg16`, or native Postgres with the `vector`
   extension installed.
5. At least 100 GB free disk. The release ZIP is about 1.2 GB, but the expanded chunks, Postgres
   indexes, sparse sidecar, model cache, and logs need room.

Set these secrets in the instance environment before running setup:

```bash
export VOYAGE_API_KEY=...
export OPENROUTER_API_KEY=...
```

If the instance already has a suitable Postgres plus pgvector database, also set:

```bash
export RECALL_DSN='postgresql://user:pass@host:5432/dbname'
```

If `RECALL_DSN` is absent, setup tries Docker pgvector first, then native user-owned Postgres.

## Setup

Run this on the Vast instance:

```bash
curl -fsSL https://raw.githubusercontent.com/GiulioDER/RE-call/codex/enterprise-rag-bench/scripts/enterprise_rag_vast_setup.sh | bash
```

The setup script:

1. Clones branch `codex/enterprise-rag-bench` into `/workspace/RE-call`.
2. Creates `.venv`.
3. Installs `.[voyage,sparse,bench,pool]`.
4. Downloads EnterpriseRAG-Bench `v1.0.0`.
5. Creates `.env` with `RECALL_DSN`, `VOYAGE_API_KEY`, and `OPENROUTER_API_KEY`.
6. Runs `scripts/enterprise_rag_vast_preflight.sh`.

The preflight refuses to proceed unless CUDA is actually usable for SPLADE.

## Smoke

Before the full run, execute one real end-to-end GPU smoke:

```bash
cd /workspace/RE-call
./scripts/enterprise_rag_vast_smoke.sh
```

This builds a five-document ZIP from the official release, including the first question's gold
document, then runs the full model stack on CUDA. It spends one answer call and a tiny number of
Voyage retrieval calls. The expected output is:

```bash
results/enterprise_rag/vast_top_splade_smoke.answers.jsonl
results/enterprise_rag/vast_top_splade_smoke.answers.jsonl.manifest.json
```

## Launch

Preferred launch after the VPS2 dense arm has completed:

1. Export the dense table on VPS2:

```bash
cd /home/sentiment/enterprise-rag-run/RE-call
nohup ./scripts/enterprise_rag_vps2_export_dense.sh \
  > logs/enterprise_rag_vps2_dense_export.log 2>&1 &
echo $! > enterprise_rag_vps2_dense_export.pid
```

2. Copy the dump to the Vast instance:

```bash
scp -P <vast_ssh_port> \
  sentiment@vps2:/home/sentiment/enterprise-rag-run/enterprise_rag_dense_12k.pgcustom \
  root@<vast_ip>:/workspace/enterprise_rag_dense_12k.pgcustom
```

If `vps2` is a local Tailscale alias only, copy through the local machine instead:

```bash
scp vps2:/home/sentiment/enterprise-rag-run/enterprise_rag_dense_12k.pgcustom .
scp -P <vast_ssh_port> enterprise_rag_dense_12k.pgcustom root@<vast_ip>:/workspace/
```

3. Import the dense table on Vast:

```bash
cd /workspace/RE-call
./scripts/enterprise_rag_vast_import_dense.sh /workspace/enterprise_rag_dense_12k.pgcustom
```

4. Run SPLADE backfill on CUDA and then answer from the imported table:

```bash
cd /workspace/RE-call
nohup ./scripts/enterprise_rag_vast_splade_from_import.sh \
  > logs/enterprise_rag_vast_splade_from_import.log 2>&1 &
echo $! > enterprise_rag_vast_splade_from_import.pid
```

Fallback launch, only if the dense dump is unavailable:

```bash
cd /workspace/RE-call
nohup ./scripts/enterprise_rag_vast_run.sh \
  > logs/enterprise_rag_vast_top_splade.log 2>&1 &
echo $! > enterprise_rag_vast_top_splade.pid
```

The fallback run has two phases:

1. `scripts/enterprise_rag_vast_index.sh`: indexes the full corpus and backfills SPLADE on CUDA.
   On success it writes `results/enterprise_rag/vast_top_splade_full.index.done`.
2. `scripts/enterprise_rag_vast_run.sh`: answers all questions with `--skip-index --resume`, so a
   restarted answer phase does not redo indexing or SPLADE backfill.

The measured config is:

1. Voyage `voyage-4-large` embeddings.
2. Postgres lexical plus SPLADE learned sparse retrieval.
3. Voyage `rerank-2.5`.
4. `openai/gpt-4o`.
5. `candidate_k=200`.
6. `k=8`.
7. `gap_threshold=0.5`.
8. `max_context_chars=12000`.

## Monitor

```bash
cd /workspace/RE-call
tail -f logs/enterprise_rag_vast_top_splade.log
wc -l results/enterprise_rag/vast_top_splade_full.answers.jsonl 2>/dev/null || true
nvidia-smi
```

Expected outputs:

1. During SPLADE backfill, the log prints `splade backfill written=...`.
2. During answering, the log prints `answered qst_...`.
3. The final artifacts are:
   `results/enterprise_rag/vast_top_splade_full.answers.jsonl` and
   `results/enterprise_rag/vast_top_splade_full.answers.jsonl.manifest.json`.

## Export Back To VPS2

After SPLADE backfill, the chunk table alone is not enough. SPLADE vectors live in
`recall_sparse_v1`, keyed by `(tenant_id, chunk_table, profile_id, id)`. A close signal for the GPU
instance is only valid after exporting both the chunk table and `recall_sparse_v1`, then restoring
and verifying the sidecar row count on VPS2.

On Vast, create the full transferable dump:

```bash
cd /workspace/RE-call
./scripts/enterprise_rag_vast_export_splade.sh
```

The export refuses to proceed unless the source database has at least one SPLADE row for every
chunk id in the EnterpriseRAG tenant. It also checks the custom dump manifest for:

```text
TABLE DATA public ber_voy_lex_12k_full
TABLE DATA public recall_sparse_v1
```

On VPS2, restore with SPLADE required:

```bash
cd /home/sentiment/enterprise-rag-run/RE-call
ENTERPRISE_RAG_REQUIRE_SPLADE=1 \
  ./scripts/enterprise_rag_vast_import_dense.sh /home/sentiment/enterprise-rag-run/enterprise_rag_splade_full.pgcustom
```

Do not close the Vast instance until the VPS2 restore reports `sparse_ids` equal to the chunk count.

## Work That Can Be Done Before Renting

Already done locally or on VPS2:

1. PR branch exists: `codex/enterprise-rag-bench`.
2. Official EnterpriseRAG data download path is validated.
3. Runner ingest for the official ZIP text layout is tested.
4. One-question top config smoke passed.
5. 20-question calibration selected `k=8`.
6. VPS2 dense plus lexical full arm is running separately.

Useful work still possible without the GPU:

1. Keep monitoring the VPS2 dense plus lexical arm.
2. Export `ber_voy_lex_12k_full` from VPS2 with `scripts/enterprise_rag_vps2_export_dense.sh`.
3. Prepare the leaderboard submission email draft after the Vast artifacts exist.
4. Run `scripts/enterprise_rag_vast_smoke.sh` immediately after setup on the rented instance.
