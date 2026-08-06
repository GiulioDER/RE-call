# Next GPU run: BGE-reranker-v2-m3 over the frozen pools

**Question:** 123 gold documents are reachable only via SPLADE, and the 22M-parameter MiniLM
buries 73% of them below rank 10 (median rank 29). Does a modern reranker convert them?

**Why it is cheap:** the candidate pools are frozen on disk. Nothing is re-encoded, re-indexed or
re-retrieved. The GPU re-scores the same 179,403 pairs with a different model; everything else is
CPU arithmetic on VPS2.

**Model:** `BAAI/bge-reranker-v2-m3`, **apache-2.0** (licence-clean, unlike the SPLADE
checkpoints), ungated, 0.6B params, bge-m3 base.
**Pin:** `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (resolved 2026-08-06).

## What is already staged on VPS2

| path | what |
|---|---|
| `/var/tmp/re_call_splade_20260806/offload_payload.tar.gz` | 43MB: queries + docs + pairs, ready to ship |
| `/var/tmp/re_call_splade_20260806/offload/pools/` | the five frozen arms |
| `/var/tmp/re_call_splade_20260806/offload/scores.jsonl` | MiniLM scores, the comparison baseline |
| `/var/tmp/re_call_splade_20260806/RE-call/scripts/score_pairs.py` | takes `--model` / `--revision` |
| `/var/lib/recall-benchmarks/2026-08-06-mtrag-splade-learned-sparse/` | archived run + manifest |

## Instance sizing

3090-class or better, 24GB is ample. Throughput is the open number: MiniLM managed 471 pairs/s
while sharing the GPU with an encode. BGE-v2-m3 is ~25x the parameters, so **expect 50-150
pairs/s, i.e. 20-60 minutes** for 179,403 pairs. That is an extrapolation, not a measurement —
step 3 measures it on 2,000 pairs first and prints a projection before the full run.

## The run, in order

Replace `PORT` and `HOST` with the new instance's SSH details. Everything runs from VPS2.

### 0. Attach VPS2's key to the instance (the fiddly step last time)

Paste this **public** key into the vast.ai console for the instance (key icon), or use
`vastai attach ssh <ID> "$(cat /root/.ssh/id_ed25519.pub)"` if the API key is set:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIITQ9dZ2rD+RO2Z43GYeaiz6ZLD7S7bEAG38WF4bhYpk vast-prr-20260629
```

Verify: `ssh -p PORT root@HOST 'nvidia-smi --query-gpu=name --format=csv,noheader'`

### 1. Ship (from VPS2)

```bash
R=/var/tmp/re_call_splade_20260806
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25"
scp $OPTS -P PORT "$R/offload_payload.tar.gz" root@HOST:/workspace/
scp $OPTS -P PORT "$R/RE-call/scripts/score_pairs.py" root@HOST:/workspace/
ssh  $OPTS -p PORT root@HOST 'cd /workspace && mkdir -p offload && tar -xzf offload_payload.tar.gz -C offload && wc -l offload/*.jsonl'
```
Expect `100125 docs`, `179403 pairs`, `777 queries`.

### 2. Deps on the instance

⚠️ `pip install --upgrade pip` FAILS on these Debian images (cannot uninstall the distro pip) and
`set -e` will abort the script. ⚠️ `python` does not exist, only `python3`. Both cost a run today.

```bash
ssh $OPTS -p PORT root@HOST 'pip install --quiet sentence-transformers 2>&1 | tail -2; python3 -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_available())"'
```

### 3. MEASURE FIRST (2,000 pairs), then decide

```bash
ssh $OPTS -p PORT root@HOST 'cd /workspace && PYTHONPATH=/workspace python3 score_pairs.py \
  --queries offload/queries.jsonl --docs offload/docs.jsonl --pairs offload/pairs.jsonl \
  --output offload/probe_bge.jsonl --batch-size 64 --limit 2000 \
  --model BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e | tail -2'
```
Read `rate_per_s`. 179403/rate = seconds for the full run. **Decide before proceeding.**
Batch 64 rather than 256: this model is 25x larger and 256 risks OOM.

### 4. Full scoring

```bash
ssh $OPTS -p PORT root@HOST 'cd /workspace && PYTHONPATH=/workspace nohup python3 score_pairs.py \
  --queries offload/queries.jsonl --docs offload/docs.jsonl --pairs offload/pairs.jsonl \
  --output offload/scores_bge.jsonl --batch-size 64 \
  --model BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  > score_bge.log 2>&1 & echo launched'
```

### 5. Fetch, then DESTROY the instance

```bash
scp $OPTS -P PORT root@HOST:/workspace/offload/scores_bge.jsonl "$R/offload/scores_bge.jsonl"
wc -l "$R/offload/scores_bge.jsonl"   # expect 179404 (179403 + header)
```
Nothing after this needs a GPU.

### 6. Gate, metrics, contrasts (VPS2, CPU, minutes)

```bash
cd $R/RE-call && export PYTHONPATH=$R/RE-call

# GATE FIRST. --model/--revision are load-bearing: without them the gate would compare BGE
# scores against a MiniLM ordering and fail for the wrong reason.
$R/.venv/bin/python -m benchmarks.mtrag.rerank_offload validate \
  --output-dir $R/offload --scores $R/offload/scores_bge.jsonl --sample 20 \
  --model BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e

# Only if MATCH:
$R/.venv/bin/python -m benchmarks.mtrag.rerank_offload apply \
  --mtrag-root /var/tmp/re_call_mtrag_20260803/mt-rag-benchmark \
  --output-dir $R/offload --split dev --scores $R/offload/scores_bge.jsonl
# writes reranked_summary__scores_bge.json -- it will NOT overwrite the MiniLM run

$R/.venv/bin/python -m benchmarks.mtrag.analyse_contrasts \
  --offload-dir $R/offload --mtrag-root /var/tmp/re_call_mtrag_20260803/mt-rag-benchmark \
  --split dev --scores $R/offload/scores_bge.jsonl
```

⚠️ The gate on VPS2 CPU runs 20 queries x ~200 candidates through a 0.6B model. That is far
slower than the MiniLM gate (which took ~20 min). Consider `--sample 5`, or run the gate on the
GPU box before destroying it.

## What the result means

Two numbers decide it, both from step 6:

- **reranked nDCG@5, `hybrid_splade` vs `hybrid_lexical`.** MiniLM gave -0.0043, CI spanning zero.
  If BGE turns this significantly positive, the buried gold is being converted and the reranker
  was the binding constraint.
- **The `coverage_destination` block.** With MiniLM: 73% of SPLADE-only gold below rank 10,
  median 29. If BGE pulls that median into single digits, that is the mechanism, visible directly
  rather than inferred from an aggregate.

If BGE does **not** move it, the conclusion is different and worth knowing: the extra evidence is
hard for any cross-encoder to rank, and the next lever is query-side (multi-query diversity),
not ranking.

## Traps that cost time today

1. `pkill -f <pattern>` matches its own ssh command line and kills the shell running it. Use
   `[p]attern`.
2. Heredocs over ssh mangle quotes. Use `scp` for anything non-trivial.
3. `pgrep` on the encoder said IDLE while the script's `tar` stage was still running, so a
   half-written tarball got hashed. Check the job, not one of its stages.
4. `scp` takes `-P` for port, `ssh` takes `-p`.
5. `--limit` before a long run is not optional. Three of today's estimates were wrong by 8x-13x.
