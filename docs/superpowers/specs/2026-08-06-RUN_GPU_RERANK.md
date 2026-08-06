# GPU run: does the multi-query coverage CONVERT under reranking?

**Question.** The raw run moved R@100 from 0.7377 to 0.8613 (+0.1236, CI [+0.1023, +0.1457]). Every
number in it is raw. SPLADE moved R@100 +0.0303 and its ranking gain did **not** follow. Of the 279
gold documents multi-query adds, **200 sit below rank 10** raw. This settles whether the headline is
something a reader ever sees.

Preregistration: `docs/superpowers/specs/2026-08-06-multi-query-rerank-design.md`. Read the
"design decision" section before quoting any number: R@100 is **invariant by construction** in the
primary analysis.

**Why it is cheap.** Nothing is retrieved, encoded or indexed. The pools are frozen on disk. The
GPU re-scores fixed `(query, passage)` pairs and everything else is CPU arithmetic on VPS2.

## Cost, measured not guessed

CPU was tried first and abandoned: MiniLM on VPS2 sustained **5.2 to 6.7 pairs/s** (MTRAG passages
hit the 512-token limit, so every pair is a full-length forward pass). That is ~4.7 h for a third
of the payload.

| model | params | expected rate | 241,270 pairs |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (**primary**, RE-call's default) | 22M | ~470/s measured previously while sharing a GPU | **~9 min** |
| `BAAI/bge-reranker-v2-m3` (secondary) | 568M | 50 to 150/s, extrapolated | **27 to 80 min** |

Both fit comfortably in one hour on a 3090-class instance, 24GB ample. Step 3 measures the real
rate on 2,000 pairs and prints a projection before committing to the full run.

## Payload, already staged on VPS2

| path | what |
|---|---|
| `/var/tmp/re_call_splade_20260806/mqrr_payload.tar.gz` | **35MB**, ready to ship |
| sha256 | `b18f7fd929c930fa6a8f8ff566a7c120a5fc1e46695d7d0b13249813b5449518` |
| `/var/tmp/re_call_splade_20260806/mqrr/queries.jsonl` | 777 queries |
| `/var/tmp/re_call_splade_20260806/mqrr/docs.jsonl` | 91,933 passages |
| `/var/tmp/re_call_splade_20260806/mqrr/pairs.jsonl` | **241,270 pairs** |
| `/var/tmp/re_call_splade_20260806/mqrr/rankings_equal_width.json` | primary: each arm's top 100 |
| `/var/tmp/re_call_splade_20260806/mqrr/rankings_whole_pool.json` | secondary: full pools |

⚠️ **No DSN, no credentials leave VPS2.** `scripts/score_pairs.py` takes queries, docs and pairs and
nothing else, for the same reason `encode_sparse.py` does: the only stage that wants rented
hardware is the only stage that runs on a machine nobody trusts.

## The run, in order

Replace `PORT` and `HOST` with the instance's SSH details. Everything is driven from VPS2.

### 0. Attach VPS2's key to the instance

Paste this **public** key into the vast.ai console for the instance (key icon):

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cat /root/.ssh/id_ed25519.pub'
```

### 1. Ship the payload and the scorer

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'scp -P PORT /var/tmp/re_call_splade_20260806/mqrr_payload.tar.gz /var/tmp/re_call_splade_20260806/RE-call/scripts/score_pairs.py root@HOST:/workspace/'
```

⚠️ `scp` takes `-P`, `ssh` takes `-p`. This has cost time before.

### 2. Unpack and install on the instance

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'ssh -p PORT root@HOST "cd /workspace && tar xzf mqrr_payload.tar.gz && pip -q install sentence-transformers && python -c \"import torch;print(torch.cuda.get_device_name(0))\""'
```

### 3. Measure the real rate on 2,000 pairs FIRST

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'ssh -p PORT root@HOST "cd /workspace && head -2000 pairs.jsonl > pairs_probe.jsonl && time python score_pairs.py --queries queries.jsonl --docs docs.jsonl --pairs pairs_probe.jsonl --output probe.jsonl --batch-size 256"'
```

Read `rate_per_s` off the progress lines and multiply out. If MiniLM is under ~200/s the instance is
throttled or sharing; fix that before paying for the full run.

### 4. Full run, primary model

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'ssh -p PORT root@HOST "cd /workspace && nohup python score_pairs.py --queries queries.jsonl --docs docs.jsonl --pairs pairs.jsonl --output scores_minilm.jsonl --batch-size 256 > score_minilm.log 2>&1 &"'
```

### 5. Full run, secondary model

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'ssh -p PORT root@HOST "cd /workspace && nohup python score_pairs.py --queries queries.jsonl --docs docs.jsonl --pairs pairs.jsonl --output scores_bge.jsonl --model BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --batch-size 64 > score_bge.log 2>&1 &"'
```

⚠️ Check liveness of the **job**, not one of its stages. A tarball once got hashed mid-write
because the encoder had exited while `tar` was still running. Confirm the process is gone AND the
line count matches 241,270 before copying anything back.

### 6. Bring the scores home

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'scp -P PORT root@HOST:/workspace/scores_minilm.jsonl root@HOST:/workspace/scores_bge.jsonl /var/tmp/re_call_splade_20260806/mqrr/ && wc -l /var/tmp/re_call_splade_20260806/mqrr/scores_*.jsonl'
```

### 7. ⛔ VALIDATE before believing any number

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cd /var/tmp/re_call_splade_20260806/RE-call && ../.venv/bin/python -m benchmarks.mtrag.rerank_multiquery validate --mq-dir /var/tmp/re_call_splade_20260806/mq --output-dir /var/tmp/re_call_splade_20260806/mqrr --scores /var/tmp/re_call_splade_20260806/mqrr/scores_minilm.jsonl --sample 20'
```

Must print `"verdict": "MATCH"`. This runs the REAL `CrossEncoderReranker` locally and requires the
offloaded ordering to match where metrics are cut. An ordering that merely looks reasonable
produces publishable nDCG that RE-call would never compute, and nothing about it looks wrong.

⚠️ Validating BGE scores against MiniLM's ordering fails for a reason that has nothing to do with
the offload. For the BGE file pass `--model BAAI/bge-reranker-v2-m3 --revision 953dc6f...` if the
validate step grows that flag; otherwise validate MiniLM only and treat BGE as secondary evidence.

### 8. Apply and decide

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cd /var/tmp/re_call_splade_20260806/RE-call && ../.venv/bin/python -m benchmarks.mtrag.rerank_multiquery apply --mq-dir /var/tmp/re_call_splade_20260806/mq --output-dir /var/tmp/re_call_splade_20260806/mqrr --mtrag-root /var/tmp/re_call_mtrag_20260803/mt-rag-benchmark --scores /var/tmp/re_call_splade_20260806/mqrr/scores_minilm.jsonl'
```

Writes `rerank_decision.json`. The verdict is one of `MATERIALLY_CONVERTS`, `CONVERTS_BUT_BELOW_BAR`,
`DOES_NOT_CONVERT`, `REVERSES`, per the preregistered rule on **C1 = mq_nested3 − mq_last, nDCG@5**.

### 9. Destroy the instance

Vast bills while it exists, not while it computes.

## What is preregistered, so it cannot be chosen afterwards

- **Decision metric nDCG@5.** R@100 is invariant in the primary analysis by construction; the code
  asserts the set-invariance rather than reporting the number.
- **Primary C1**, `mq_nested3` − `mq_last`. Converts iff CI excludes zero and Holm-significant;
  materially converts iff also >= +0.010.
- **C2** `mq_nested2_nogold` − `mq_last` decides whether the only deployable arm is still blocked
  by the ranking regression that was −0.0447 raw.
- **Predicted: C1 small and possibly null, 0.00 to +0.02.** If it lands above +0.03 my model of
  where the added gold sits is wrong.
- The whole-pool numbers are **confounded with pool width** (medians 167 / 315 / 284) and are
  secondary. Do not compare them against the equal-width contrast.

## Traps that have already cost time on this project

1. `scp` takes `-P`, `ssh` takes `-p`.
2. Heredocs over ssh mangle quotes. Use `scp` for anything non-trivial.
3. `pkill -f <pat>` matches its own ssh command line. Use `[p]at`.
4. Check liveness of the JOB, not one of its stages.
5. VPS2 is shared and rebooted on a kernel upgrade mid-session on 2026-08-06. Re-check `uptime`
   before assuming a long run survived.
