# How this artifact was produced

**Prior work.** `docs_search(source_type="memory")` was run this session on MTRAG retrieval and on
MTRAG generation, and returned substantial prior art, none of which covers store latency
attribution. That matches what `benchmarks/store_latency_share.py`'s own module docstring already
records from 2026-08-04: *"no memo covers it; every hit was about retrieval QUALITY. The 9a5165b
figure came from git, not memory."* The one prior measurement of these legs is commit `9a5165b`
(sparse median 496 ms on a real 72k-chunk corpus), which is cited in the caveat block of
`SPLIT.md` rather than re-measured here.

`splits.json` and `SPLIT.md` in this directory were measured on 2026-08-07 on a **rented vast.ai
instance**, not on VPS2. This note records the facts that were in the run log, which was lost with
the instance before it could be copied down. Everything below was read from that log while it
existed; nothing here is reconstructed or inferred.

## Why not VPS2

VPS2 was the intended host and cannot do this measurement. Sampled six times over a minute on
2026-08-07 it read 12.85, 12.03, 12.39, 11.82, 12.41 and 15.94 on 12 cores, so its **floor is about
1.0 load per core** because it runs 64 services and 170 timers continuously. The quiescence guard's
0.30 per-core ceiling is not a bar that host clears when it is quiet; it is a bar that host never
clears.

VPS2 also measured the SPLADE corpus encode at **0.38 chunks/s**, which projected a 20,000 chunk
run to roughly **15 hours**, across which its load varied by a factor of three.

## The host that was used

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, capability (12, 0) |
| CPU / RAM | 192 cores, 503 GB |
| torch | 2.11.0+cu128, `arch_list` includes `sm_120` |
| Postgres | 16, pgvector **0.8.0** (image shipped 0.6.0, which predates `sparsevec`; rebuilt from source) |
| Load at launch | 6.30 on 192 cores = 0.033 per core |

The quiescence guard **passed on merit**: `host_load_override` is `false` in `splits.json`, with
0.0239 per core before and 0.2409 after, against the 0.30 default ceiling. This is the only
measurement in this directory taken on a genuinely quiet machine.

## Measured stage timings, from the lost log

```
  200 files, 20100 chunks, 100 timed queries
  indexed 20100 chunks in 1110.5s
  encoded in 96.3s
```

- **Dense index: 18.1 chunks/s.** fastembed `bge-small-symmetric-v1` on CPU. The GPU sat at 0%
  during this stage and it was the longest one, so renting a GPU bought nothing here.
- **SPLADE corpus encode: 208.7 chunks/s** on the 5090, against 0.38 chunks/s measured on VPS2.
  That is the stage the GPU was rented for.
- **SPLADE query encode: 12 to 18 ms** per query (the `splade enc` column), against 265.5 ms
  measured on a CPU box.

The encoder line read `learned sparse device: cuda (requested cuda)` and the profile fingerprint
was `c2be003cbedcaa47bb5c7f952d44826b5ef6b091c4728f9a92e0e9a8f30add11`, **identical** to the
fingerprint produced on a Windows CPU box and on VPS2.

⚠️ That identity is of the PROFILE, not of the vectors. `SparseProfile` does not include the
device, and the encode ends in `log1p(relu(logits))` followed by top-k pruning to 1000 terms, so a
near-tie at the k-th position can resolve differently under GPU and CPU float arithmetic. These
vectors are **not** proven bit-identical to CPU-encoded ones.

## What is missing from this artifact, deliberately

No `--rerank` configuration was measured. On a CPU box the cross-encoder dominates the wall clock,
and its absence **inflates the store's share**, which is a bias toward "porting the store is worth
it". The caveat block inside `SPLIT.md` states this alongside the numbers.
