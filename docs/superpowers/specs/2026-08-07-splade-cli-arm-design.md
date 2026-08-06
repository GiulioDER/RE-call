# A selectable, measured SPLADE arm: learned sparse indexing as a library operation

`design · 2026-08-07`

## Why

`benchmarks/store_latency_share.py` carries a known limitation in its own docstring (lines 34 to 42
at `4983f44`):

> The CLI below cannot yet select the arm, because `splade` needs a `sparse_encoder` and this file
> has no flag to build one.

Every committed artifact under `results/store_latency/` is therefore a lexical run, and the learned
sparse leg of that benchmark has never been exercised against a real checkpoint. The attribution
code for it is correct and guarded, and nothing has ever run through it.

Closing this needs more than an argparse flag, and the docstring's framing of the gap is
incomplete. See the next section.

## What is actually missing

`benchmarks/mtrag/run.py:476` already builds a real `SpladeEncoder` from a `--sparse-model` flag,
so the CLI gap is specific to `store_latency_share.py`, not general.

Adding a flag there would not make the arm work. That benchmark indexes a synthetic corpus through
`recall.eval.harness._throwaway_store`, which calls `Indexer(store, emb).index_path(...)`. `Indexer`
writes the dense vector and the Postgres full text column. It never writes the learned sparse
sidecar. `store.query_learned_sparse` raises `LookupError` against an unencoded corpus, so the arm
would die at the fire rate probe before it timed anything.

The real gap is that **encoding a corpus into the learned sparse sidecar is not an operation the
`recall` package offers.** It exists only as two offline scripts that hand a JSONL file between
them, `scripts/encode_sparse.py` and `scripts/load_sparse.py`, neither importable as a function.
That split is deliberate and correct for the rented-GPU case (the encoder takes no DSN, so a
reclaimed spot instance never held a database credential), and it is the wrong shape for anything
running in one process against a store it already has.

## Decisions recorded from this session

These were settled with the user on 2026-08-07 and are not reopened here.

| Question | Decision |
|---|---|
| Where the encode path lives | Library API in `recall/sparse.py`, **plus** `Indexer` integration |
| What counts as tested | Both arms measured on **VPS2**, not on the local Windows box |
| Host contention | Ship a quiescence guard, then run when the box is quiet |
| Launch | `nohup`, detached |

## Measured platform facts

Established by probing on 2026-08-07, not read from documentation.

| Fact | Value | Consequence |
|---|---|---|
| VPS2 cores / RAM | 12 / 47 GB | the load ceiling is expressed per core |
| VPS2 load average at probe time | **33.71** | latency is unmeasurable until this falls |
| VPS2 GPU | none | SPLADE encode is CPU; the query encode will dominate the per-query bracket |
| VPS2 Postgres | 17.10, pgvector **0.8.2** | `HNSW_MAX_NONZERO` was measured on 0.8.4; reverify |
| `Splade_PP_en_v1` | already in `/root/.cache/huggingface` | no download on the critical path |
| Available venv | `/opt/recall-beam/.venv`, torch 2.13.0, transformers 5.14.1 | belongs to the beam lane; do not share it |
| Available checkout | `/opt/recall-beam-master` at `cfe69e3` | belongs to the beam lane; do not share it |
| `HISTOGRAM_CAPACITY` | 1024 | 100 queries x 3 repeats = 300, comfortably inside the ring |

## Design: the library

One primitive, two entry points, one corpus level guard. All in `recall/sparse.py` beside
`SpladeEncoder`, with `torch` and `transformers` imported inside the functions so a lexical only
install never needs the `sparse` extra.

```
store_sparse_vectors(store, encoder, items, *, batch_size)   the only encode and upsert loop
    ├── Indexer(store, embedder, sparse_encoder=...)         new corpora, one pass
    └── backfill_learned_sparse(store, encoder, ...)         existing corpora, streams iter_chunks()
assert_sparse_coverage(store, profile_id, empty_ids=())      the guard that has to fire
```

### `store_sparse_vectors`

```python
def store_sparse_vectors(
    store: PgVectorStore,
    encoder: SpladeEncoder,
    items: Iterable[tuple[str, str]],
    *,
    batch_size: int = 32,
    progress: Callable[[int], None] | None = None,
) -> SparseIndexResult
```

`progress` receives the running count of rows written, and exists so a caller can print something
during a twenty minute CPU encode.

`items` is `(chunk_id, text)`. Each batch goes to `encoder.encode()` and then to
`store.upsert_sparse(encoder.profile.profile_id, {...})`. The profile id is read off the encoder
rather than passed as a separate argument, so vectors cannot be filed under a name a different
model produced. That mistake produces plausible scores rather than an error, which is what the
profile column exists to prevent.

`SparseIndexResult` carries `written: int` and `empty_ids: list[str]`.

### The empty vector split

`upsert_sparse` refuses an empty weights mapping outright, and it is right to: an all empty run
means a broken encoder, and the table's CHECK requires `nnz > 0`. But a single punctuation only
chunk must not kill a 20,000 chunk index. So the decision is split by level.

At the **row** level the primitive skips an empty vector and records its id, which is what
`scripts/encode_sparse.py` already does. At the **corpus** level `assert_sparse_coverage` compares
`store.sparse_row_count(profile_id)` against `store.count()` and raises on any shortfall, naming
the recorded empty ids as the explanation when that is the cause. Skipping is forced by the store's
contract. Refusing is where the operator's decision belongs.

### `backfill_learned_sparse`

Streams `store.iter_chunks()`, which already uses a server side cursor and excludes the dense
vector, so a corpus larger than memory is fine. Feeds the primitive. Returns the same
`SparseIndexResult`.

**Idempotent, not resumable.** `upsert_sparse` is `ON CONFLICT DO UPDATE`, so re invoking the
backfill is safe and simply re encodes. Skipping ids already present would need a new
`store.sparse_ids(profile_id)` method; at the corpus sizes this serves that buys nothing, so it is
deliberately out of scope and the docstring says so rather than leaving it looking like an
oversight.

### The `Indexer` hook, and why its test is unusual

`Indexer.__init__` gains `sparse_encoder: SpladeEncoder | None = None`. When present, the primitive
runs on each written batch, inside `Indexer._flush` (`index.py:654`), which is where the dense
`store.upsert` actually executes. Both call sites of `_flush` (`index.py:610` and `index.py:617`)
therefore inherit it, and neither can be forgotten independently.

`Indexer` already has a feature of exactly this shape: `shadow: ShadowIndexTarget | None`, an
optional secondary write target driven during indexing. **It failed silently once.** The fix commit
is `b0e74e5`, merged as PR #218, and its subject is "attaching a shadow to an indexed corpus wrote
nothing to the shadow": every active fingerprint matched, so every file was skipped, and the shadow
write lived past the `continue`. The run reported success with a skipped count and an empty shadow.
That defect is repaired on master and is **not** open, contrary to the entry still standing in
`MEMORY.md`, which this document corrects.

The precedent is therefore history rather than a live defect, and it is the sharper argument for it:
a second write hooked into this exact indexing loop has already once read as protection and not
fired.

So the hook's test asserts `store.sparse_row_count(profile_id) == store.count()` after a real
`index_path` against a real Postgres. A **row count, not a call count.** And it is run against the
un-integrated `Indexer` first, to watch it go red. A test written after a change and never shown to
fail is a hypothesis, not a guard. It must also cover the skip path the shadow bug lived on: index
a corpus, *then* attach a `sparse_encoder` and re-index, where every dense fingerprint matches and
every file is a candidate for `continue`.

## Design: the benchmark

### One invocation, both arms

`--candidate-k` already uses `action="append"`. `--sparse-backend` does the same. Both arms then run
inside **one invocation, against one store and one sparse backfill**. This is strictly better than
two runs: the corpus is identical by construction rather than by matching seeds, the CPU encode is
paid once, and both rows land in one `splits.json` where they can be read side by side.

`LegSplit` gains a `sparse_backend: str` field so the rows are distinguishable in the JSON and in
the markdown table.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--sparse-backend` | `["lexical"]` | appendable; `lexical`, `splade`, `both` |
| `--sparse-model` | `recall.sparse.DEFAULT_MODEL` | apache-2.0 `Splade_PP_en_v1` |
| `--sparse-revision` | `None` | see the `_commit_hash` risk below; **set on the real run** |
| `--sparse-top-k` | `HNSW_MAX_NONZERO` | pruning changes the stored vector and the fingerprint |
| `--accept-noncommercial-license` | off | required for the `naver` checkpoints |
| `--max-load-per-core` | `0.30` | quiescence ceiling |
| `--allow-busy-host` | off | overrides the ceiling and stamps the artifact |

The encoder is built lazily and only when a selected backend needs it, matching
`benchmarks/mtrag/run.py`.

### Wiring

After `_throwaway_store` yields, and only when some selected backend wants the learned leg, the run
calls `backfill_learned_sparse` and then `assert_sparse_coverage`. Progress is printed: a silent
twenty minute CPU encode is indistinguishable from a hang.

The `measure()` call at `store_latency_share.py:662` already accepts `sparse_backend` and
`sparse_encoder` and simply never passes them. That is a one line fix.

The docstring paragraph at lines 34 to 42 is rewritten. Shipping a false statement about our own
tool is the thing being closed here, and leaving the paragraph would reintroduce it.

## Design: the host quiescence guard

A new `recall/eval/hostload.py`.

- Read the one minute load average and divide by the core count.
- Sample **before** the timed phase and **after** it. Both go into `_provenance`.
- A **pre run** breach refuses before the corpus build is paid for, matching the refuse up front
  style already at `store_latency_share.py:621`.
- A **post run** breach cannot un measure the run, so it appends to `notes`, which already drives
  the exit code, and stamps the artifact as contended.
- `--allow-busy-host` overrides and stamps.

**The threshold is a judgement call and is labelled as one.** `0.30` load per core leaves roughly
seventy percent of cores free. It is not derived from a measurement, and the spec says so rather
than dressing it up as one.

**The guard cannot fire on Windows.** `os.getloadavg` is Unix only. On Windows the provenance field
records JSON `null` and the run warns, rather than refusing, because refusing would break every
local dev run of this benchmark. This is stated rather than hidden: the artifact that gets published
comes from Linux.

## Two defects fixed in passing

**`drop_table` leaks sidecar rows.** `PgVectorStore.drop_table()` drops the chunk table and cleans
the migration ledger, but does not delete that table's rows from the learned sparse sidecar, which
is keyed `(tenant_id, chunk_table, profile_id, id)`. Under `splade`, every `_throwaway_store` run
would leave a uuid named orphan set behind permanently. `drop_table` gains the matching DELETE.

**The benchmark docstring asserts a limitation that will no longer hold.** Rewritten as part of the
change that removes it.

## Error handling

| Condition | Behaviour |
|---|---|
| `torch` / `transformers` absent | ImportError from inside the function, naming the `sparse` extra |
| `profile.top_k > HNSW_MAX_NONZERO` | already raised by `SpladeEncoder.__init__` |
| a chunk encodes to an empty vector | skipped, id recorded, surfaced by the coverage assertion |
| a chunk exceeds the nonzero budget | already raised by `upsert_sparse` before any INSERT |
| sidecar coverage short of chunk count | `assert_sparse_coverage` raises, naming both counts |
| learned leg fires below 5% of queries | already raised by `measure()`; the issue #81 alarm |
| host load above the ceiling, pre run | refuse before indexing |
| host load above the ceiling, post run | `notes` entry, artifact stamped, exit 1 |

## Testing

Every item below is a row count or an observed refusal. None is a call count.

1. `index_path` with `sparse_encoder` leaves `sparse_row_count == store.count()`. **Run against the
   un-integrated `Indexer` first and shown to go red.**
2. `backfill_learned_sparse` over an already indexed store reaches the same equality.
3. `assert_sparse_coverage` raises on a deliberately partial sidecar.
4. The quiescence guard is fed a load reading high enough to refuse, and is shown refusing.
5. A splade arm run on a tiny corpus with the small `BertForMaskedLM` that
   `tests/test_learned_sparse_retriever.py` already uses: the split has a non null
   `learned_sparse_fire_rate`, a null `sparse_fire_rate`, and zero recorded samples on the lexical
   leg.
6. `drop_table` leaves no sidecar rows for the dropped table.
7. The new refusals are registered in `scripts/ablate_store_latency_guards.py`. That file's own
   docstring records what a stale anchor there costs: the rule prints SKIP and silently stops being
   exercised, which is the same class of defect the rule exists to catch.

## The VPS2 run

**Isolation.** A fresh checkout at `/opt/recall-splade-cli` with its own venv, not
`/opt/recall-beam-master` and not `/opt/recall-beam/.venv`. Those belong to the beam lane, and a
`pip install` from either side changes both. A dedicated `recall_storelat` database, not the beam
lane's 3.4 GB `recall_splade`.

**Verified on the host before the long run, not assumed.**

1. pgvector 0.8.2 enforces the same 1000 nonzero HNSW ceiling that `HNSW_MAX_NONZERO` recorded from
   0.8.4. A single over budget INSERT probe answers this.
2. `transformers` 5.14.1 still exposes `model.config._commit_hash`. If it does not,
   `artifact_digest` degrades silently to `"unpinned"` and the run is not reproducible. Either way
   the run pins `--sparse-revision` explicitly, so the digest is the caller's pin.

**The run.** `--filler 20000 --queries 100 --repeats 3`, both candidate_k values,
`--sparse-backend lexical --sparse-backend splade`, under `nohup` so a dropped SSH does not kill it.

**No `--rerank` on the first run.** On a CPU box the cross encoder dominates the wall clock. Its
absence inflates the store's share, which is a bias **toward** "porting the store is worth it", and
the artifact will say so. A rerank run follows separately if the box stays quiet.

## What this does not do

- It does not touch `benchmarks/mtrag/run.py`, which already builds a real encoder.
- It does not replace `scripts/encode_sparse.py` or `scripts/load_sparse.py`. The DSN free split is
  correct for rented hardware and stays.
- It does not make the backfill resumable. See the note under `backfill_learned_sparse`.
- It does not touch the `shadow` dual write. That mechanism is named here only as precedent for how
  the new hook is tested; its defect was already repaired by `b0e74e5` (PR #218).
- It does not measure a real corpus. The synthetic corpus measures the regime where the store is
  cheap by construction, exactly as the existing module docstring warns, and commit `9a5165b`
  remains the number that frames any result from this file.

## Addendum, 2026-08-07: local GPU selection

Added after the design above was approved, at the user's request: an option to use a local GPU
when there is one good enough. Implemented as Task 8b of the plan.

**Measured, not assumed.** The local box has an **NVIDIA GeForce GTX 1070 Ti, 8192 MiB**, driver
582.66. `Splade_PP_en_v1` is BERT-base, so fp32 inference at batch 32 needs well under 2 GB and
VRAM is not the constraint. The installed torch is **`2.12.1+cpu`** with `torch.version.cuda`
reporting `None`, so CUDA is unreachable because of the wheel rather than the hardware. The card is
Pascal, compute capability **6.1**, and recent PyTorch CUDA wheels have been dropping older
architectures; whether the current one still ships `sm_61` is **read at runtime from
`torch.cuda.get_arch_list()`** and is asserted nowhere.

**`SpladeEncoder.from_pretrained` already takes `device` and already defaults to CUDA when
available.** What was missing is a way to ask for it and a way to be told no.

`--sparse-device {auto,cpu,cuda}`. `auto` keeps today's behaviour and additionally prints the
device it chose. `cuda` **refuses** rather than falling back, naming which of four checks failed:
a CPU-only wheel, no visible device, an architecture absent from the wheel's arch list, or
insufficient free VRAM. That choice follows the note already in `recall/sparse.py` about the
rented-GPU case, where a silent fallback yields correct vectors roughly a hundred times more
slowly and "the only symptom is the bill". Locally there is not even a bill.

The refusal logic is a pure function over those four facts, so every branch is shown firing on a
box with no CUDA build. The resolved device is stamped into `_provenance`, because
`learned_sparse_encode_ms_mean` is a transformer forward pass and its CPU and GPU values measure
different things.

VPS2 has no GPU, so Task 9 is unaffected, and it passes `--sparse-device cpu` explicitly so the
artifact records a chosen device rather than an incidental one.

## Open risks

| Risk | Handling |
|---|---|
| VPS2 never gets quiet enough to measure | the guard refuses rather than publishing a contention measurement; the run waits |
| transformers 5.x changes the SPLADE load path | verified on the host before the long run; `--sparse-revision` pins regardless |
| pgvector 0.8.2 differs from the 0.8.4 measurement | probed on the host before the long run |
| CPU encode of 20,000 chunks is slower than expected | `scripts/encode_sparse.py` already reports a measured rate; take the rate from a `--limit` probe before committing to the full run |
