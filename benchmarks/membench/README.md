# mem-bench adapters

RE-call's submitter-side adapters for [mem-bench](https://github.com/GiulioDER/mem-bench), the
judge-free memory benchmark. These produce the RE-call rows on its **isolation** (axis 2) and
**temporal** (axis 3) boards.

## Why they live here and not in mem-bench

mem-bench's `membench/` package is **stdlib-only**, and its whole claim — that any figure on the
board is recomputable by a hostile party for $0, with no model and no API key in the verification
path — depends on it staying that way. An adapter importing `recall` would break that on its first
line.

mem-bench's runners take `--system module:Factory`. That seam exists precisely so a vendor's adapter
lives in the vendor's repo, versioned alongside the system it wraps. This is the vendor's repo.

They were homeless until 2026-07-31: they existed only on VPS2 and in a session scratchpad, and when
VPS2's root volume failed one of the two copies went with it. Code that is the only way to reproduce
a published figure belongs under version control.

## Running them

Both adapters need mem-bench importable and a Postgres with `pgvector`.

```bash
pip install -e .                      # this repo, for `recall` and `benchmarks`
export PYTHONPATH=/path/to/mem-bench  # for `membench`
export MEMBENCH_DSN=postgresql:///membench_iso
```

Isolation (axis 2):

```bash
python -m membench.axes.isolation.run \
    --system benchmarks.membench.recall_isolation:RecallIsolation \
    --out iso.jsonl --submitter github:you --system-name RE-call \
    --run-started 2026-08-01T09:00:00Z \
    --config '{"embedder": "BAAI/bge-small-en-v1.5", "dim": 384, "reranker": null}'
```

Temporal (axis 3) is the same with `membench.axes.temporal.run`,
`benchmarks.membench.recall_temporal:RecallTemporal`, and a `membench_tmp` DSN.

**Every run needs an EMPTY store.** Drop the table before each one:

```bash
psql -qd membench_iso -c "DROP TABLE IF EXISTS iso_chunks"
```

Not tidiness — correctness. All three adapters create a fresh `mkdtemp` work directory per run, and
`Indexer.index_path` prunes and replaces by the resolved absolute `source` path, so a second run
into the same table INSERTS a whole new copy of the corpus rather than replacing it. Every query
from then on searches a corpus containing its predecessors' documents.

Measured 2026-08-02, running the full axis matrix twice: a shared isolation store ended with 22
distinct work-dir roots and 1424 rows against a fresh store's 15 and 1024, and `own_tenant_recall`
on an *identical* config read 0.9917 against an empty store and 0.8417 against a shared one. Six
artifacts came back VERIFY OK and none of their figures measured the corpus their manifest
describes — mem-bench cannot catch this, because the artifact is internally consistent, all six
checks pass, and the delivery layer is satisfied: the run really did retrieve what it says it
retrieved, out of a corpus nobody described. `scripts/run_campaign.sh` in mem-bench does the drop
for you.

**Do not pass `--system-version`.** Each adapter exposes `system_version`, read live from
`recall.__version__`, and mem-bench aborts the run if a flag contradicts it. That guard exists
because the eight artifacts currently on the board all declare `0.6.0` with nothing to confirm it —
they render `(declared)` and will be replaced by re-runs.

## Configuration

| variable | default | meaning |
|---|---|---|
| `MEMBENCH_DSN` | `postgresql:///membench_iso` (iso) · `postgresql:///membench_tmp` (temporal) | Postgres + pgvector |
| `MEMBENCH_TABLE` | `iso_chunks` · `tmp_chunks` | chunk table |
| `MEMBENCH_EMBEDDER` | fastembed default | `voyage:<model>` for Voyage, else a fastembed name |
| `MEMBENCH_RERANKER` | none | `voyage:<model>`; anything else ignored |
| `MEMBENCH_K` | `5` | hits returned |
| `MEMBENCH_CANDIDATE_K` | `20` | pool reranked down to `k` |

The tuned arm is **the same code path** with different environment values, not a subclass — a second
class would be a second thing that could differ between arms.

## Two things these adapters do not do

- **`known_as_of` is not wired into the temporal adapter.** RE-call 0.7.0 added bi-temporal
  retrieval and it is tempting to read axis 3's low covering rate as measuring a pre-fix version.
  It does not: `known_as_of` is *transaction* time, axis 3 scores *valid* time. See the module
  docstring.
- **Neither adapter decides whether it was right.** They report `cited_ids`; mem-bench owns the
  ownership map and the intervals, and derives leakage and coverage itself. A run that scored itself
  would be marking its own exam.
