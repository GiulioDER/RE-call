# Your first calibrated corpus

`docs/CALIBRATION.md` is the reference. This is the walkthrough: everything between "I have a
folder of markdown" and "the server answers and says `trusted`", with the traps named at the point
you hit them.

Every error string below is quoted verbatim from a real run, so you can search for the one you are
looking at. The measurements come from calibrating two corpora on 2026-08-18: 1,080 memo files
(8,671 chunks, bge-large) and 573 Python files (6,487 chunks, voyage-code-3).

## The short version

```bash
recall index ./corpus --project my-project      # 1. index
recall --tenant t manifest create ...           # 2. manifest
recall --tenant t generation build ...          # 3. build, re-embeds everything
recall --tenant t generation validate GEN       # 4. validate
recall --tenant t generation promote GEN --unsafe-development-promotion
recall --tenant t calibration calibrate --generation GEN --queries q.json --publish
```

Steps 2 to 6 are not optional if you want a trusted answer, and step 3 costs as much as step 1 did.
The rest of this page explains why.

## Indexing alone gives a working search that says `degraded`

This is the first surprise, and it is not a bug. After `recall index` the search works and returns
good hits, but every answer carries:

```
[DEGRADED:INDEX_NOT_READY]
```

The rule is in `recall/readiness.py`: a tenant with no active generation is `INDEX_NOT_READY`
**regardless of its calibration state**, because telling you to recalibrate against a generation
that does not exist is useless advice. No configuration clears this. You need a generation.

⚠️ **`recall calibrate` writes a file that the serving path never reads back.** It is easy to run
it, get a `calibration.json`, and assume you are calibrated. `recall/cli.py` skips loading it
deliberately: a legacy JSON artifact has "no tenant, generation, pipeline, corpus, or labelled
query-set binding", and passing one would set `calibration_status="legacy_unbound"`, which strict
policy maps to `CALIBRATION_UNCERTIFIED` and refuses the search. Only
`calibration calibrate --generation ... --publish` produces something that serves.

## Building a generation re-embeds the whole corpus

There is no path that adopts the vectors you already have. `_reuse_source` copies chunks only from
rows already in `recall_chunks_v1` that join a generation with a matching `pipeline_fingerprint`,
and a freshly indexed corpus lives in the legacy `chunks` table with no generation at all.

Budget accordingly. Measured on the same 12 core box:

| corpus | embedder | wall clock | peak RSS |
|---|---|---|---|
| 1,080 files, 8,671 chunks | bge-large, local CPU | about 10 hours | 11.7 GB |
| 573 files, 6,487 chunks | voyage-code-3, hosted API | 7 minutes | 807 MB |

A hosted model was roughly 35 times faster here and used almost no memory, because the embedding
happens on the provider's hardware. A local model is free per call and costs you CPU and RAM.

⛔ **Do not hand-write rows into `recall_chunks_v1` to skip the re-embed.** The schema gap is only
five columns and the vectors already exist, so it is tempting. It also fabricates the provenance a
generation exists to establish, which is the same false immutability `recall/lineage.py` refuses
for `file://` objects.

## Four refusals while building the manifest

All four are the system working. Each is a one line fix.

**The tenant belongs on the global flag, not the subcommand.**

```
GenerationError: manifest tenant 'default' does not match authenticated tenant 'memory'
```

Write `recall --tenant memory manifest create ...`, not `recall manifest create --tenant memory`.

**A `file://` object's `version_id` must be the FULL content digest.**

```
LineageError: a file:// manifest object's version_id must be its content digest. A local file has
no version other than its contents, and any other value would name an immutability guarantee the
filesystem does not provide.
```

A truncated digest is refused. Set `version_id` equal to `sha256`.

**Every object needs a `media_type`.**

```
LineageError: manifest object media_type must be non-empty
```

**`RECALL_LOCAL_ALLOWLIST` is required for `file://` manifests**, and deliberately has no default:
without it a manifest could name any file on the machine. Set it to the corpus root.

## One generation, one chunker

`generation build --chunker` takes `text` or `code` and applies it to every object in the manifest.
If your corpus mixes prose and source, either build two generations or accept that one of them is
chunked the wrong way. Splitting is usually right: a Python file chunked as prose loses its
function boundaries, and markdown chunked as code loses its headings.

The Python corpus above was built from `.py` files only for exactly this reason, which is why it
has 573 objects where the mixed legacy tenant had 615.

## Promotion and serving want different modes today

A known rough edge, with a design proposal in `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`.

```
UnsafePromotion: development promotion requires unsafe_development=True
```

`promote()` refuses outright in production ("generation promotion is unavailable in production
until certification gates land") and requires an explicit flag in development. Meanwhile the MCP
server only builds a generation-aware store when `RECALL_ENV=production`. So today a generation is
promoted in one mode and served in the other. The flag name is alarming, but it marks an incomplete
feature rather than a destructive action.

## The labelled query set

`--queries` takes a JSON array of `{query, answerable, relevant_ids}`. Two things decide whether
your calibration certifies.

**At least 20 answerable AND 20 unanswerable queries.**

```
abstention threshold 0.662 is NOT certified, too few samples (answerable=19; need >= 20 of each):
a q05/q95 boundary is not identifiable from a handful of points and collapses onto the extremes
```

Write more unanswerable than feels necessary. The built-in eval set is 20 answerable to 26
unanswerable. A set that is mostly answerable never shows the threshold a case where it should
abstain, so it calibrates a floor that never fires.

**Unanswerable queries must be plausible and genuinely absent.** "What is the capital of France"
measures nothing about where your boundary sits. Ask the kind of question a user would ask that
your corpus happens not to answer, then check each one against the built index. An unanswerable
label the corpus can in fact answer is the single most damaging entry in the file, because it
teaches the threshold to abstain on a good hit.

`relevant_ids` are `"<file>:<ordinal>"`, and the ordinals only exist once the corpus is chunked, so
resolve them from the index rather than typing them. The calibration itself never reads them; they
matter only if you also measure recall or nDCG.

## Reading the number you get

**A threshold belongs to one model and does not transfer between models.** Measured:

| corpus | embedder | threshold |
|---|---|---|
| memory memos | bge-large | 0.7100 |
| Python source | voyage-code-3 | 0.6620 |
| prose docs | voyage-4 | top hits at 0.269 to 0.413 |

`voyage-4` and `voyage-code-3` are both Voyage models and both 1024 wide, and they sit on entirely
different scales. Predicting the code threshold by extrapolating from the docs one was falsified by
more than 0.3. Never carry a threshold across an embedder change. Recalibrate.

**Expect the two distributions to overlap.** In four corpora out of four the worst answerable query
scored BELOW the best unanswerable one: memory -0.048, Python source -0.007 and -0.032, a second
code corpus -0.097. No single cosine cleanly separates them, so a calibrated threshold is a
least-bad cut and not a boundary.

⚠️ **Separability hides this.** Those same corpora reported 0.97 to 0.99, because the distributions
are almost perfectly ORDERED while still overlapping. Read the separation
(`min answerable` minus `max unanswerable`) alongside it, or you will report a clean win where the
measurement shows a trade.

## Two hazards worth knowing before trusting a number

**One embedding model per tenant.** Two models of the same width produce no error when mixed: the
insert succeeds and the cosines are quietly meaningless. During this work a probe accidentally
queried a `voyage-code-3` corpus with `bge-large`, both 1024 wide, and returned similarities of
0.02 to 0.10 that looked like a broken corpus rather than a misconfiguration. Verify with
`vector_dims(embedding)` and the profile id together.

**`query_dense(k=1)` can return nothing, or a wrong top score.** The HNSW index is built over the
whole table while `tenant_id` is applied as a post-filter, so for a tenant that is a fraction of the
table the graph walk can return other tenants' candidates and filter them all away. Measured on one
tenant with a single query, same vector each time:

```
k=  1 ->   0 hits
k= 20 ->   1 hit,  top=0.5277   <- WRONG, and silently so
k=100 -> 100 hits, top=0.5743   <- the true nearest neighbour
```

It varies per tenant and per query, so check rather than assume. Retrieve at `k>=200` when you need
a true top-1.

## Hosted models cannot be verified

A local model gets a real identity: `recall/wizard/identity.py` digests the downloaded artifact and
the generation comes back `verified: true`. A hosted model has no weights on disk to hash and no
revision to pin, and the provider can change what sits behind a model name without telling you. A
generation built on one therefore needs `--unverified-development` and reports `verified: false`.

That is an honest limitation rather than a workaround, and the enterprise pinning gate refuses such
a profile on purpose. If you need a verified generation, use a local model.

## When it works

```
abstained    : False
failure_code : None
trust_state  : trusted
```

`recall_stats` reports the chunk count, and `calibration_status` reads `certified` with the
generation id beside it. If you see `CALIBRATION_MISSING` where you previously saw
`INDEX_NOT_READY`, you have moved one rung up the ladder: the generation is being seen, and only
the published calibration is missing.
