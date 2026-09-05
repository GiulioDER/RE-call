# Operating Modes

RE-call is deliberately configurable because memory deployments differ by legal boundary, hardware,
latency target, quality bar, and cost model. The default path keeps retrieval local. Hosted
embedding, reranking, and stricter serving profiles are opt-in choices after measurement.

| Mode | Use when | Trust policy | Egress | Typical configuration |
|---|---|---|---|---|
| Local demo | You want to try the product surface on the bundled corpus. | Development, with degraded confidence marked clearly. | None with FastEmbed. | `RECALL_TRUST_MODE=development`, `RECALL_EMBEDDER=fastembed`. |
| Local production | You need private memory over a controlled corpus. | Strict, with a generation-bound calibration. | None with FastEmbed and local artifacts. | Build and calibrate with `RECALL_ENV` unset, **then** serve with `RECALL_ENV=production` — see [the two switches](#recall_env-and-recall_trust_mode-are-different-switches). Plus `RECALL_RETRIEVAL_PROFILE=fast` and split serving/migration DSNs. |
| Quality production | Answer quality is worth higher latency or local model cost. | Strict, with a calibration fitted for that embedder and corpus. | None with local artifacts. Possible egress if a hosted embedder is selected. | `RECALL_RETRIEVAL_PROFILE=quality`, pinned reranker path and digest. |
| Hosted embedder | Legal and privacy review permits external embedding calls. | Strict or development, depending on whether calibration has been promoted. | Query and corpus text may leave the environment. | Select the hosted embedder and document the approved egress path. |
| Evaluation | You are reproducing published evidence or comparing corpus options. | Explicit benchmark policy or development mode. | Depends on the selected embedder. | Use `benchmarks/`, `results/`, and a run-specific configuration record. |

Selection rule: start local, calibrate on the real corpus, measure the failure modes, then opt into
hosted embedding or reranking only when the measured gain pays for egress, latency, and operating
cost.

## `RECALL_ENV` and `RECALL_TRUST_MODE` are different switches

They are easy to conflate, they both accept the word `development`, and conflating them is the
single most common way to get stuck. Read this before setting either.

| Variable | Values | Controls |
|---|---|---|
| `RECALL_ENV` | `development` (default) / `production` | Where content may be **ingested** from, and whether **generations** are used at all |
| `RECALL_TRUST_MODE` | anything not `development` is strict (default) / `development` | Whether a search may **answer** without a certified, generation-bound calibration |

What `RECALL_ENV=production` changes, in both directions:

- ⛔ **`recall index` refuses.** `local filesystem indexing is development-only; build from an
  immutable S3 manifest in production`.
- ⛔ **`recall generation build` refuses any manifest that is not `s3://`.** `production generation
  builds require a versioned S3 manifest`.
- ✅ **Generations are read.** Under `development` the server uses the legacy store, which knows
  nothing about generations, so every search resolves `calibration_status=missing` and **strict
  trust refuses**.

Those refusals are the product working. A generation binds chunks to immutable objects, and a local
file has no version other than its own bytes.

### The combination that traps people, and the way out

Read the table above and you will conclude that a private corpus on your own disk cannot be served
under strict trust: indexing needs `development`, generations need `production`, and you cannot set
both. Many readers get here, reach for `RECALL_TRUST_MODE=development`, and ship a deployment whose
every answer is stamped `degraded` — which is the one thing this product exists to avoid.

**That conclusion is wrong, and the fix is not a trust-mode override.** Build-time and serve-time
environments are *meant* to differ. `GenerationManager.certification_required` reads "the SERVING
environment, not the build one" — the split is deliberate, not a loophole.

Every command below was run end to end against a real PostgreSQL + pgvector database on
2026-08-26, in this order, on a 22-file local corpus. The flags that are easy to omit are the
reason it is written out in full rather than summarised.

```bash
export RECALL_LOCAL_ALLOWLIST=/path/to/corpus   # a manifest may name no file outside this root

# 1. MANIFEST. `objects.json` is a JSON array; each entry needs uri, sha256, size, version_id
#    (= the sha256, for file:// there is no other version) and media_type.
recall --tenant acme manifest create --corpus-version v1 \
    --objects objects.json --output manifest.json

# 2. BUILD. `--unverified-development` is REQUIRED for a local corpus: a file on disk has no
#    immutable revision or artifact digest, and without the flag this fails with
#    `embedder identity needs an immutable revision or artifact digest`.
recall --tenant acme generation build manifest.json \
    --project acme-corpus --unverified-development

# 3. VALIDATE. The build leaves the generation in `validating`; calibration refuses it in that
#    state with `expected ready, active, or retired`.
recall --tenant acme generation validate gen_...

# 4. CALIBRATE and PUBLISH.
recall --tenant acme calibration calibrate --generation gen_... \
    --queries queries.json --publish

# 5. PROMOTE, with RECALL_ENV=production. See the note below on why not development.
RECALL_ENV=production recall --tenant acme generation promote gen_...

# 6. SERVE with RECALL_ENV=production. Leave RECALL_TRUST_MODE unset: strict is the default.
RECALL_ENV=production recall --tenant acme search "your question"
```

`RECALL_TRUST_MODE` is never set anywhere in that sequence. You get strict trust over a corpus that
never left your disk. Verified by contrast on the same database and query: step 6 answers `[ok]`
with per-hit verdicts, while the identical search with `RECALL_ENV` unset is refused, because
without production there is no generation for the trust gate to resolve.

⚠️ **The query set takes `{query, answerable}` and nothing else.** Extra keys are rejected —
`query set entry N requires a non-empty query and boolean answerable`. The bundled
`recall/eval/queries.json` mixes in supersession-test entries carrying a `trust` key, so filter
those out before passing it here.

⚠️ **Promote under `production`, not under development.** Promotion consults the *serving*
environment (`GenerationManager.certification_required` reads "the SERVING environment, not the
build one"). Under production it checks that your calibration is certified, which it is, and
promotes cleanly. Under development it instead demands `--unsafe-development-promotion`, because
the generation is unverified — so the development route makes you pass a flag saying "unsafe" to
promote something you have in fact just certified. Promoting under production is both cleaner and
the check you actually want.

⚠️ **Calibrate and publish before you promote, not after.** If you promote first and calibrate
later, serving refuses what you promoted with `CALIBRATION_MISSING` or `CALIBRATION_UNCERTIFIED`.
The generation is fine; it simply has no certified calibration bound to it. Calibrate, then promote
again.

⚠️ **The vector column has one fixed width, set when the schema was applied.** Building with an
embedder of a different dimension fails with `pipeline dimension 64 does not match
recall_chunks_v1 vector(384)`. Pick the embedder before the schema, or use a separate database.

⚠️ **Re-indexing later means repeating steps 1 to 3**, because adding files means a new generation.
`recall index` will keep refusing under production, and that refusal is not telling you to change
`RECALL_TRUST_MODE`. If you want incremental local indexing without generations, that is the
`Local demo` row of the table above, and its answers are honestly marked `degraded`.

⛔ **`RECALL_TRUST_MODE=development` is for looking at results while you finish setting up.** It
retrieves, marks every result degraded, and claims no abstention decision. It is not a way to run a
deployment, and it is never the answer to an ingestion refusal.
