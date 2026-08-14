# Enterprise retrieval and evidence deployment

This implementation keeps the existing chunk table and retrieval flow. It adds immutable embedding identities, fixed process retrieval profiles, deterministic contextual passage text, a generator neutral evidence boundary, and PostgreSQL generation routing.

## Embedding profile registry

`recall/embedding_registry.py` is the single place a profile is defined. It owns the profile identifier, the model, the artifact digest, the dimension, the query and passage encoder modes, the normalization, the instruction version, the chunker version and the context version. Nothing else may hold a second copy of that vocabulary.

Three properties follow from that and are enforced by tests:

* `context_version` is derived from `context_mode`, never declared next to it. `Indexer` refuses an embedder whose profile does not spell its context exactly `raw-v1` or `context-<mode>-<policy version>`, so the derivation in the registry is that contract written once.

* `query_mode` and `passage_mode` name the encoder that is actually called. A profile declaring `query_embed` gets `TextEmbedding.query_embed`, and a backend without that encoder refuses to start rather than falling back to the symmetric one.

  ⚠️ **Naming a distinct encoder is not the same as getting distinct vectors, and on this deployment it does not.** Under fastembed 0.8.0, `BAAI/bge-small-en-v1.5` returns byte-identical vectors from `embed`, `query_embed` and `passage_embed`: the model resolves to `OnnxTextEmbedding`, which overrides neither method, so both come from `TextEmbeddingBase` and `yield from self.embed(...)` with no instruction. `bge-small-symmetric-v1` and `bge-small-asymmetric-v1` share every other identity INPUT to a stored vector (model, dimension, context mode, normalization, instruction version, chunker version, backend) and one provisioned artifact tree, so they cannot produce different vectors here, and **a paired comparison of the two is a null by construction rather than an experiment**. Their FINGERPRINTS still differ, because `EmbeddingProfile.fingerprint` covers `profile_id` and both mode fields, so the two index into separate physical tables and a comparison of them would carry tie-break noise rather than exact zeros. Reproduce with `benchmarks/check_profile_encoder_distinctness.py` on a host with the tree provisioned (elsewhere pass `--cache-dir`); the delegation half needs no weights at all and is checkable from library source anywhere fastembed is installed. The dispatch still has to be correct, because `qwen3-embedding-0.6b-384-v1` does use a distinct instruction-prefixed query encoder and a future backend could add one without changing a profile identifier. Giving BGE a query instruction would be a new experiment and needs registering.

* The declared dimension is checked against the artifact at startup. An artifact that embeds at a different width is not that profile, and the process refuses rather than writing vectors no other process can interpret.

Registered identifiers: `bge-small-symmetric-v1`, `bge-small-asymmetric-v1`, `bge-small-context-document-v1`, `bge-small-context-section-v1`, `bge-small-context-neighbor-v1`, and the rejected `qwen3-embedding-0.6b-384-v1`.

Passage encoding is used for indexing and for dimension discovery. Query encoding is used for retrieval, calibration, semantic lint, evaluation and the timing wrappers. An embedder that implements only `embed` keeps working: both helpers fall back to it, and its cached vectors are keyed under a legacy descriptor that no verified profile can collide with.

## Embedding cache identity

The embedding cache is keyed by the complete immutable profile identity, not by the profile identifier. `EmbeddingProfile.fingerprint` covers every identity field, including `artifact_digest`, `context_version` and the pinned inference library version, and the cache key adds the purpose (`query`, `passage` or `legacy`), the dimension and the text.

The identifier alone is not an identity. A re-provisioned artifact and a context mode change both move the stored vectors while the identifier stays fixed, and a key that misses either serves a vector computed from different weights or from different text. A cache hit is a plausible vector of the right width, so nothing downstream can detect it. Cross identity reuse therefore fails closed: the key misses and the text is embedded again.

Changing the fingerprint encoding invalidates every cache in existence at once. If that is ever wanted, bump the domain tag inside `EmbeddingProfile.fingerprint` so the change is legible.

## Deterministic context modes

Three context modes build the text handed to the embedder. They are declared by the profile (`bge-small-context-document-v1`, `bge-small-context-section-v1`, `bge-small-context-neighbor-v1`) and implemented in `recall/context.py`. `mode="none"` is the symmetric baseline and embeds the chunk as stored.

**Embedding text is built separately from stored text, and the stored text never moves.** `chunk_text()` and the chunk row are untouched by every mode. The rendered passage is assembled from `StructuredChunk`, which carries source offsets and the heading hierarchy alongside the chunk's own bytes.

| Rule | Behaviour |
|---|---|
| Title precedence | frontmatter `title`, then the first H1, then the root-relative basename. The frontmatter key must be **top level**; an indented `title:` belongs to a sub-object and is skipped. Whether there is a block at all is `frontmatter_span`'s call: a leading `---` followed by prose is markdown's thematic break, not an open fence, so a `title:` line sitting in that prose is **not** the document's title. The basename is taken from the whole path, before any cap |
| Paths | root-relative only. An absolute path, a drive letter, a UNC path or any `..` segment is **refused**, in every mode including `none`. `root_relative_source` validates and **does not truncate**: the cap belongs to the rendered field, because a cap applied inside the guard runs after its own checks and can reintroduce what they refused. The refusal names the rule, never the path, since the value it fires on is an absolute host path |
| Control characters | stripped from every structural field (title, source, section hierarchy). The chunk is content and is left exactly as stored |
| Caps | title 256 characters, source 256, section hierarchy 512 |
| Neighbour context | at most 200 characters from each adjacent chunk: the tail of the preceding one, the head of the following one. Folded to one line **before** the 200 is counted, so the neighbour budget is in the same unit as the other caps and an adjacent chunk cannot put a second `source:` line into this chunk's passage. Folding also collapses whitespace runs, so a neighbour excerpt is normalised **more** than the other structural fields, which keep theirs. None is invented at a document's first or last chunk |
| Degradation under a token limit | drop neighbour context first, shorten then drop section detail second, drop title detail last. **The complete current chunk is preserved at every rung**; it is never shortened to make room, and the last resort is the bare chunk |
| Recorded identity | the mode and the policy version are written into each chunk's metadata (`context_mode`, `context_version`) and into the profile identity, where `context_version` is derived from the mode and is part of the cache fingerprint |

**The load-bearing invariant: raw chunk content and raw content hashes are byte-identical across generations and across all three modes.** A context mode changes what is embedded and nothing else, so a cutover between generations built under different modes changes how the corpus is retrieved, never what it says. `tests/test_context_modes.py` asserts it over five corpus shapes (with frontmatter, without, no headings, nested headings, and across chunker boundaries) and `tests/test_context_modes_index.py` asserts it again against stored PostgreSQL rows, including a real dual write with the two generations on different modes.

The one field that deliberately **does** change with the mode is `index_fingerprint`, the value the indexer compares to decide whether a file needs re-indexing. If it did not move, switching a generation's context mode would skip every unchanged file and leave vectors built under the old mode in place.

The rendered form is one `field: value` per line. It is text to EMBED and never text to parse: the chunk itself is interpolated verbatim, because rule 5 preserves it, so a document containing a line that looks like a field will render one. Structural fields cannot forge a line (control characters are stripped) and neither can a neighbour excerpt (it is folded), but do not write a parser against this format.

**The 256-character cap on the rendered `source:` field applies to a path the guard has already accepted, so the FIELD can still end in a truncated `..`.** `root_relative_source` refuses traversal and its return value carries none; the cap is applied afterwards, where the field is built, and truncating any path at a fixed length can produce that shape. It is inert — the field is embedding text and is never resolved — but a future consumer of it must not read "traversal is refused" as a property of the rendered string. The cap also keeps the HEAD of a long path, so two paths sharing 256 characters render identically; keeping the tail would identify a document better and is a deliberate open decision, not an oversight.

Which mode retrieves best is not decided here, and no measurement in this repository claims it.

## Security boundary

All model artifacts must exist locally before startup. An explicit embedding profile verifies the configured artifact tree against its SHA256 digest and requests local only loading. The artifact is verified before the backend library is even imported, so a missing or tampered tree fails the same way whether or not the optional extra is installed. The quality retrieval profile also requires a local reranker path and digest. Production should block outbound network access at the workload boundary.

Runtime model downloads are prohibited. Startup is proven to complete with every socket entry point blocked, and to refuse when the artifact is missing or its checksum does not match.

Tenant routes never accept a physical table from a client. The runtime resolves table names only from validated control plane rows. Chunk tables, tenant routes, and migration events use row level security. The runtime database role must be neither superuser nor `BYPASSRLS`.

## Operator runbook

The whole sequence, in order, with the credential each step takes, what it refuses, and how to
verify that it did what it says. Every command exits non-zero on failure, so each step is a gate:
do not proceed past a red one.

⚠️ **A generation table can exist, fully indexed, with no per-table migration ledger rows, and
`readiness` reports `SchemaTooOld` for it.** That is the state both generation tables on the
reference deployment were found in: table present, every index valid, RLS forced, and not one row in
`recall_schema_migrations` for either of them. **The route that produced it is not known.**
`create-generation` on current code does write those rows (it calls `PgVectorStore.ensure_schema()`,
which is `apply_migrations(table=…)`), so this is not a property of that command, and the tables in
question were provisioned by an older build. Run step 2 for every chunk table, including each
generation table, as a cheap precaution: it is idempotent, and the failure it prevents is invisible
until a readiness check.

| # | Step | Credential | Exits non-zero when |
|---|---|---|---|
| 0 | Preconditions | operator | see below; these are checks, not commands |
| 1 | `recall-enterprise migrate` | `RECALL_MIGRATION_DSN` | control-plane DDL fails |
| 2 | `recall schema apply` per chunk table | `RECALL_MIGRATION_DSN` | a migration fails, drifted, or is unknown |
| 3 | `recall-enterprise create-generation` | `RECALL_MIGRATION_DSN` | pgvector absent, or table DDL fails |
| 4 | index the shadow corpus | serving | (application step) |
| 5 | `recall-enterprise mark-ready` | `RECALL_MIGRATION_DSN` | the generation id is unknown. ⚠️ **It does NOT check the counts** |
| 6 | `recall-enterprise set-route --shadow-generation` | `RECALL_MIGRATION_DSN` | the generation is not servable |
| 7 | `recall-enterprise replay` | `RECALL_SERVING_DSN` (⚠️ **write path**) | anything is still pending afterwards |
| 8 | `recall-enterprise parity` | `RECALL_SERVING_DSN` | sources, hashes or counts disagree; an index is invalid; RLS is not forced; **both generations are empty** |
| 9 | `recall-enterprise readiness` | `RECALL_SERVING_DSN` | any startup check fails. ⚠️ It evaluates the **ACTIVE** generation, not the shadow |
| 10 | `recall-enterprise cutover` | `RECALL_MIGRATION_DSN` | an event is pending, or the shadow is not ready |
| 11 | `recall-enterprise retire` | `RECALL_MIGRATION_DSN` | the named tenant still routes at that generation, in **either** slot |
| R | rollback: `recall-enterprise set-route` | `RECALL_MIGRATION_DSN` | the previous generation is not servable |

⚠️ **"Takes the serving credential" is a credential statement, not a read/write one.** `replay` is a
**write path**: it calls `replace_sources` into both the active and the shadow generation and
updates the outbox. Do not run it speculatively against a production tenant because a table above
groups it with the read-only commands.

### 0. Preconditions

Check these before step 1. **Three have been actual failures on the reference deployment**: the
missing grant, the unrecorded licences, and the unblocked egress boundary. The rest passed there and
are cheap to repeat. ⚠️ The 2.2x headroom figure is a policy rule of thumb, not a measurement:
nothing in this repository derives it.

* **Roles.** A migration role and a runtime role, both `NOINHERIT` and neither `SUPERUSER` nor
  `BYPASSRLS`; the runtime role must not own the chunk tables and must not hold `CREATE` on the
  schema. [MIGRATIONS.md](MIGRATIONS.md) is the authoritative recipe; do not restate a subset of it
  here. Attribute columns alone are **not** a sufficient check: a role that is merely a *member* of
  a `BYPASSRLS` role can `SET ROLE` to it, and `pg_roles` does not show membership, so read
  `pg_auth_members` too.

  ```sql
  SELECT r.rolname, r.rolsuper, r.rolbypassrls, r.rolinherit, m.roleid::regrole AS member_of
    FROM pg_roles r LEFT JOIN pg_auth_members m ON m.member = r.oid
   WHERE r.rolname IN ('recall_migrator', 'recall_runtime');
  ```

* **GENERATE the runtime role's grants; never hand-write them.** The package ships the generator and
  [MIGRATIONS.md](MIGRATIONS.md) says why: a copied list drifts out of step with the tables the code
  actually creates, and an earlier hand-written list in that very document omitted four control-plane
  tables and a sequence, so an operator who followed it exactly got `permission denied` at startup.

  ```console
  recall --table <each chunk table> schema grants --role <runtime role> --enterprise
  ```

  ⚠️ **`--table` is a top-level argument and must precede `schema`.** Putting it after the
  subcommand exits 2 with `unrecognized arguments`, and the obvious repair, dropping the flag, is
  the dangerous one: `recall schema grants --role R --enterprise` **succeeds** and grants against
  the default table name `chunks`, leaving every generation table ungranted while the step reads as
  done.

  🛑 **This bullet is filed under "Preconditions" and CANNOT be completed here.** The generator
  emits grants for tables that do not exist yet: the chunk table is created by `schema apply` and
  each generation table by `create-generation`. Executed end to end on a clean database, the order
  that works is **`migrate` → `schema apply` → grants for the chunk table → `create-generation` →
  grants again for each generation table → index**. Skipping that second grant pass is not a
  deferred chore, it is a hard stop at the indexing step: building the shadow dies with
  `InsufficientPrivilege: permission denied for table <generation table>`. Read this bullet now,
  run it twice later.

  Run it **once per chunk table, including every generation table you create at step 3**, and apply
  the output **verbatim** as the object owner. With `--enterprise` it emits six statements covering
  fourteen objects plus one sequence; without it, three. ⚠️ **Do not check your work against a summary, including this one** (that is
  the failure this bullet exists to prevent): diff what you applied against what the command
  printed. For orientation only, the six statements are the migration ledger, the chunk table, the
  eight generation and calibration tables, three read-only control-plane tables, the outbox, and the
  outbox sequence. A `permission denied` at step 8 or 9 means the generated set was
  not applied in full. It never means a broader grant is needed: do not answer it with `GRANT ALL`,
  and do not make the runtime role the table owner. The symptom this deployment actually produced,
  `InsufficientPrivilege: permission denied for table recall_schema_versions`, reads like a database
  outage and is a missing grant.
* **The pgvector EXTENSION**, installed once by a database operator: `CREATE EXTENSION vector;`. The
  migration role must not be elevated to do this. `sparsevec`, which migration 0012 needs, requires
  **extension** version 0.7 or later; check with
  `select count(*) from pg_type where typname = 'sparsevec'`. ⚠️ Do not confuse this with the
  `pgvector` **Python client**, whose declared floor is 0.4.0: two independent version series share
  one name.
* **Model artifacts present and verified locally**, with their digests recorded. Recompute rather
  than trust the record: a tree that agrees with its own manifest proves nothing, so hash it with an
  independent implementation of `artifact_tree_sha256` and compare against the value pinned in the
  package.
* **Licence recorded for every artifact.** A digest says which bytes; it does not say whether you
  may ship them.
* **Outbound network blocked at the workload boundary.** Runtime model downloads are prohibited and
  startup is proven to complete with every socket entry point blocked, but the package cannot
  enforce the boundary. `ufw` defaulting to `allow (outgoing)` satisfies nothing here.
* **Disk headroom at least 2.2x <!--@ citation-pending: a policy rule of thumb, not a measurement; nothing in this repository derives it --> the active index size** before any build, since the shadow is built
  alongside the active generation rather than in place. Measure with
  `pg_indexes_size('<active table>')` against the free bytes on the data directory's mount, not
  against total capacity.

### 1. Migrate the control plane

```console
RECALL_MIGRATION_DSN="$RECALL_MIGRATION_DSN" recall-enterprise migrate
```

Verify: `recall-enterprise status` prints `control plane ledger is current`.

🛑 **On a FRESH deployment that verify line fails, and it is not a symptom of a failed migration.**
`status` takes the SERVING credential, and the serving role has no grants on the control-plane
tables until you apply the generated set. Executed on a clean database: `migrate` exits zero, and
`status` immediately after raises `InsufficientPrivilege: permission denied for table
recall_index_generations`. Apply the generated grants first (see the ordering note in the
preconditions), then this line passes. An operator who reads that refusal as a broken migration
will go looking in the wrong place, and this document used to send them there.

### 2. Apply the chunk-table migrations, per table

Run this for the legacy chunk table and for **every** generation table. It is the step that writes
the per-table ledger rows, and `readiness` reports `SchemaTooOld` for the ACTIVE generation if it is
missing them.

```console
recall --migration-dsn "$RECALL_MIGRATION_DSN" --table chunks_g2026_08 schema --dim 384 apply
recall --table chunks_g2026_08 schema --dim 384 status
```

⚠️ **Pass `--dim` explicitly on BOTH lines, `status` included.** It defaults to the dimension
inferred from `--embedder`, which defaults to `fastembed`, and that resolution happens **before** the
subcommand branches, so even the read-only `status` **constructs an embedder and fetches a model**
without it. On a host with the egress boundary closed (as the preconditions require) the step then
fails on a network fetch that has nothing to do with the schema, and on a host where the model is
already cached it appears to work, which is worse. Note the flag positions: `--migration-dsn` and
`--table` are top-level and precede `schema`; `--dim` follows it.

Verify: `schema status` exits non-zero when the table is not current, and every index on it is still
`indisvalid`.

> ⚠️ **A migration whose bytes changed after you applied it is a hard stop, by design.** The ledger
> stores the checksum of what was applied, and any schema call then raises
> `MigrationChecksumMismatch` rather than migrating forward. There is no flag for this and there
> should not be.
>
> **First work out which mismatch it is. It takes both the command and the message, and neither
> alone.** `MigrationChecksumMismatch` has **nine** raise sites across four functions, and several
> wordings are near-identical while meaning opposite things. The procedure, in order:
>
> 1. **The command bounds which LEDGER wording is possible** (the table's second column).
> 2. **The message decides whether you are in that ledger case at all**, because `load_migrations`
>    runs first inside all four functions and its six wordings can surface from any command.
> 3. Concretely: a message naming the migration and printing **two digests** is `load_migrations`;
>    one beginning **`applied migration`** is `schema_status` or `apply_migrations`; one ending
>    **`does not match the running package`** is `check_schema`. The last three are all ledger.
>
> | Raised by | Reached from | Wording | Meaning |
> |---|---|---|---|
> | `load_migrations` (6 sites) | **any** command, before the database is touched | `migration <name> checksum drift: committed X, actual Y` · `migration manifest/file mismatch (unlisted=…, missing_files=…)` · `migration checksum manifest must be a JSON object` · `duplicate migration version <v>` · `migration <name> must declare exactly one execution mode` · `no packaged migrations found` | **working tree corrupt** |
> | `schema_status` | `schema status`, `schema plan` | `applied migration <file> has checksum X, package has Y` | **ledger** |
> | `apply_migrations` | `schema apply` (step 2) | `applied migration <file> checksum drift` | **ledger** |
> | `check_schema` | **`readiness` (step 9)**, and serving startup | `migration <file> checksum does not match the running package` | **ledger** |
>
> ⚠️ Two traps in the wordings, both of which have caught a previous version of this document. Row
> one and the `apply_migrations` row **both say "checksum drift"** and mean opposite things. And the
> `check_schema` row, the one an operator meets at **step 9**, says neither "checksum drift" nor
> "applied migration": it resembles the working-tree wording most and is a **ledger** error. An
> earlier version of this table listed four wordings, omitted this one, and offered a shape-matching
> rule that sent it to the wrong branch.
>
> A working-tree diagnosis means restore the files and touch nothing in the database.
>
> **The default remedy is to ship the change as a NEW migration version, or to restore from backup.**
> Clearing a ledger row is the exception, not the procedure.
>
> If you do clear it, know exactly what you are doing. The table is `recall_schema_migrations` and
> its primary key is `(target_table, version)`. **Name both columns, always.** Which rows exist
> depends on the version number, and that is a fact about the schema, not a licence to omit the
> predicate:
>
> * **0001 to 0007** are recorded **per chunk table**. A `DELETE ... WHERE version = '0003'` strips
>   the row for *every* chunk table in the database, and every serving process then refuses on
>   restart with `SchemaTooOld`.
> * **0008 and above** are recorded once, under `target_table = '__global__'`, and re-apply
>   database-wide. Only one row exists today, which is why an earlier version of this section used
>   `0012` to argue that both columns are needed and thereby chose the one case that cannot
>   demonstrate it. Do not read that as permission: it is an inference about database state, on a
>   deployment whose ledger contents current code already cannot explain.
>
> ```sql
> -- global (0008+): one row
> DELETE FROM recall_schema_migrations WHERE target_table = '__global__' AND version = '0008';
> -- per-table (0001-0007): one row PER TABLE, so name the table
> DELETE FROM recall_schema_migrations WHERE target_table = 'chunks' AND version = '0003';
> ```
>
> Preconditions, and the first one is the one people get wrong:
>
> * The real question is not "are the two versions equivalent" but **"is re-application safe against
>   the tables as they stand"**. Equivalence says nothing about what re-running does.
> * Take a **restorable backup first**, and do not treat the recorded row as a rollback. Restoring it
>   does not undo DDL the re-apply already executed, and **after any re-apply the ledger's
>   `applied_by` no longer answers "who applied the version that is in the ledger now"**: nothing in
>   the codebase writes that column, so it is set by its `DEFAULT current_user` on a fresh INSERT and
>   left untouched by the upsert. Which of those you get depends on the path taken, so do not rely on
>   it either way. **If you restore the row, do not re-apply on top of it.**
>
>   ⚠️ This bullet has been rewritten three times, and each rewrite asserted a different single
>   branch of `apply_migrations` as though it were the whole behaviour. If you need the exact
>   attribution for an incident, read `recall/schema.py` rather than trusting this paragraph.
> * Re-applying takes **ACCESS EXCLUSIVE** on the target chunk table, and `apply_migrations` sets
>   `statement_timeout = 0` deliberately, so there is no automatic escape. Re-running 0008 also does
>   `ALTER TABLE ... NO FORCE ROW LEVEL SECURITY` and a full scan of the corpus under that lock.
>   Maintenance window only, with an abort procedure agreed before you start.
> * Afterwards the ledger asserts that the **new** bytes were applied, so `schema_status`,
>   `readiness` and `parity` all pass on a premise nobody verified. That is the guarantee you are
>   spending. Get the equivalence argument reviewed by a second person.

### 3. Create the shadow generation

```console
recall-enterprise create-generation g2026_08 chunks_g2026_08 bge-small-asymmetric-v1 384
```

The declared dimension is checked against the artifact at startup, so a generation registered at the
wrong width refuses to serve rather than writing vectors nothing can interpret. Then run step 2
against `chunks_g2026_08`.

### 4 and 5. Build, then mark ready with measured counts

```console
recall-enterprise mark-ready g2026_08 --chunks 1000000 --sources 120000
```

🛑 **And `status` DISPLAYS that unchecked assertion, which is where it will mislead you.** The
`chunks=` column in `recall-enterprise status` is the declared value from the registry, not a count
of the table. Executed: a generation holding five rows, marked ready with `--chunks 0`, prints
`chunks=0` in `status` while `parity` reports five. If you are checking whether a generation is
populated, read `parity`'s counts, never `status`'s.

⚠️ **`--chunks` and `--sources` are an operator ASSERTION and nothing ever checks them.**
`mark-ready` stores the two integers verbatim; `parity` **never compares them** (it reads the
registry to resolve which physical table each generation names, then compares the two tables
themselves), and `cutover` quotes the declared count only inside a refusal message. Measure
them anyway, for your own audit trail, but do not treat the step as a gate: a fabricated
`--chunks 1000000` on an empty generation exits 0. `mark-ready` also has no state guard, so it will
move a `failed` generation to `ready`. Never use it to clear a failed state.

### 6. Attach the shadow route

```console
recall-enterprise set-route acme g2026_07 --shadow-generation g2026_08
```

### 7 to 9. Drain, compare, then check readiness

```console
recall-enterprise status --tenant acme
recall-enterprise replay acme
recall-enterprise parity acme
recall-enterprise readiness acme
```

🛑 **STOP if `shadow chunks` differs from the `--chunks` you measured at step 5.**

✅ **The both-empty case is now refused by the command itself.** When neither generation holds a
chunk for the tenant, `parity` exits non-zero with a refusal naming the tenant and both generation
ids, which says `so the comparison is vacuous and certifies nothing` and then tells you what to do:
while a shadow route exists, indexing writes both generations, so index the corpus and compare
again. It also prints any other parity failure **before** that refusal, so an empty pair whose
row-level security is not forced still says so — for an empty pair this is the only place that
failure surfaces, since `readiness` evaluates the active generation and `cutover` refuses on
emptiness before reaching its own parity check. It used to exit zero and print `parity: OK` — two empty generations cannot
disagree, so every comparison it makes was vacuously satisfied. That guard lived only in this
paragraph, which is the weakest place to keep one: prose, in the document an operator is reading
for permission to proceed. There is **no override flag**, and that is deliberate — see
`--allow-divergent-corpus` below for why a refusal that advertises its own escape hatch is worse
than no refusal.

⚠️ **What that closes is the misleading green at THIS step, not a path to a promoted empty index.**
`cutover` calls `_require_non_empty_shadow` before its parity check, deliberately outside it so
`--allow-divergent-corpus` cannot skip it, and that refusal has always been there. Do not read the
parity refusal as the thing standing between you and an empty promotion.

⚠️ **A shadow partially filled relative to the ACTIVE generation IS caught** — parity fails on
`shadow generation is missing sources`, on `shadow generation contains extra sources`, or on
`chunk counts differ between generations`. What no guard catches is a pair that agrees with each
other and is short of the **corpus on disk**, because parity compares the two generations to each
other and to nothing else. That is why the paragraph below tells you to compare the shadow's source
set against the corpus rather than only against the active generation, and why you should read the
chunk counts this command prints against the counts you measured at `mark-ready`.

⚠️ **Emptiness is not the only vacuous green.** `source_raw_hashes()` reads
`coalesce(metadata->>'content_hash', '')`, so two generations whose rows all lack a content hash
compare an empty string against an empty string and agree perfectly while certifying nothing.
`parity` does **not** detect that; `benchmarks/check_generation_parity.py` gates it as a separate
blocking control.

`cutover`'s own emptiness check only catches a **totally** empty shadow, so a partially filled one
passes *that* check. It does not pass `parity`, which compares the two generations and fails on
missing sources, extra sources or differing chunk counts — and `cutover` runs that comparison too
unless `--allow-divergent-corpus` is passed. The gap on the delete path is therefore a gap in what
the comparison can *see*, not a hole it waves through: `_prune_vanished` keys its candidate set on
the active generation, so a source the shadow holds and the active does not survives the prune, and
it is reported here as `extra sources` — which an operator under time pressure can mistake for a
deliberate corpus change and clear with the very flag that then skips the comparison. Compare the
shadow's source set against the corpus on disk, not only against the active generation.

Run `readiness` with `RECALL_SERVING_DSN` set. Its row level security verdict is about the role it
connects as, and it prints that role, so the verdict names its own subject.

⚠️ **`readiness` is the one step that requires the MODEL ARTIFACTS to be present.** It builds the
route's declared embedding profile in order to verify model identity, so on a host without the
provisioned tree it raises from the embedder rather than reporting a readiness verdict. That is
consistent with the preconditions, which require the artifacts anyway; it is called out here
because it is the step where their absence first stops you, and the traceback names the embedder
rather than the missing precondition. On the migration role a
green verdict would certify a credential that never serves a request.

⚠️ **`readiness` names its own ROLE subject and not its own GENERATION subject.** It opens
`route.active`, so run before cutover it certifies the **outgoing** generation, not the shadow you
are about to promote. Nothing in steps 7 to 9 runs the startup checks against the incoming one.

### 10. Cutover, then verify

```console
recall-enterprise cutover acme
recall-enterprise status --tenant acme      # confirm the swap
recall-enterprise readiness acme            # NOW evaluates the promoted generation
```

`cutover` prints nothing on success, so the two verify lines are how you learn it worked. The second
one is the real gate: it is the first time the startup checks run against the generation now serving
traffic. A red verdict here is what section R exists for.

⚠️ **`--allow-divergent-corpus` skips the parity comparison entirely.** The parity refusal advertises
it in its own message, which puts the flag in front of an operator at the exact moment they are
under pressure. It is permitted **only** when the source-set difference has been enumerated and
matches a deliberate corpus change (documents genuinely added or removed). It is forbidden as a
response to any parity failure whose cause is unknown, and a half-filled shadow produces a failure
indistinguishable from an intended change. It does not skip the pending-event, ready, or emptiness
checks.

### 11. Retire, after the rollback window

⚠️ **Not immediately after step 10.** `cutover` SWAPS the slots: the old generation becomes the
tenant's **shadow**, and `retire` refuses while the named tenant routes at that generation in
*either* slot. Detach it from the shadow slot first, then retire.

⚠️ **Retirement is DATABASE-GLOBAL.** `recall_index_generations` has no tenant column: `retire`
checks one tenant's route and then sets `state='retired'` for the whole database, so every other
tenant still routed at that generation is refused by the serving path immediately. Enumerate every
tenant routed at the generation before retiring. Afterwards `set-route` rejects it as an active
generation, so **section R no longer works as written**.

There is no un-retire *command*, but there is an un-retire *path*, and you should know it before you
order a restore: `mark-ready` has no state guard (step 5), so
`recall-enterprise mark-ready <retired-generation> --chunks N --sources M` writes it back to `ready`
and `set-route` will then accept it. Treat that as an incident procedure, not a rollback step:
nothing re-validates the table, `retired_at` is left set, and the counts you pass are unchecked.

⚠️ `retire` has **no state precondition**: it checks only the named tenant's two route slots, so a
`failed` or `building` generation, which is never routed, can be retired too. Do not read the
un-retire path above as safe for any generation that happens to be in state `retired` — it says
nothing about whether that table was ever complete.

```console
recall-enterprise retire g2026_07 --tenant acme
```

Keep the old table for seven days and two successful backup cycles before retiring.

### R. Rollback

Cutover swaps the previous active generation into the shadow slot, so rolling back means naming it
active again. **Pass `--shadow-generation` explicitly**: it defaults to `None` and is written
straight through, so the obvious short form silently NULLs the shadow route and detaches the
generation that was serving seconds ago.

```console
recall-enterprise set-route acme g2026_07 --shadow-generation g2026_08
recall-enterprise status --tenant acme      # expect active=g2026_07 shadow=g2026_08
```

The serving path independently refuses a retired or failed generation, per request, so a retired
table cannot be reached even if a route still names it. This works only before step 11.

### Credentials by subcommand

`recall-enterprise` picks its credential by subcommand, so the operator no longer has to.
`migrate` and `create-generation` perform DDL and read `RECALL_MIGRATION_DSN`; `readiness`,
`status`, `parity` and `replay` take `RECALL_SERVING_DSN`; `mark-ready`, `set-route`,
`cutover` and `retire` are DML against the control-plane tables and take the migration credential.
⚠️ Of that middle group only `readiness`, `status` and `parity` are read-only. **`replay` writes**,
into both generations; it takes the serving credential because that is the credential the drain
needs, not because it is a read.
All of them fall back to `RECALL_DSN`, so a single-variable deployment keeps working.

The split matters most for `readiness`, which reports whether row level security constrains "the
runtime database role". That check reads `current_user` of the connection it was handed, so on the
migration role a green verdict would certify a credential that never serves a request. The command
prints the role it evaluated, so the verdict names its own subject.

```console
RECALL_MIGRATION_DSN="$RECALL_MIGRATION_DSN" recall-enterprise migrate
```

> ⚠️ `RECALL_DSN` is also the deprecated fallback the serving process and the MCP server
> read when `RECALL_SERVING_DSN` is unset (see [MIGRATIONS.md](MIGRATIONS.md#configuration)).
> Exporting it globally as the migration role therefore hands a schema-owner credential to
> every serving process, which is exactly what the role split in
> [SECURITY_MODEL.md](SECURITY_MODEL.md) forbids. Set it per command, never in the serving
> environment.

The database operator must install pgvector once in a new database before the restricted
migration role creates a generation:

```sql
CREATE EXTENSION vector;
```

Do not grant superuser or `BYPASSRLS` to the migration or runtime roles.

Create an empty generation table and register its profile identity:

```console
recall-enterprise create-generation g2026_08 chunks_g2026_08 bge-small-asymmetric-v1 384
```

Build and validate the shadow corpus, then mark it ready with measured counts:

```console
recall-enterprise mark-ready g2026_08 --chunks 1000000 --sources 120000
recall-enterprise set-route acme g2026_07 --shadow-generation g2026_08
```

While a shadow route exists, indexing prepares both vector sets before either generation changes. It records a durable ordered event, applies the active and shadow writes, then clears the event payload on completion. A crash leaves an idempotent replay record. The `recall_forget` MCP tool deletes from both generation tables in one database transaction, then scrubs the erased sources out of any pending replay record, so a later replay cannot restore them. The scrub is keyed on the sources the caller named, not on what still had rows, because the case that most needs it is the one where a crash left the text in the outbox and nowhere else. One window remains and is reported rather than hidden: the deletes commit before the scrub, so a crash between them leaves the outbox entry, and the result carries `outbox_events_scrubbed = -1` when the scrub failed after the deletion succeeded. ⚠️ The `recall forget` CLI is single-generation and does NOT scrub the outbox; on an enterprise deployment use the MCP tool.

Cutover refuses to proceed while any migration event is pending or while the shadow is not ready:

```console
recall-enterprise cutover acme
```

A crash that left an event pending blocks cutover until the outbox is drained. Drain it, then compare the two generations before promoting:

```console
recall-enterprise status --tenant acme
recall-enterprise replay acme
recall-enterprise parity acme
```

`replay` opens only the generations the pending events name, resolving each physical table from `recall_index_generations`, and exits non-zero if anything is still pending afterwards. `parity` exits non-zero when the generations disagree on sources, raw content hashes or chunk counts, when either generation has an invalid required index or does not have row level security forced, and when both generations are empty (two empty generations cannot disagree, so the comparison would be vacuous). The step table above is the single authoritative list. `status` reports generations, the tenant's route and the outbox depth; it never prints a pending event's payload, which holds corpus text and vectors. It also lists any registry row whose `physical_table` the identifier allowlist rejects, rather than failing on it: such a row cannot serve, and the command an operator uses to find it must not be the command that dies on it. Run `recall-enterprise status` before upgrading.

`readiness` runs the startup checks for one tenant without starting a server, and exits non-zero when any of them fails. Run it with `RECALL_SERVING_DSN` set: its row level security verdict is about the role it connects as, and it prints that role so the result names its own subject.

```console
recall-enterprise readiness acme
```

The route update is transactional and sends a content free PostgreSQL notification. Service processes invalidate their cached route immediately. The fallback is a five second cache TTL on the route, not a poll: a process whose notification never arrives picks the new route up within that window on its next request. Existing requests keep their acquired store object. New requests use the new generation.

## Runtime configuration

Set `RECALL_ENTERPRISE_CONTROL_PLANE=1` only on authenticated HTTP deployments. Enterprise readiness then fails startup when a route is missing, the control plane is unreachable, the profile or dimension differs, the active generation is not `ready` or `active`, either schema ledger is not current, required indexes are invalid, row level security is ineffective, model identity is unverified, a loaded calibration names a different embedding profile, or stored rows lack profile metadata. A database carrying migrations this package does not ship is reported as degraded rather than fatal (readiness returns `degraded=true` with a warning the server logs), so migrating forward and then rolling the application back does not refuse to boot.

Choose one service cost profile per process:

* `RECALL_RETRIEVAL_PROFILE=fast` uses twenty candidates per retrieval leg and no reranker. It returns five hits and budgets 250 ms.

* `RECALL_RETRIEVAL_PROFILE=quality` uses the same candidate pool and the local pinned reranker. It returns five hits and budgets 1500 ms. The reranker scores the COMPLETE fused candidate pool before truncation, so a relevant passage sitting just below the fused cutoff can still be rescued, which is the only reason the stage exists.

Run separate deployments when both profiles are required. Clients cannot select the expensive path per request: the profile is read from the process environment and a request's `k` is clamped down to the profile's returned count, never raised. Leaving `RECALL_RETRIEVAL_PROFILE` unset preserves the legacy `RECALL_RERANK` switch exactly; setting the two to values that contradict each other refuses **startup**, not the first search.

### The latency budget at request time

`latency_budget_ms` means two enforced, observable things.

It **bounds the admission wait**. A request that cannot acquire a running slot within the budget is shed with `RetrievalOverloaded` before the query is embedded. That ordering is the point: admission is taken ahead of the embedder, so a refused request costs nothing. Without the bound, `queue_capacity` caps how many threads may be parked and says nothing about how long any of them waits, and a process can hold a client for minutes behind a slow reranked query while every counter reads healthy. `RetrievalOverloaded` carries a `reason` (`queue_full` or `budget_exhausted`) and a `retry_after_seconds`, matching how `RateLimited` reports a retryable refusal.

It **labels an overrun**. A request that completes over budget still returns its answer, and reports `total_ms`, `latency_budget_ms` and `budget_exceeded` on the response. Aborting in flight was rejected: there is no cancellation point inside a blocking cross-encoder `predict`, so the process would pay the whole cost and then discard the answer, converting a latency regression into an availability incident.

The budget is charged **once**. `budget_exceeded` is computed on the work a request actually did (`total_ms` minus `admission_wait`), not on its end-to-end latency. Since the budget is already spent as the admission timeout, a request may legitimately wait almost all of it before starting; comparing the same allowance against the total as well would label a fast retrieval slow because another request was ahead of it, and would saturate the counter under any queueing. `total_ms` still reports client-visible latency, which is a different and also necessary number.

The **legacy profile enforces no budget**, and reports `latency_budget_ms` as `null` rather than as the 24-day sentinel the code uses internally. `budget_exceeded` is then always false.

Each profile carries its **own** concurrency budget rather than one shared default: fast admits 8 concurrent with 32 queued, quality 2 with 8, legacy 4 with 16. Quality's per-request budget is six times fast's, so an equal queue depth would make its clients wait roughly six times as long; the numbers hold `queue_capacity * latency_budget_ms` within one order of magnitude (fast 8000 slot-ms, quality 12000). These values are a policy choice, not a measurement, and the latency blocker in `archive/ENTERPRISE_PROGRAM_STATUS.md` is why. `RECALL_SEARCH_CONCURRENCY` and `RECALL_SEARCH_QUEUE` override them for the selected profile.

The admission gate is entered inside a worker thread, so its capacity is denominated in threads whether or not it says so. The server therefore **sizes the worker pool from the profile** at startup (`worker_thread_budget`: admission capacity plus eight reserved threads), and only ever raises it. Without that, fast's 8 + 32 would exactly equal anyio's 40-token default: the request that should be shed would never reach the gate at all, it would wait in anyio's limiter, which has no timeout and no counter, and `recall_retrieval_rejected_total` would read zero while clients waited unboundedly. The reserved headroom keeps queued searches from starving `recall_index`, `recall_forget`, `recall_stats` and token validation.

`RECALL_RERANK_THREADS` bounds inference threads **on the quality profile only**; it is not read on the legacy `RECALL_RERANK` path. One reranker is built per worker process, under a construction lock: a cache lookup is not a lock, and a cold start under load would otherwise have every concurrent first request load its own copy of the model. A construction that fails is cached too, so a broken artifact fails immediately instead of re-hashing the model tree on every request.

### The quality profile's reranker is pinned by digest

`RECALL_RERANK_PATH` is deployment specific. The artifact digest is not: `recall/rerank.py` pins artifact SHA256 `db6ad87969c7dc78320152e68a16118aeb4b2a6f7d8cc979c57f61ddb5e2ab2a`, and `RECALL_RERANK_SHA256` must equal it. Verifying the tree against a digest the operator supplied would prove only that the tree hashes to its own hash; the pin is the value chosen elsewhere that makes the comparison mean something.

Two limits on what that pin says, both deliberate.

The model name `cross-encoder/ms-marco-MiniLM-L-6-v2` and revision `c5ee24cb16019beea0893ab7796b1df96625c6b8` are recorded beside it as **provenance, not as a runtime check**. Nothing reads them at load time: the quality profile loads from a local tree with `local_files_only`, where the Hub revision is unused.

And the digest is a hash of a whole provisioned **tree**, path names included, so it identifies one provisioned directory rather than the model in general. A differently laid out copy of the same weights (a Hugging Face `blobs`/`snapshots` cache, a `snapshot_download` that left a lock file behind) hashes differently and is refused. This deployment's tree is the one recorded in `/opt/recall-enterprise/manifest.json`. There is no shipped command that reproduces it elsewhere, which is a real gap for any operator outside that host and is recorded as such in `archive/ENTERPRISE_PROGRAM_STATUS.md`.

### What every result reports

Every search response carries, additively: the embedding profile identity, the retrieval profile, the index generation, the candidate pool size, whether reranking ran, `total_ms`, `latency_budget_ms`, `budget_exceeded`, and per-stage wall time for `admission_wait`, `query_embedding`, `dense_retrieval`, `sparse_retrieval`, `learned_sparse_retrieval`, `fusion`, `reranking`, `trust_evaluation` and `evidence_assembly`. Every one of those keys is present on every response, including for a retrieval leg the configuration switched off: `sparse_retrieval` on a SPLADE-only pipeline, or `learned_sparse_retrieval` on a pipeline with no learned sparse arm, reports ~0 rather than dropping its key. That is deliberate, so that an absent series is unambiguously a missing instrument rather than possibly a disabled leg. The same stage timings are observed into `METRICS` under `recall_retrieval_stage_ms`, labelled by profile and stage, so per-stage percentiles exist across a population of queries rather than only per response. `recall_retrieval_total_ms` is observed on failures as well as successes, because a timer that only records on success hides the slow path worth finding. It deliberately excludes requests that were **shed**: those did no work by construction, so booking them would make healthy load shedding indistinguishable from an outage and would contaminate the served-latency population with rejections in exactly the overload regime where a p95 matters most. A shed request appears in `recall_retrieval_rejected_total{profile,reason}` and nowhere else.

Every hit's `score` is its dense cosine, including hits that arrived through the sparse leg and hits the reranker moved. Reranking reorders and never rewrites the score, because calibration thresholds are stated in cosine units and a cross-encoder logit is not one.

Metric labels and log records are library-authored throughout. No query text and no corpus text reaches a log record, a metric label or an exception message; a test drives a distinctive sentinel through the search path and asserts its absence from every captured record, with a positive control proving the detector can fire.

## Evidence integration

Use `build_evidence_bundle`, `render_evidence_prompt`, and `validate_answer`. They are exported from the package root (`from recall import build_evidence_bundle`) as well as from `recall.evidence`. `generate_from_evidence` is the optional orchestration helper. It never invokes its generator when retrieval abstains. The fixed system prompt contains no corpus controlled value. Evidence is JSON escaped inside the user data message, and successful answers require citations that resolve to supplied chunk IDs.

Validation is structural. It does not claim that a cited passage entails an answer.

### What enters a bundle, and what never does

Only `ok` verdicts. A DEGRADED result — the trust gate could not run, every verdict is `unverified`, and `abstained` is forced False — produces an EMPTY bundle with `reason_code="no_trusted_evidence"`, not an unjudged one. Retrieval order is preserved: no newest wins, no re-sort by score. There is no semantic deduplication, so two chunks with identical text remain two citable identifiers. There is no neighbour retrieval: the module holds no store and `build_evidence_bundle` takes no argument through which one could be supplied, so a passage that was not retrieved cannot appear.

An abstained retrieval produces an empty bundle and bypasses the generator entirely — `generate_from_evidence` returns `insufficient_evidence=true` with `generator_invoked=False` without constructing or calling anything.

### The prompt boundary

`render_evidence_prompt` returns the module constant `SYSTEM_PROMPT` itself. There is no format string and no argument on that path, so there is no site at which a corpus controlled value could be interpolated. Every corpus byte lives inside `<evidence_data>…</evidence_data>` in the second message, JSON escaped — including both angle brackets, which `json.dumps` does not escape and which the delimiter is made of. A frozen adversarial suite (`benchmarks/evidence_injection.py`) runs thirteen payloads through four carriers (file name, chunk metadata, memory text, chunk id) and records the escape rate in `results/evidence_injection_baseline.json`, with a positive control against the previous renderer so that a zero cannot be produced by an inert detector, and a negative control against a renderer that ships no evidence at all so that a perfect score cannot be bought by deleting the corpus text. The chunk-metadata arm carries no payload into the prompt at all — an evidence item has no corpus metadata dict — so its zero is structural rather than earned, and it is excluded from the rate's denominator. The suite and its baseline ship in the source distribution and the repository, not in the wheel.

### Citations

At least one per answer, and every one must resolve to a chunk ID in the bundle. Duplicates are collapsed deterministically by `normalize_citations` — first occurrence order, idempotent, and only ever subtractive, so normalisation cannot mint an identifier that would then satisfy the resolution check. `GenerationResult.citations_normalized` reports whether that edit happened. `validate_answer` on its own remains strict and reports a duplicate as an error.

A token budget requires an injected tokenizer: `EvidencePolicy(max_tokens=…)` with no `tokenizer` raises rather than estimating.

### Reaching it from the library and the four integrations

| Surface | Entry point |
|---|---|
| Library | `from recall import build_evidence_bundle, render_evidence_prompt, validate_answer` |
| CLI | `recall search "<query>" --evidence` prints the bundle and the rendered prompt as JSON |
| MCP | the `recall_evidence` tool returns the bundle plus `system_prompt` and `user_message` |
| LangChain | `RecallRetriever.evidence(query)` / `.evidence_prompt(query)` |
| LlamaIndex | `RecallRetriever.evidence(query)` / `.evidence_prompt(query)` |

All five surfaces are additive: every pre-existing field, metadata key and tool is unchanged, and each of the four integrations carries a test asserting a frozen list of its pre-existing keys. (Four integrations plus the library import itself, which is why the table has five rows.)

`recall_evidence` runs no generator. This deployment chooses none and ships none, so the tool stops one step short and hands back the two messages for the client to run its own model against. That is what generator neutrality means here, and it is also why the end-to-end path with a real generator remains unexercised: no approved local generator has been confirmed for this program. The neutral flow is tested against a stub.

The retriever adapters deliberately do NOT honour their `include_untrusted` escape hatch in `evidence()`. That flag exists so a caller can inspect what the trust layer refused; what a generator may cite is a rule, not a constructor setting.

## Promotion

`recall.promotion.evaluate_retrieval_promotion` implements the paired macro bootstrap interval, per corpus regression limit, paired sign tests with Holm correction, safety parity checks, security gate, and latency budget. Experiments remain opt in until the decision reports `promoted=true`. Negative artifacts should be retained with fixed question identifiers and model digests.

### Producing a decision

`recall/eval/promotion/` builds the gate's input. Run it in three steps, in this order:

```bash
python -m recall.eval.promotion freeze --corpus labelled \
    --out results/promotion/labelled.manifest.jsonl

python -m recall.eval.promotion run --manifest results/promotion/labelled.manifest.jsonl \
    --expected-digest <the digest freeze printed> --corpus labelled --arm baseline \
    --dsn "$RECALL_DSN"

python -m recall.eval.promotion decide --manifest results/promotion/labelled.manifest.jsonl \
    --baseline <baseline ledger> --candidate <candidate ledger> \
    --out results/promotion/decision.json
```

`freeze` must run **before** either arm. It fixes question ids and input hashes while no candidate result exists, which is what makes "the same questions" checkable rather than assumed. It refuses to overwrite an existing manifest, and the reader refuses a body that no longer matches its digest. Each arm's rows carry the frozen `input_hash`, and `decide` refuses a pair of arms that do not cover the manifest exactly.

Supported corpora: `labelled` (in-repo), `peps`, `locomo`, `ladder`, `longmemeval`, `mtrag`. Every adapter declares which id space its labels live in, with no default, because a wrong one scores an entire corpus as a total miss and reports a successful run.

**Latency is PENDING by default, and PENDING blocks promotion.** `decide` emits `latency_p95_ms=None` unless `--certified-latency-p95-ms` is passed, and that flag is for a number measured on an idle reference host. This program has no such host, so every decision it can produce today is blocked on latency; the observed p95 is recorded in the artifact under `observed_diagnostic_only` and is not a gate input.

**Calibration is scoped to the complete embedding identity.** Use `recall.calibration.save_for_profile` and `load_for_profile`, not `save` and `load_for`, for anything a gate reads. The latter pair keys on the profile ID alone, and two runs sharing a profile ID can differ in artifact digest, dimension, encoder modes or chunker version, each of which moves the cosine regime a threshold was fitted in. `load_for_profile` fails closed on a calibration that cannot show which identity it belongs to.

**A run against a plain store is DEGRADED.** Strict trust mode refuses a store with no generation-bound certified calibration, which is correct. `--trust-policy development` runs anyway and records every hit's verdict as `unverified`, which carries no trust claim at all: the safety metrics of such a run describe a degraded system and are not a measurement of the trust layer. The decision artifact records the verdict distribution so a reader cannot mistake one for the other.

## Rejected profile: Qwen3-Embedding-0.6B truncated to 384 dimensions

`qwen3-embedding-0.6b-384-v1` is registered and **rejected**. It is kept, with its measurement, so that the decision is reproducible and so that no later session re-measures it by accident. It is not a candidate and it is not gated on anything that could still open.

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-Embedding-0.6B` |
| Revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Artifact SHA256 | `0e9f06588b7e661b8d8e6d393b5936750e428ec422f9971c7f02838dbe70fc9f` |
| Dimension | 384 (truncated, renormalized after truncation) |
| Licence | Apache 2.0 |
| Verdict | rejected on CPU serving latency, 2026-08-03 |

Measured offline on the provisioned artifact at a four thread budget:

| Measurement | Value |
|---|---|
| Query p50 | 4638.83 ms <!--@ citation-pending: measured 2026-08-03 on the VPS2 provisioned artifact at a four-thread budget; the run predates this repository's artifact convention and no committed results/*.json retains it --> |
| Query p95 | 5816.34 ms <!--@ citation-pending: measured 2026-08-03 on the VPS2 provisioned artifact at a four-thread budget; the run predates this repository's artifact convention and no committed results/*.json retains it --> |
| Passage batch of 20, p50 | 41016.64 ms <!--@ citation-pending: measured 2026-08-03 on the VPS2 provisioned artifact at a four-thread budget; the run predates this repository's artifact convention and no committed results/*.json retains it --> |
| Model load | 24558.4 ms <!--@ citation-pending: measured 2026-08-03 on the VPS2 provisioned artifact at a four-thread budget; the run predates this repository's artifact convention and no committed results/*.json retains it --> |
| Peak RSS | 1739.47 MB <!--@ citation-pending: measured 2026-08-03 on the VPS2 provisioned artifact at a four-thread budget; the run predates this repository's artifact convention and no committed results/*.json retains it --> |

The fast retrieval profile budgets 250 ms and the quality profile 1500 ms. A query p95 of 5.8 seconds is more than three times the quality budget for the embedding step alone, before any store or reranker cost, and a 41 second batch of twenty passages makes bulk indexing impractical on the same hardware.

Two limits on what this says. It is a latency verdict, not a quality one: retrieval quality was never measured against `bge-small-asymmetric-v1`, so nothing here claims the model retrieves worse. And it was measured on CPU, at four threads, on the host described under the latency blocker in `archive/ENTERPRISE_PROGRAM_STATUS.md`. GPU requirements are out of scope for this program, so a GPU number would not change the decision.

The registry pins the artifact digest for this profile. A different artifact tree is a different experiment and is refused rather than inheriting this verdict.

## Rollback and retirement

Cutover swaps the previous active generation into the shadow route. Restore it with `set-route` if rollback is required. Keep the old table for seven days and two successful backup cycles. Removal is an explicit operator migration after the rollback period. Never allow a request field to name a retired table.

After the rollback window, retire the old generation:

```console
recall-enterprise retire g2026_07 --tenant acme
```

Retirement is confirmed one tenant at a time, and the reason is the isolation model rather than convenience: `recall_tenant_routes` carries forced row level security keyed on the tenant, and neither the migration role nor the runtime role may enumerate every tenant's routes to prove a generation is globally unrouted. The command therefore refuses while the named tenant's route references the generation, and the serving path refuses a retired or failed generation independently, per request. That second refusal is the one that protects a request; weakening the isolation model to make a single global check possible would have cost more than it bought.
