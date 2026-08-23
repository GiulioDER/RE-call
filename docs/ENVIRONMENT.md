# Environment Reference

This is the detailed environment reference. The root `.env.example` keeps the short copyable
template, while this file records the operational rationale behind each group.

## Root Template Notes

Copy `.env.example` to `.env` and fill it in. The `.env` file is only read by local entry points via
`recall/_env.py`. It is never committed.

```dotenv

# Optional: enables cloud embedders. Everything else in the project runs key-free with the local
# FastEmbed embedder.
VOYAGE_API_KEY=
OPENROUTER_API_KEY=

# Deployment environment: development (default) | test | production. Selects the production
# code paths: the v1 GenerationStore for `search` and `forget`, generation mode in the MCP
# server, refusal of local-filesystem indexing, pinned-embedder verification, and the
# certification gate on `generation promote` (production promotes only a CERTIFIED generation and
# refuses --unsafe-development-promotion; development is the reverse). `generation rollback` is
# ungated in both, by design. It also governs MCP authentication: `production` REFUSES the
# static token file
# (RECALL_AUTH_TOKENS_FILE below), so an HTTP transport there must authenticate via OIDC.
# Anything other than "production", including an unset value, a typo such as "prod", or a stray
# trailing space, resolves to development and leaves every one of those guards OFF.
# The full table of what `production` enables is in docs/MIGRATIONS.md.
# RECALL_ENV=development

# Optional overrides (defaults shown):
# The serving/migration credential split (docs/MIGRATIONS.md): RECALL_SERVING_DSN is the
# unprivileged credential used by indexing, search, forget and MCP; RECALL_MIGRATION_DSN is
# the schema owner and belongs only in the migration job. Keep them distinct outside a
# disposable local database.
# RECALL_SERVING_DSN=postgresql://recall_server:...@localhost:5432/recall
# RECALL_MIGRATION_DSN=postgresql://recall_migrator:...@localhost:5432/recall
# RECALL_DSN=postgresql://recall:recall@localhost:5432/recall   # deprecated serving fallback
# RECALL_EMBEDDER=fastembed          # or "hashing" for the fully-offline embedder
# RECALL_EMBEDDER=gemini-embedding-2 # OpenRouter google/gemini-embedding-2
# RECALL_EMBEDDER=openrouter:<provider/model>  # any other OpenRouter embedding model
# RECALL_EMBED_DIMENSIONS=1536       # optional for OpenAI-compatible embedders
# RECALL_EMBEDDER=sfr-code           # Salesforce/SFR-Embedding-Code-2B_R, research/Gemma terms
# RECALL_ACCEPT_RESEARCH_MODEL_LICENSE=1
# RECALL_ACCEPT_REMOTE_MODEL_CODE=1  # required only for models that need trust_remote_code
# RECALL_INDEX_ROOT=/srv/recall/corpus  # corpus-only root for the MCP recall_index tool
# Legacy RECALL_CALIBRATION files are import-only evidence; v1 search resolves calibration from Postgres.

# Appends one record per search decision (answered, abstained, or refused) to the tenant's
# audit table, best-effort. Off unless enabled; a malformed value warns once and stays off
# rather than refusing searches. See docs/DECISION_LEDGER.md.
# RECALL_DECISION_LEDGER=0

# --- MCP transport & authentication (see docs/AUTH.md) ---
# Default is stdio: a private pipe to one client, which needs no authentication.
# The HTTP transports open a socket and REFUSE TO START without either a token file or an OIDC
# issuer. Setting both refuses too, UNLESS RECALL_AUTH_MODE declares which one is active — that
# is the supported way to stage a cutover (see the bottom of this block).
# Which of the two is permitted depends on RECALL_ENV (documented above): `production` refuses
# the static token file, leaving OIDC as the only option there.
# RECALL_TRANSPORT=stdio                     # or streamable-http / sse
# RECALL_AUTH_TOKENS_FILE=/etc/recall/tokens.json   # chmod 600; there is deliberately NO
#                                            # env var that accepts a raw token
# RECALL_AUTH_ISSUER_URL=https://recall.example.com   # optional with OIDC: defaults to the issuer
# RECALL_AUTH_RESOURCE_URL=https://recall.example.com
# RECALL_TENANT=default                      # stdio only; on HTTP the token carries the tenant

# Option 2, and the production one: identity from an OIDC provider, so revocation, rotation and
# expiry belong to the IdP. The tenant list is MANDATORY — absent is not "every tenant", and the
# IdP knows nothing about this deployment's topology.
# RECALL_OIDC_ISSUER=https://idp.example.com   # https only; NOT RECALL_AUTH_ISSUER_URL
# RECALL_OIDC_AUDIENCE=recall-prod             # the audience this server accepts
# RECALL_OIDC_TENANTS=acme,globex              # the tenants this deployment serves
# RECALL_OIDC_ALGORITHMS=RS256,ES256           # optional; defaults to the asymmetric set.
#                                            # HS*/none are refused: with a published key an
#                                            # HMAC algorithm IS the confusion attack
#
# WHO may name a tenant. The list above bounds WHICH tenants exist, not who may reach them, so
# exactly one of the next two is REQUIRED (the server refuses to boot without a decision).
# RECALL_OIDC_SUBJECT_TENANTS=alice@corp:acme,svc-etl:globex   # pin sub -> tenant(s) here
# RECALL_OIDC_TRUST_TENANT_CLAIM=1             # or: the IdP mints `tenant` from an AUTHORITATIVE
#                                            # subject-to-org mapping, never a user-editable
#                                            # attribute. Setting both refuses to boot.
#
# Staged cutover from the token file: set both mechanisms, then declare which is active.
# RECALL_AUTH_MODE=static                    # step 1: nothing changes. Then flip to `oidc`,
#                                            # then remove RECALL_AUTH_TOKENS_FILE.

# --- Abuse bounds (see SECURITY_MODEL.md) ---
# Per-request caps bound ONE call; the budgets below bound the aggregate, per TENANT, so a
# client staying under the per-request cap cannot simply issue it in a loop.
# RECALL_INDEX_MAX_FILES=2000        # per request: candidate file count
# RECALL_INDEX_MAX_BYTES=20000000    # per request: candidate bytes (~20 MB)
# RECALL_RATE_READ_PER_MIN=120       # per tenant: recall_search / recall_evidence / recall_stats calls
# RECALL_RATE_WRITE_PER_MIN=20       # per tenant: recall_index / recall_ingest / recall_calibration_run calls
# RECALL_RATE_FORGET_PER_MIN=10      # per tenant: recall_forget calls
# RECALL_RATE_ADMIN_PER_MIN=10       # per tenant: recall_calibration_publish calls
# RECALL_INDEX_BYTES_PER_HOUR=209715200  # per tenant: aggregate indexed bytes (200 MiB).
#                                    # Keep this >= RECALL_INDEX_MAX_BYTES, or requests between
#                                    # the two sizes can never succeed.
# Each of the five call budgets takes a number or the literal `off`. A malformed value, a
# non-finite one, or one too small to yield a non-zero rate falls back to its default rather than
# being read as "unlimited" — only `off` disables a limit.
# RECALL_RATE_AUTH_FAILURES_PER_MIN=60   # PROCESS-GLOBAL (not per tenant): the pre-auth failure
#                                    # throttle that caps a forgery storm's JWKS/RSA work on the
#                                    # OIDC path. `off` disables it. A sustained storm above this
#                                    # rate refuses valid OIDC tokens too (see docs/AUTH.md); the
#                                    # static token path is never gated by it.
# Read once at startup: changing a budget takes effect on restart.

# --- Optional presentation localization ---
# Disabled unless explicitly enabled. `recall_search` and `recall_evidence` accept an optional
# locale argument and then add a `localized` object. The original fields, provenance, evidence
# items, system prompt, and user message remain canonical and unchanged.
# RECALL_TRANSLATION_ENABLED=0
# RECALL_TRANSLATION_ENDPOINT=https://api.translate.zvo.cn/translate.json
# RECALL_TRANSLATION_TIMEOUT_SECONDS=5
# RECALL_TRANSLATION_MAX_BATCH=32
# RECALL_TRANSLATION_MAX_TEXT_CHARS=20000
# RECALL_TRANSLATION_MAX_RESPONSE_BYTES=2000000
# RECALL_TRANSLATION_ALLOW_HTTP=0       # only needed for an explicitly permitted HTTP endpoint
#
# The integration calls the upstream text JSON endpoint only. It does not load the browser
# translator, remote scripts, or runtime-evaluated code. Provider failures fall back to canonical
# values and are reported as a fixed warning without exposing provider errors or corpus text.
# Enabling it sends selected retrieved passage text to the configured endpoint. Use a self-hosted
# endpoint when corpus confidentiality requires it, and treat localized values as display data,
# not as a replacement for the canonical evidence prompt.

# --- Schema DDL ---
# RECALL_SCHEMA_LOCK_TIMEOUT_MS=5000  # how long ensure_schema() may WAIT FOR A LOCK before
#                                     # giving up. NOT a bound on the work — an HNSW build is
#                                     # deliberately unbounded — only on queueing behind another
#                                     # transaction. `0` waits forever. The DDL is idempotent and
#                                     # retried on the next store open, so failing fast here
#                                     # loses nothing and is diagnosable where a stall is not.
# Immutable process profiles. Registered identifiers live in recall/embedding_registry.py and
# nowhere else. They come in two kinds, and the kind decides which variables below apply.
#
# LOCAL profiles, from a provisioned artifact tree (set RECALL_EMBEDDER=fastembed):
#   bge-small-symmetric-v1, bge-small-asymmetric-v1, bge-small-context-document-v1,
#   bge-small-context-section-v1, bge-small-context-neighbor-v1,
# and qwen3-embedding-0.6b-384-v1, which is registered and REJECTED on CPU serving latency
# (docs/ENTERPRISE_RETRIEVAL.md records the measurement). Selecting it logs a warning.
# RECALL_MODEL_SHA256 is the SHA256 of the whole provisioned artifact tree. It is verified before
# anything loads, and a mismatch or a missing tree refuses startup. The BGE profiles read their
# tree from RECALL_MODEL_CACHE, the Qwen profile from RECALL_QWEN_MODEL_PATH.
#
# HOSTED profiles, served by a provider's API. They take an API key and NOTHING else: there is no
# artifact tree to point at and no bytes to hash, so RECALL_MODEL_CACHE and RECALL_MODEL_SHA256
# are not merely optional here, they are refused.
#   voyage-code-3-v1, voyage-3-v1                      RECALL_EMBEDDER=voyage,     VOYAGE_API_KEY
#   openai-text-embedding-3-small-v1                   RECALL_EMBEDDER=openai      OPENROUTER_API_KEY
#   openai-text-embedding-3-large-v1                     or =openrouter
#   gemini-embedding-001-v1
#
# ⚠️ A hosted profile is SERVABLE but not ATTESTABLE, and the difference is deliberate. The
# provider can replace the weights behind a stable model name, so nothing this process can reach
# proves which weights wrote the vectors it searches. check_enterprise_readiness() therefore
# REFUSES a hosted profile unless the operator passes allow_legacy_profile=True, which is the
# same explicit escape a legacy unpinned profile uses. Retrieval and calibration work normally;
# what you do not get is an attestation. The declared vector width IS checked, at construction,
# against what the endpoint actually returns, so a provider changing the width behind a model
# name fails startup instead of quietly filling a store built at the other width.
# Re-check the declared widths at any time with:
#   python scripts/measure_hosted_embedding_widths.py
RECALL_EMBED_PROFILE=
RECALL_MODEL_CACHE=
RECALL_MODEL_SHA256=
RECALL_QWEN_MODEL_PATH=

# Fixed service cost profile. Run separate processes for fast, quality, and code traffic; a client
# cannot select the expensive path per request. Leaving this unset keeps the pre-profile
# behaviour, in which the legacy RECALL_RERANK switch still decides reranking. Setting BOTH to
# values that contradict each other refuses STARTUP, not the first search.
#
# unset   = 20 candidates/leg, RECALL_RERANK decides reranking, k not clamped,
#           NO budget (nothing is ever shed on time, budget_exceeded is always false),
#           4 concurrent + 16 queued.  <-- this is what the line below selects as shipped
# fast    = 20 candidates/leg, no reranker,       returns 5,   250 ms budget, 8 concurrent + 32 queued
# quality = 20 candidates/leg, pinned reranker,   returns 5,  1500 ms budget, 2 concurrent +  8 queued
# code    = 20 candidates/leg, CoREB code rerank, returns 5, 10000 ms budget, 1 concurrent  +  2 queued
#
# The budget is enforced at the door: a request that cannot START within it is shed before the
# query is embedded. A request that queued and then ran fast is NOT reported over budget; the
# verdict is computed on the work the request itself did, because the budget is already spent as
# the admission timeout and charging it twice would label a fast retrieval slow.
#
# Each profile carries its OWN concurrency budget. The two overrides below are COMMENTED OUT
# rather than present-and-blank, deliberately: a blank value is read as unset by this release but
# as a malformed integer by earlier ones, so leaving them present would make a rollback fail.
RECALL_RETRIEVAL_PROFILE=
# RECALL_SEARCH_CONCURRENCY=
# RECALL_SEARCH_QUEUE=
# Quality profile local artifact settings. The PATH is deployment specific; only the DIGEST is
# enforced, and it must equal the value pinned in recall/rerank.py. The model name and Hub revision recorded next to it
# (cross-encoder/ms-marco-MiniLM-L-6-v2, c5ee24cb...) are provenance, not a runtime check: the
# quality profile loads from a local tree with local_files_only, where the revision is unused.
# The digest is a hash of that whole provisioned TREE, so it identifies one provisioned directory
# rather than the model in general; a differently laid out copy of the same weights hashes
# differently and is refused.
RECALL_RERANK_THREADS=1
RECALL_RERANK_PATH=
RECALL_RERANK_SHA256=
# Code reranking uses the bounded `code` profile. `coreb-code` expands to the pinned
# hq-bench/coreb-code-reranker revision and uses the Qwen yes/no causal-LM scoring path.
# RECALL_RETRIEVAL_PROFILE=code
# RECALL_RERANK_MODEL=coreb-code
# RECALL_RERANK_BATCH_SIZE=4
# RECALL_ACCEPT_REMOTE_MODEL_CODE=1

# Candidate shadow artifacts may differ from the active generation.
RECALL_SHADOW_MODEL_CACHE=
RECALL_SHADOW_MODEL_SHA256=
RECALL_SHADOW_QWEN_MODEL_PATH=

# Enables database backed tenant generation routing and fail closed readiness.
RECALL_ENTERPRISE_CONTROL_PLANE=0
```
