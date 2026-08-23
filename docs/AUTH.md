# Authentication and multi-tenancy

RE-call serves two transports with deliberately different security postures.

| Transport | Listener | Auth | Tenancy |
|---|---|---|---|
| `stdio` (default) | none — a private pipe to one client | not required | one tenant, `RECALL_TENANT` |
| `streamable-http`, `sse` | TCP socket | **required, enforced at startup** | one tenant per token |

`stdio` needs no authentication because there is no remote caller: the client owns the process and
the pipe *is* the boundary. The HTTP transports open a socket, and starting one with no
authentication configured at all **raises `AuthConfigError` and refuses to boot**. That is a
deliberate choice over logging a warning — a warning produces a server that comes up looking
healthy with every memory in it readable by anything that can reach the port, and the warning is
found afterwards.

There are two ways to configure it, and exactly one may be active:

| Mechanism | Set | Use |
|---|---|---|
| **Static token file** | `RECALL_AUTH_TOKENS_FILE` | Development. Refused when `RECALL_ENV=production`. |
| **OIDC provider** | `RECALL_OIDC_ISSUER` + `RECALL_OIDC_AUDIENCE` + `RECALL_OIDC_TENANTS`, **and** one of `RECALL_OIDC_SUBJECT_TENANTS` / `RECALL_OIDC_TRUST_TENANT_CLAIM` | Production. See [below](#taking-identity-from-an-oidc-provider). |

## Provisioning tokens

Tokens come from a **file**, named by `RECALL_AUTH_TOKENS_FILE`. There is no environment variable
that accepts a raw token, and that omission is intentional: environment variables leak through
`/proc/<pid>/environ`, `ps e`, container inspection APIs, crash dumps, and every child process the
server spawns. A file is also what Kubernetes and Docker mount for secrets anyway, and unlike an
env var it can be permission-checked.

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

```json
{
  "principals": [
    {
      "name": "research-agent",
      "token": "PASTE_THE_GENERATED_TOKEN_HERE",
      "tenant": "team-research",
      "scopes": ["recall:read", "recall:write"]
    },
    {
      "name": "retention-job",
      "token_sha256": "REPLACE_WITH_THE_SHA256_OF_A_32_CHAR_TOKEN",
      "tenant": "team-research",
      "scopes": ["recall:forget"],
      "expires_at": "2027-01-01T00:00:00Z"
    }
  ]
}
```

```bash
sudo install -o "$USER" -m 600 tokens.json /etc/recall/tokens.json
export RECALL_AUTH_TOKENS_FILE=/etc/recall/tokens.json
```

> ⚠️ **Keep the token file outside `RECALL_INDEX_ROOT`**, which defaults to the server's working
> directory. `recall_index` reads files as the server's own user, so a token file sitting in the
> index root is a file an authenticated principal can ask the server to index and then read back
> through `recall_search` — other tenants' credentials included. `chmod 600` does not help against
> that, because the server is the owner. Indexing now refuses anything the corpus glob excludes,
> so `tokens.json` is no longer reachable that way; keeping it on a different path means you are
> not relying on that one check. Prefer `token_sha256` over `token` for the same reason: then
> there is no recoverable credential on disk at all.

`token_sha256` accepts a precomputed digest, so an operator provisioning access never has to write
a live credential to disk in recoverable form:

```bash
python -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$TOKEN"
```

### Fields

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Identifies the principal in logs. Must be unique. |
| `token` *or* `token_sha256` | exactly one | Plaintext must be ≥ 32 characters. The floor is not configurable. |
| `tenant` | no (defaults to `default`) | **The isolation boundary.** Selects the store, and therefore the RLS namespace. |
| `scopes` | no (defaults to `["recall:read"]`) | Least privilege by default. |
| `expires_at` | no | ISO-8601 **with an offset**. A naive timestamp is refused at startup, not at first request. |

Anything malformed refuses to start. Nothing is skipped — a skipped entry is either a principal
that cannot authenticate, or worse, one that authenticates into the wrong tenant.

## Scopes

Scopes mirror the risk each tool actually carries, rather than collapsing into read/write:

| Scope | Tools | Why separate |
|---|---|---|
| `recall:read` | `recall_search`, `recall_evidence`, `recall_stats` | |
| `recall:write` | `recall_index`, `recall_ingest`, `recall_calibration_run` | Indexing burns embedding spend — with a paid embedder that is real money. |
| `recall:forget` | `recall_forget` | Deletion is irreversible. |
| `recall:admin` | `recall_calibration_publish` | Publication changes the serve/abstain decision for the whole tenant. Deliberately NOT implied by write: a principal that may add memory should not be able to change what every query trusts. |

A principal holding only `recall:read` gets a `PermissionError` from `recall_index`, and the
denial is logged with the principal name.

Since 0.10.0, publishing a calibration requires `recall:admin`; a write token that published
before must be re-provisioned with the extra scope (static tokens: add it to `scopes` in the
token file; OIDC: grant it in the identity provider's role). The full tenant inventory from
`recall_tenants` is also admin-only now — a plain read principal sees only its own tenant.

## Running it

```bash
export RECALL_TRANSPORT=streamable-http
export RECALL_AUTH_TOKENS_FILE=/etc/recall/tokens.json
export RECALL_AUTH_ISSUER_URL=https://recall.example.com
export RECALL_AUTH_RESOURCE_URL=https://recall.example.com
export RECALL_HOST=127.0.0.1        # default; set 0.0.0.0 deliberately, never by inheritance
export RECALL_PORT=8000
python -m recall_mcp.server
```

The two URLs are published in the server's protected-resource metadata so a client knows where to
get a token and which audience it is for. When you provision tokens by hand, set both to the
server's own public URL.

Clients send the token as an ordinary bearer credential:

```
Authorization: Bearer <token>
```

**Terminate TLS in front of this server.** A bearer token over plaintext HTTP is readable by every
hop in between.

## Taking identity from an OIDC provider

The token file is a development affordance, and `load_token_registry` **refuses to load it when
`RECALL_ENV=production`**. In production, identity comes from an external provider instead, so
revocation, rotation and expiry belong to the IdP rather than to whoever remembers to edit a file.

```bash
export RECALL_TRANSPORT=streamable-http
export RECALL_OIDC_ISSUER=https://idp.example.com
export RECALL_OIDC_AUDIENCE=recall-prod       # the audience this server accepts
export RECALL_OIDC_TENANTS=acme,globex        # the tenants this deployment serves
export RECALL_OIDC_SUBJECT_TENANTS=alice@corp:acme,svc-etl:globex   # who may name which tenant
                                              # (or RECALL_OIDC_TRUST_TENANT_CLAIM=1; see below)
export RECALL_AUTH_RESOURCE_URL=https://recall.example.com
python -m recall_mcp.server
```

`RECALL_OIDC_ALGORITHMS` is optional and defaults to the asymmetric set (`RS256`, `RS384`, `RS512`,
`ES256`, `ES384`, `PS256`, `PS384`, `PS512`). Symmetric and unsigned algorithms are refused at
construction rather than merely omitted: with a published verification key, an HMAC algorithm *is*
the algorithm-confusion attack. An algorithm outside the serviceable set is also refused at boot,
because the JWKS loader serves RSA and EC keys only, and an unserviceable entry would otherwise
boot cleanly and then reject every token.

### The token this server expects

| Claim | Required | Notes |
|---|---|---|
| `iss` | yes | Must equal `RECALL_OIDC_ISSUER`. |
| `aud` | yes | Must contain `RECALL_OIDC_AUDIENCE`. |
| `exp` | yes | A token with no expiry never stops being valid, so absence is refused. |
| `tenant` | yes | **The isolation boundary.** Must be in `RECALL_OIDC_TENANTS`. Leading or trailing whitespace is refused, not trimmed. |
| `scope` | no | Space-delimited string or JSON list. Unrecognised scopes are **dropped**, not refused, so a token with none gets a principal that can do nothing. |
| `azp` | conditional | Required to name us when `aud` carries several audiences (OIDC Core 3.1.3.7). |
| `sub` | conditional | **Required whenever `RECALL_OIDC_SUBJECT_TENANTS` is set** — it is the binding's lookup key, and its absence is refused with `missing_subject`. Under `RECALL_OIDC_TRUST_TENANT_CLAIM` it is optional and only names the principal in logs, falling back to the tenant. |

`RECALL_AUTH_ISSUER_URL` is **not** required here: it defaults to `RECALL_OIDC_ISSUER`, because
with a provider there is exactly one right answer and restating it is a chance to state it
differently, at which point clients are sent to a provider that did not sign the tokens this
server accepts.

### Migrating from the static token file

Setting `RECALL_OIDC_ISSUER` and `RECALL_AUTH_TOKENS_FILE` together refuses to boot **unless you
declare which is active**, because two trust models with no stated precedence means one of them is
sitting in the configuration looking effective. Declaring it is what makes a staged cutover
possible:

```bash
RECALL_AUTH_MODE=static   # step 1: add every OIDC variable, change nothing. Verify.
RECALL_AUTH_MODE=oidc     # step 2: flip one variable. Rollback is flipping it back.
                          # step 3: remove RECALL_AUTH_TOKENS_FILE at leisure, then the mode.
```

The OIDC block is **validated** whenever it is present, even under `mode=static`, so step 1 is a
real rehearsal: a malformed subject binding, an unserviceable algorithm or a missing decision fails
there rather than at the flip. Only the *token file* is conditionally loaded, because loading it has
a side effect that must not happen while it is standing down (see production, below).

Two limits, stated because discovering them mid-cutover is the expensive way:

- **`RECALL_ENV=production` cannot run step 1.** Production refuses the static token file outright,
  and under `mode=static` that file is the active mechanism, so the rehearsal step is refused on
  every transport including stdio. A production host still swaps in one revision; stage the cutover
  somewhere the token file is allowed, and treat production as the flip alone.
- **Rollback needs two things kept.** Flipping `mode` back to `static` is a rollback only while the
  token file still exists, so do not remove it until you have renounced rollback. Set
  `RECALL_AUTH_ISSUER_URL` explicitly for the duration too, rather than leaning on the OIDC default;
  it is defaulted from the OIDC block whenever one is present, but an explicit value is one less
  thing that changes meaning when the mode does.

Without `RECALL_AUTH_MODE`, both-set still refuses. The guard was never wrong, only too broad.

### Why the tenant list is mandatory

`RECALL_OIDC_TENANTS` has no default and absent does not mean "every tenant". The IdP vouches for
*identity*; it knows nothing about this deployment's topology. A token reading `tenant: initech`
is not forged merely because initech was never provisioned here — but admitting it would open a
store for a namespace no operator configured, which is RE-call creating tenants on the say-so of a
third party. A token naming an unlisted tenant is refused with reason `tenant_not_allowed`.

### Who may name a tenant: one of these two is required

The allowlist bounds **which** tenants exist. It does not bind **who** may name one: nothing
correlates `sub` with `tenant`, so on its own, any subject that can obtain a token from your issuer
with `aud = RECALL_OIDC_AUDIENCE` reaches whichever provisioned tenant its claim names.

The server therefore **refuses to boot** unless you answer the question one way or the other.

**Either pin the mapping here.** A subject reaching a tenant it is not bound to is refused with
`subject_tenant_mismatch`, even though the tenant is provisioned and the token is genuine:

```bash
RECALL_OIDC_SUBJECT_TENANTS=alice@corp:acme,svc-etl:globex
```

Repeat a subject to give it several tenants (`svc-etl:acme,svc-etl:globex`). A subject absent from
the map is refused with `subject_not_bound`, and with a binding configured a token carrying no
`sub` is refused outright, because `sub` is the lookup key.

**Or declare that your IdP is authoritative for the claim:**

```bash
RECALL_OIDC_TRUST_TENANT_CLAIM=1
```

This is a legitimate and common deployment, and it means exactly one thing: **the `tenant` claim is
minted by the IdP from an authoritative subject-to-organisation mapping.** If it can be set from a
user-editable profile attribute, a client-requested claim, or a self-service app registration, then
it is caller-controlled, and a caller-controlled claim selecting the RLS namespace is a cross-tenant
read. The `azp` check covers only the multi-audience case; a same-audience token from another
subject of the same IdP is indistinguishable from a legitimate one.

Setting both refuses to boot: they answer the same question differently, and whichever won, the
other would sit in the configuration looking effective.

The binding is checked **after** signature verification, for the same reason the tenant allowlist
is: ahead of it, a forged token would answer "is this a known subject here?" for a caller holding
no credential.

The **allowlist** check runs after signature verification for the same reason. Ahead of it, the
reply would differ between a forged token naming a real tenant (`bad_signature`) and one naming an
unknown tenant (`tenant_not_allowed`), which enumerates your tenant list to a caller holding no
credential at all.

### Rejection reasons

Every refusal carries a stable `reason` slug, and they are logged rather than collapsed into
"invalid token", because `bad_signature` and `unknown_kid` are two different events: a forgery
attempt, and a key rotation this process has not caught up with yet.

`IdentityProviderUnavailable` is logged distinctly from a token rejection. The MCP SDK's
`verify_token` can only return a token or `None`, so the 401/503 distinction cannot survive that
hop — but "the IdP is unreachable" and "somebody is forging tokens" call for opposite responses,
so it survives into the log.

## How tenant isolation actually works

A `PgVectorStore` is bound to one tenant for its lifetime: the pool's `configure` hook sets
`recall.tenant_id` on every connection, and the row-level-security policy compares each row
against that GUC. The tenant is a property of the *connection*, not of the query — which is what
makes isolation hold even if a `WHERE tenant_id = …` predicate is ever forgotten.

So the server keeps **one pool per tenant** (`recall_mcp/stores.py`). Serving two tenants from one
pool would mean re-setting the GUC per request on a shared connection, and that is precisely how
cross-tenant leaks happen: a connection returned to the pool mid-request, or an exception between
`set_config` and the query, and the next caller inherits someone else's tenant.

Stores are created on first use, and a tenant exists only if an operator provisioned it: a token
for it in the static file, or an entry in `RECALL_OIDC_TENANTS`. Nothing a caller sends can add
one.

The connection ceiling is `RECALL_POOL_SIZE` for the whole process, **independent of how many
tenants are provisioned**, because the tenants share one pool (`StoreRegistry.max_connections`).
It used to be `len(tenants) × RECALL_POOL_SIZE`, which had to be re-checked against the server's
`max_connections` every time somebody was onboarded; a constant is what makes a thousand tenants a
configuration question rather than a capacity one. The figure is logged at startup.

> **RLS does not apply to superusers.** A role with `SUPERUSER` or `BYPASSRLS` ignores the policy
> entirely, leaving only the query predicates. The server checks this at startup and warns.
> Connect as an unprivileged role.

## Limits of the static token scheme

Stated plainly, because the alternative is discovering them in production. Every one of these is a
property of the **token file**, and the OIDC path above exists because of them:

- **No revocation without a restart.** The token file is read at startup.
- **No rotation protocol.** Overlapping validity is manual: add the new token, restart, migrate
  clients, remove the old one, restart again.
- **Bearer means bearer.** A leaked token grants that principal's access until it is removed.
  There is no proof-of-possession and no audience binding.
- **`token_sha256` bypasses the length floor.** Length cannot be recovered from a hash, so a
  digest-provisioned token is accepted however weak the plaintext was. Generate tokens with
  `secrets.token_urlsafe(32)` and the point is moot; the trade is that plaintext never touches
  disk.
- **Rate limits are per process.** Per-tenant call budgets and an indexing byte quota do ship
  (`recall_mcp/limits.py`, tuned with `RECALL_RATE_*_PER_MIN` and `RECALL_INDEX_BYTES_PER_HOUR`;
  see [SECURITY_MODEL.md](SECURITY_MODEL.md)), but the buckets are in-memory and unshared, so N worker
  processes admit roughly N times each rate. A fleet needs a shared limiter.

Against [the OIDC path](#taking-identity-from-an-oidc-provider), item by item, since the split is
not a clean prefix:

- **Revocation** and **rotation** are answered. Both belong to the IdP, and a key roll is a JWKS
  refresh the server picks up on its own.
- **Bearer means bearer** is *partly* answered. OIDC adds audience binding (`aud`, plus the `azp`
  check on multi-audience tokens) and an enforced `exp` carried onto the principal. It does not add
  proof-of-possession: a stolen JWT still works until it expires.
- **The `token_sha256` length floor** does not exist under OIDC — there is no token file.
- **Per-process rate limits** are unchanged. That one is orthogonal to identity.

The static file is intended for the case it handles honestly: a small number of machine principals
provisioned out of band, in development. `RECALL_ENV=production` refuses it outright.
