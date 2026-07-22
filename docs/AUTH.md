# Authentication and multi-tenancy

RE-call serves two transports with deliberately different security postures.

| Transport | Listener | Auth | Tenancy |
|---|---|---|---|
| `stdio` (default) | none — a private pipe to one client | not required | one tenant, `RECALL_TENANT` |
| `streamable-http`, `sse` | TCP socket | **required, enforced at startup** | one tenant per token |

`stdio` needs no authentication because there is no remote caller: the client owns the process and
the pipe *is* the boundary. The HTTP transports open a socket, and starting one without tokens
**raises `AuthConfigError` and refuses to boot**. That is a deliberate choice over logging a
warning — a warning produces a server that comes up looking healthy with every memory in it
readable by anything that can reach the port, and the warning is found afterwards.

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
chmod 600 tokens.json     # the server warns if this is readable by group or other
```

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
| `recall:read` | `recall_search`, `recall_stats` | |
| `recall:write` | `recall_index` | Indexing burns embedding spend — with a paid embedder that is real money. |
| `recall:forget` | `recall_forget` | Deletion is irreversible. |

A principal holding only `recall:read` gets a `PermissionError` from `recall_index`, and the
denial is logged with the principal name.

## Running it

```bash
export RECALL_TRANSPORT=streamable-http
export RECALL_AUTH_TOKENS_FILE=/etc/recall/tokens.json
export RECALL_AUTH_ISSUER_URL=https://recall.example.com
export RECALL_AUTH_RESOURCE_URL=https://recall.example.com
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

## How tenant isolation actually works

A `PgVectorStore` is bound to one tenant for its lifetime: the pool's `configure` hook sets
`recall.tenant_id` on every connection, and the row-level-security policy compares each row
against that GUC. The tenant is a property of the *connection*, not of the query — which is what
makes isolation hold even if a `WHERE tenant_id = …` predicate is ever forgotten.

So the server keeps **one pool per tenant** (`recall_mcp/stores.py`). Serving two tenants from one
pool would mean re-setting the GUC per request on a shared connection, and that is precisely how
cross-tenant leaks happen: a connection returned to the pool mid-request, or an exception between
`set_config` and the query, and the next caller inherits someone else's tenant.

Pools are created on first use and bounded by *configuration*, not traffic: a tenant exists only
if an operator provisioned a token for it, so the worst case is
`len(tenants) × RECALL_POOL_SIZE` connections — a number you can compute at startup and check
against the server's `max_connections`. It is logged when the server starts.

> **RLS does not apply to superusers.** A role with `SUPERUSER` or `BYPASSRLS` ignores the policy
> entirely, leaving only the query predicates. The server checks this at startup and warns.
> Connect as an unprivileged role.

## Limits of this scheme

Stated plainly, because the alternative is discovering them in production:

- **No revocation without a restart.** The token file is read at startup.
- **No rotation protocol.** Overlapping validity is manual: add the new token, restart, migrate
  clients, remove the old one, restart again.
- **Bearer means bearer.** A leaked token grants that principal's access until it is removed.
  There is no proof-of-possession and no audience binding.
- **`token_sha256` bypasses the length floor.** Length cannot be recovered from a hash, so a
  digest-provisioned token is accepted however weak the plaintext was. Generate tokens with
  `secrets.token_urlsafe(32)` and the point is moot; the trade is that plaintext never touches
  disk.
- **No rate limiting or per-tenant quota.** `recall_index` has hard file-count and byte caps
  (`recall_mcp/service.py`), but there is no limit on call *frequency*.

If you need any of those, put a real identity provider in front and supply the MCP SDK's
`auth_server_provider` in place of the static verifier. This module is intended for the case it
handles honestly: a small number of machine principals provisioned out of band.
