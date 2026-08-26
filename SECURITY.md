# Security Policy

RE-call protects the memory retrieval boundary. It does not redact, encrypt, or classify the corpus.
The main asset is the text you index.

## Supported Versions

Pre-1.0, only the current `0.x` line gets fixes. There is no LTS branch.

| Version | Supported |
|---|---|
| Current 0.x release | Yes |
| Older 0.x releases | Best effort only |

## Report a Vulnerability

Open a private security advisory on GitHub, or contact the maintainer through the repository owner
profile. Do not paste private corpus text, credentials, tokens, DSNs, or API keys into a public
issue.

Useful reports include:

| Area | Report when |
|---|---|
| Tenant isolation | One tenant can read another tenant's rows through the library, MCP server, or integrations. |
| Index confinement | `recall_index` can read files outside the configured index root or outside the allowed glob. |
| Credential handling | A DSN, token, or API key is logged or returned in an exception, tool response, or result. |
| Cloud egress | A cloud embedder is reached without the caller explicitly selecting it. |
| Trust refusal | Strict mode returns corpus text when calibration or generation identity is not certified. |
| Instruction channel | Corpus-controlled text reaches a field a model is told to obey (`SearchResult.advice`), or an adapter hands an agent a hit whose verdict is not `ok`. This is the prompt-injection surface; see the threat model's "Retrieved memory reaches an instruction channel". |

Expected behavior that is not a vulnerability:

| Behavior | Why |
|---|---|
| Retrieved chunks contain the original memory text | Returning memory verbatim is the purpose of the library. |
| A tenant principal can read all chunks for its own tenant | Isolation is tenant-level, not per chunk. |
| Voyage embeddings send chunk text to Voyage | That is the explicit cloud embedder path. Use the local FastEmbed path for private corpora. |
| Local Docker credentials are `recall:recall` | They are for local development only; production DSNs must use different credentials. |

## Deployment Rules

- Run application traffic with an unprivileged serving role.
- Keep the migration DSN out of the MCP process.
- Set `RECALL_INDEX_ROOT` to a directory that contains only indexable corpus files.
- Keep token files and `.env` files outside the index root.
- Use local embeddings for sensitive corpora unless cloud egress has been approved.
- Treat calibration artifacts as tenant data, because they can reveal corpus topics and evaluation
  intent.

Detailed threat model: [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md).
