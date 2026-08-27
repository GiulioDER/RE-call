# Using RE-call with the Claude Agent SDK

`recall_agent` is the supported way to give a Claude Agent SDK (Python) application RE-call
memory without running an MCP server. The tools run in-process, the trust layer applies per
call, and the model-facing surface is identical to the MCP server's, so skills and prompts
written against `recall_search`/`recall_evidence` transfer unchanged.

```bash
pip install "recall-rag[agent,fastembed]"
```

`agent` brings the Claude Agent SDK; `fastembed` brings the default local embedder. Installing
`agent` alone succeeds and then fails on the first tool call, because resolving the default
embedder needs the fastembed extra. Swap it for `voyage` or `openai` if you serve a corpus built
with a hosted model, and note that the Claude Code CLI is a separate install the SDK expects to
find on PATH.

## Quickstart

```python
import anyio
from claude_agent_sdk import query

from recall_agent import RecallAgentMemory

async def main() -> None:
    with RecallAgentMemory(dsn="postgresql://recall:recall@localhost:5432/recall") as memory:
        options = memory.options(model="claude-sonnet-5")
        async for message in query(prompt="What do we already know about X?", options=options):
            print(message)

anyio.run(main)
```

`options()` assembles a `ClaudeAgentOptions` carrying the in-process server, the fully qualified
`allowed_tools`, and a `SessionStart` hook that injects a short memory digest. It merges with
anything you pass and never replaces it: your `mcp_servers` entries are added beside RE-call's,
your `allowed_tools` are appended, and your hooks are appended per event. A collision on
RE-call's own server name raises; pick a different `server_name` at construction instead.

## What the agent gets

| Tool | Backed by | Notes |
|---|---|---|
| `mcp__recall__recall_search` | `recall_mcp.service.search_memory` | Same description the MCP server publishes, drift-tested. |
| `mcp__recall__recall_evidence` | `recall_mcp.service.evidence_memory` | Citable bundle plus the exact prompt to answer with. |
| `mcp__recall__recall_index` | `recall_mcp.service.index_memory` | Only with `write_tools=True`. Confined to `RECALL_INDEX_ROOT`. |
| `mcp__recall__recall_forget` | `recall_mcp.service.forget_memory` | Only with `write_tools=True`. Right-to-erasure; irreversible. |

Write tools are off by default because an in-process server has no scope or auth layer: the MCP
server gates these operations behind authenticated scopes, and here the host application is the
authority, so it must opt in visibly. Three boundaries to know before you do:

- **`RECALL_INDEX_ROOT` defaults to the host process's working directory.** Set it to a directory
  you would be content for the model to read, because indexing something makes it retrievable
  through `recall_search`. Indexing is also refused outright under `RECALL_ENV=production`, which
  builds corpora from immutable manifests instead.
- **The model cannot widen the file scan.** `recall_index` takes a path and nothing else, matching
  the MCP server: the safe default scan excludes config and secret files (`tokens.json`,
  `credentials.json`, dotfiles), and no tool argument turns that off. If your host genuinely needs
  a different glob, that is a decision for the host, not for the model.
- **Erasure here does not reach a shadow generation or the migration outbox.** The MCP server
  passes a shadow store and a control plane to `forget_memory`; this surface has neither, so while
  a shadow migration is in flight for the tenant, erased text can survive in the shadow table or
  the outbox and be replayed back. Do not point the in-process write tools at a corpus that is
  mid-migration; use the server for erasure there.

## Trust semantics

The trust policy applies on every call, exactly as it does behind the MCP server:

- **Strict (the default)** raises `TrustRefusal` before any retrieval runs. The tool renders the
  refusal's wire form: `trust_state: "refused"`, the failure `code`, and `advice` (library-authored
  text the model is instructed to follow). No hits, no query echo. It is deliberately not marked
  as a tool error, because SDK clients may summarise or hide error results and the advice is the
  actionable part.
- **An empty result from a working gate** looks different by construction: `trust_state:
  "trusted"`, `hits: []`, `abstained` set. An outage and an absence of memory never share a shape.
- **Development mode** (`policy=TrustPolicy.development()`) stamps `trust_state: "degraded"` on
  every result. Pair it with an explicit `Calibration` when you need the verdict machinery on an
  uncalibrated corpus; without one, every hit degrades to `unverified`. Development mode is for
  finishing setup, never the answer to a refusal (see [OPERATING_MODES.md](OPERATING_MODES.md)).

## Configuration

Explicit arguments win; the environment is the fallback, and the variables are the same ones the
MCP server reads, so pointing both surfaces at one corpus needs no extra configuration:

| Setting | Falls back to |
|---|---|
| `dsn` | `RECALL_SERVING_DSN`, then `RECALL_DSN` |
| `embedder` | `RECALL_EMBEDDER` (an `Embedder` instance also passes through) |
| `policy` | `RECALL_TRUST_MODE` |
| `tenant` | `RECALL_TENANT` |
| `table` | `RECALL_TABLE` |

`use_generation_store=True` serves the generation-bound table and is a constructor decision,
deliberately not an environment variable: which store class answers is something the host should
see at the construction site. `RecallAgentMemory.from_env(env)` resolves everything from an
explicit mapping for testability.

The store and the embedder are both created lazily, on first use in a worker thread, so
constructing this object on an event loop does not stall it while a model loads. The store is
owned by the object (`close()` or the context manager releases it, and a closed memory refuses to
reopen); pass `store=` to bring your own, which then stays yours to close.

## Example

`examples/agent_sdk_memory.py` rebuilds the anti-re-litigation loop from
`examples/self_recall_agent.py` on top of `RecallAgentMemory`: consult memory before acting, back
off when a closed decision surfaces, and fail closed when the gate itself is unavailable.
