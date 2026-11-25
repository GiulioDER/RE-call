# MCP server

The MCP server is what an agent calls: it serves RE-call's tools over stdio (or HTTP) against the
configured store, and answers with structured JSON.

## Sub-features

- `mcp-handshake`: initialize and list tools.
- `mcp-search`: `recall_search`, strict refusal and development answer.

## How to get to it (user POV)

A client launches `python -m recall_mcp.server` (or `recall-mcp`) with `RECALL_SERVING_DSN`,
`RECALL_EMBEDDER` and `RECALL_INDEX_ROOT` set, and calls tools over stdio.

## Driving it with control_recall

- **Handshake.** Run `mcp --run <run> recall_search --args '{"query": "how long do pricing
  snapshot cache entries live"}'`. `tools` lists 26 names on 2026-11-25, `recall_search` among them.
- **Strict refusal.** In the same call, `is_error` is true and `text[0]` holds a JSON refusal with
  `"error": "trust_refusal"` and a `code` such as `CALIBRATION_UNCERTIFIED` or `CALIBRATION_STALE`.
- **Development answer.** Add `--development`. `is_error` is false, `structured.trust_state` is
  `degraded`, and `structured.hits` lists results with `source`, `score`, `confidence`,
  `superseded_by`.

## Gotchas

- The tool count changes as tools are added; assert the tool you need is present, not the count.
- The server needs the `mcp` extra; without it `control_recall mcp` says which install to run.
- One call starts a fresh server (about 5 to 20 seconds); there is no long-lived instance to share.
