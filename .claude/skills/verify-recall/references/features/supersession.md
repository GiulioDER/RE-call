# Supersession

A memo that declares `supersedes: <older memo>` retires the older one: search keeps showing the
older memo, marked superseded and pointing at its replacement, and never serves it as the answer.

## Sub-features

- `supersession-cli`: the CLI row reads `superseded ... <old> -> use <new>`.
- `supersession-mcp`: the MCP hit carries `"superseded_by": "<new>"`.

## How to get to it (user POV)

Write the newer memo with `supersedes: cache_ttl_v1.md` in its front matter, index, then search for
the subject. The seed already holds two such pairs.

## Driving it with control_recall

- **CLI.** Run `search --run <run> --development "how long do pricing snapshot cache entries live"`.
  The `rows` hold `cache_ttl_v2.md` with `state` `ok` and `cache_ttl_v1.md` with `state`
  `superseded` and `use` `cache_ttl_v2.md`, even though v1 has the higher `cos` (about 0.52 against
  0.36 with `hashing`).
- **MCP.** Run `mcp --run <run> --development recall_search --args '{"query": "how long do pricing
  snapshot cache entries live"}'`. In `structured.hits`, `cache_ttl_v2.md` comes first and
  `cache_ttl_v1.md` has `superseded_by` `cache_ttl_v2.md`.

## Gotchas

- The older memo is the nearer lexical match on purpose: a plain vector index would answer from
  it. A proof must show v1 present AND marked, not merely absent.
- `hashing` cosines differ from a real model's; assert the states and the pointer, not the numbers.
