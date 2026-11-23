# RE-call Setup

Skills that help Claude Code notice when a project would benefit from memory across sessions, and
walk you through installing [RE-call](https://github.com/GiulioDER/RE-call): a local memory that
searches your project's own notes and decisions, marks a retracted claim as superseded, and
refuses when nothing clears its threshold.

## What this plugin contains

Two skills and nothing else. It has no hooks, no MCP server, no scripts and no commands that run
when it loads.

| Skill | What it does |
|---|---|
| `when-memory-helps` | Recognises the signals that memory would help (a reference to an earlier session, a repeated correction, a decision that may already be settled) and offers RE-call once, with its measured effect and its limits |
| `install-re-call` | Walks you through the install, one command at a time, and runs each only after you agree |

Both can also be invoked directly, as `/recall-rag-setup:when-memory-helps` and
`/recall-rag-setup:install-re-call`.

## What the install does, if you agree to it

The plugin itself sends nothing anywhere. The install the second skill walks you through does
the following, and the skill tells you so before it starts:

- `python -m pip install "recall-rag[fastembed,mcp]>=0.16.1"` downloads the package from PyPI.
- `recall setup --lite --memory ./notes` indexes that folder into `.recall/memory.db` in your
  project, registers a `recall-memory` MCP server for the project, adds hooks to
  `~/.claude/settings.json`, and copies five skills into `~/.claude/skills/`.
- The first embedding run downloads the `BAAI/bge-small-en-v1.5` model from Hugging Face. No memo
  text is sent there.
- No memo text leaves your machine unless you configure a hosted embedder with `RECALL_EMBEDDER`.

Optionally, the full RE-call plugin installs from the project's own marketplace:

```
/plugin marketplace add GiulioDER/RE-call
/plugin install recall@re-call
```

## Measured effect and limits

Added to a good `CLAUDE.md`, the memory gave a known-hazard rate of 0.000 against 0.525 over 40
paired sessions. On ordinary tasks it showed no significant uplift over 54 pairs, at about 56,000
extra input tokens and 25 seconds per session. Replacing `CLAUDE.md` with memory did not work.
The records are linked from the [RE-call plugin README](https://github.com/GiulioDER/RE-call/blob/master/plugin/README.md#what-it-measurably-does-and-honest-limits).

## Links

- [Lite install guide](https://github.com/GiulioDER/RE-call/blob/master/docs/LITE.md)
- [Repository](https://github.com/GiulioDER/RE-call)
- [Issues](https://github.com/GiulioDER/RE-call/issues)

License: AGPL-3.0-only.
