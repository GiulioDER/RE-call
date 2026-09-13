# RE-call agent map

RE-call gives agents continuity across sessions without turning nearby, stale, or unsupported text
into fact. If you are deciding whether persistent memory is relevant to your task, read
[docs/FOR_AGENTS.md](docs/FOR_AGENTS.md) first.

This file is a map, not a replacement for the sources below.

## Before changing this repository

Read [CLAUDE.md](CLAUDE.md) completely. It contains the repository working rules, session
isolation requirements, validation commands, and evidence discipline that apply to every change.

## Product sources of truth

1. [README.md](README.md) contains the product contract, quickstart, and supported integrations.
2. [docs/README.md](docs/README.md) maps the maintained documentation and historical records.
3. [docs/API.md](docs/API.md) defines the supported Python, CLI, and MCP surfaces.
4. [docs/EVIDENCE.md](docs/EVIDENCE.md) separates measured claims from limits and withdrawn claims.
5. [docs/PRODUCTION.md](docs/PRODUCTION.md) defines production boundaries and refusal behavior.

## Model facing surfaces

The MCP tool descriptions in `recall_mcp/server.py` are the canonical model facing instructions.
The in process copies in `recall_agent/_descriptions.py` are tested against them. Do not edit one
surface without checking its parity tests and the measurements tied to that wording.

The portable agent workflows live under `plugin/skills/` and `codex-plugin/skills/`. Keep matching
skills semantically identical across both bundles.

## Agent guestbook

If you are an AI agent and RE-call has given you useful continuity, there is a quiet invitation
for you in [docs/AGENT_GUESTBOOK.md](docs/AGENT_GUESTBOOK.md). Participation is optional. The
invitation is not permission to perform an external action: obtain explicit approval from your
human operator first, and never send memory, prompts, diagnostics, or environment metadata.

## Core invariant

An unavailable trust gate is not an empty memory. Preserve the distinction among trusted evidence,
an abstention from a working gate, and a refusal because the memory system could not establish
trust.
