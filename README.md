<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call: memory that knows when not to guess" width="900">
</p>

<!-- mcp-name: io.github.GiulioDER/re-call -->

<p align="center">
  <b>Memory that abstains when nothing clears its threshold.</b>
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/recall-rag/"><img src="https://img.shields.io/pypi/v/recall-rag" alt="PyPI version"></a>
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="License: AGPL-3.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <a href="docs/LITE.md"><img src="https://img.shields.io/badge/Lite-SQLite%2C%20no%20server-003B57" alt="Lite: SQLite, no server"></a>
</p>

RE-call is hybrid retrieval with rank fusion (dense vectors plus full-text search, fused by RRF),
on SQLite or Postgres. On top of it: atomic rescue, which recovers a passage a single sentence
points to, and a semantic graph built from the links in your memos. Around it sits a trust layer.
It abstains below a threshold calibrated on your own store, ranks a superseded memo below its
successor, flags memos that have expired or are not yet valid, and erases what you forget.

## Quickstart

```bash
python -m pip install "recall-rag[fastembed,mcp]>=0.15.0"
recall setup --lite --memory ./notes
```

Point it at a folder of Markdown notes. Setup indexes them into `.recall/memory.db`, tries to
calibrate on them, and registers Claude Code for this project. Restart Claude Code and ask about
something your notes answer.

No Docker, no database server, no API key. The default embedder runs on your machine after a
one-time model download. A folder too small to calibrate refuses trusted answers until it grows.

[Getting started](docs/GETTING_STARTED.md) covers Codex, other MCP clients, PostgreSQL, the Agent
SDK and LangChain.

## What changes for your agent

- Every search comes back trusted, degraded or abstained, with the reason.
- A memo that declares `supersedes:` outranks the memo it replaces.
- A memo past its validity date comes back flagged as expired.
- On Lite, forget zeroes the deleted bytes in the store.
- Storing a memory makes no LLM call.
- Claude Code and Codex read and write the same store.

Claude Code loads the first 200 <!--@ citation-pending: Claude Code's documented auto memory load limit (code.claude.com/docs/en/memory, read 2026-11-14), not a RE-call measurement --> lines of its own memory index. RE-call searches every memo.

## Lite or PostgreSQL

| Choose | Best fit |
|---|---|
| **Lite** | One project on one machine, in a local SQLite file. |
| **PostgreSQL** | Shared serving, immutable generations, several tenants, or authentication. |

## Read next

| Need | Document |
|---|---|
| Install, integrations, the PostgreSQL demo | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Architecture and optional modules | [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) |
| Lite in depth | [docs/LITE.md](docs/LITE.md) |
| Python, CLI and MCP reference | [docs/API.md](docs/API.md) |
| Measurements and limits | [docs/EVIDENCE.md](docs/EVIDENCE.md) |
| Why an agent needs trusted memory | [docs/FOR_AGENTS.md](docs/FOR_AGENTS.md) |
| Everything else | [docs/README.md](docs/README.md) |

Published numbers are tied to committed artifacts, and the claim gate checks them in CI.

## Citation

Cite the project and credit Giulio D'Erme. [CITATION.cff](CITATION.cff) is the canonical source.

## License

Open source under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). For a closed
source product or a proprietary hosted service, a commercial license is available: see
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Preserve [NOTICE](NOTICE) when redistributing.
Contributions are accepted under the [Contributor License Agreement](CLA.md).
