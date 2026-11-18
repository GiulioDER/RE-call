<p align="center">
  <picture>
    <source media="(prefers-reduced-motion: no-preference)" srcset="docs/banner.gif">
    <source media="(prefers-reduced-motion: reduce)" srcset="docs/banner.png">
    <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call. One memory, every agent." width="900">
  </picture>
</p>

<!-- mcp-name: io.github.GiulioDER/re-call -->

**RE-call is persistent memory for agents, built on hybrid retrieval with rank fusion.**
On SQLite or PostgreSQL, it combines dense vectors and full-text search using reciprocal rank
fusion (RRF).

**Atomic rescue** can recover a passage when a single sentence matches the query.
A **semantic graph** maps the links between your memos.

**The trust layer** abstains when no result clears a threshold calibrated on your own store.
It ranks superseded memos below their successors, flags memos that have expired or are not yet
valid, and lets you erase memos from the store.

[Benchmarks and limits](https://github.com/GiulioDER/RE-call/blob/master/docs/EVIDENCE.md) ·
[Security policy](https://github.com/GiulioDER/RE-call/blob/master/SECURITY.md) ·
[For agents](https://github.com/GiulioDER/RE-call/blob/master/docs/FOR_AGENTS.md)

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

## What a search returns

Real `recall_search` responses from two certified corpora, returned on 2026-10-20: RE-call's own
source code, and the author's working memory. They are abridged: fields and text are cut, never
edited, and chunk ids are shortened.

**An answer, with the chunks it may cite.**

```json
recall_search {"query": "where is reciprocal rank fusion implemented?", "k": 3}

{"decision_state": "supported", "abstained": false,
 "calibration_status": "certified", "trust_state": "trusted",
 "hits": [
  {"chunk_id": "b52539f1ef1f", "source": "recall/retriever.py", "score": 0.7776,
   "confidence": 0.9984, "verdict": "ok", "superseded_by": null,
   "text": "def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]: ..."},
  {"chunk_id": "69f9573b0bde", "source": "benchmarks/mtrag/multiquery.py", "score": 0.7712,
   "confidence": 0.9977, "verdict": "ok"},
  {"chunk_id": "68806c06538a", "source": "scripts/agent_ab_fusion_frontier.py", "score": 0.7665,
   "confidence": 0.997, "verdict": "ok"}],
 "evidence": {"decision": "answer",
  "citable": ["b52539f1ef1f", "69f9573b0bde", "68806c06538a"]}}
```

Every hit carries a calibrated confidence and a verdict. `evidence.citable` lists exactly what an
answer may cite.

**An abstention, on a question the corpus cannot answer.**

```json
recall_search {"query": "how does the Kubernetes operator autoscale the GPU inference pods?", "k": 3}

{"decision_state": "corpus_gap", "abstained": true,
 "reason": "no hit above the calibrated confidence threshold (probable corpus gap)",
 "calibration_status": "certified",
 "hits": [
  {"source": "benchmarks/mtrag/rerank_offload.py", "score": 0.5628, "confidence": 0.0036,
   "verdict": "low_confidence"},
  {"source": "recall/index.py", "score": 0.4139, "confidence": 0.0, "verdict": "low_confidence"},
  {"source": "recall/ops/__init__.py", "score": 0.5553, "confidence": 0.0023,
   "verdict": "low_confidence"}],
 "evidence": {"decision": "abstain", "reason_code": "corpus_gap", "citable": []}}
```

Plain vector search would answer from the first file. RE-call shows the candidates, says why none
qualifies, and offers nothing to cite.

**A superseded note, kept out of the answer.**

```json
recall_search {"query": "does the 0.12.0 release candidate include the first class Codex integration from PR 583?", "k": 3}

{"decision_state": "supported", "abstained": false,
 "calibration_status": "certified", "trust_state": "trusted",
 "hits": [
  {"chunk_id": "5781fb607b3d", "source": "recall/2026-09-02-release-0.12.0-published.md",
   "score": 0.4695, "confidence": 0.8525, "verdict": "ok", "superseded_by": null,
   "text": "The release changelog covers ... and Codex integration from PR #583. ..."},
  {"chunk_id": "9f245a7a6959", "source": "recall/2026-09-02-release-preparation-0.12.0-pr583.md",
   "score": 0.5838, "confidence": 0.9941, "verdict": "superseded",
   "superseded_by": "recall/2026-09-02-release-0.12.0-published.md",
   "text": "RE-call PR #583 ... merged into master on 2026-09-02 ... the candidate now includes PR #583's merged changes as well."},
  {"chunk_id": "00f4a94bab54", "source": "recall/project_index.md",
   "score": 0.4081, "confidence": 0.486, "verdict": "low_confidence"}],
 "evidence": {"decision": "answer", "citable": ["5781fb607b3d"]},
 "historical": {"candidates": ["9f245a7a6959"], "scope": "historical_only",
  "requires_validity_check": true}}
```

The later note declares that it supersedes the earlier one. The stale note is the closest match,
so similarity alone would answer from it. RE-call names its successor, keeps it out of
`evidence.citable`, and offers it only as history.

**A bug fixed from its symptom.**

```json
recall_search {"query": "my long background job on the server died when I closed the SSH session, even though I started it with nohup and &", "k": 3}

{"decision_state": "supported", "abstained": false,
 "calibration_status": "certified", "trust_state": "trusted",
 "hits": [
  {"chunk_id": "00192d073e84", "source": "recall/detached-runs-need-setsid.md",
   "score": 0.6413, "confidence": 0.9989, "verdict": "ok",
   "text": "A long run ... launched as `nohup bash script.sh > log 2>&1 &` died silently at row 29 of 500 when the SSH session closed. ... Relaunched with `setsid nohup ... < /dev/null &` it ran to completion. ..."},
  {"chunk_id": "c73d42e81759", "source": "recall/detached-runs-need-setsid.md",
   "score": 0.5695, "confidence": 0.991, "verdict": "ok",
   "text": "A judge run launched with the Bash tool's `run_in_background` died at query 527 of 759 when the Claude session restarted ... **Fix that worked:** launch the long job as its own Windows process ..."},
  {"chunk_id": "cfcbb0b31d77", "source": "recall/systemd-user-scopes-need-linger.md",
   "score": 0.5393, "confidence": 0.9784, "verdict": "ok",
   "text": "... **`setsid` and `nohup` do not help.** They protect against terminal HANGUP. This is systemd stopping the whole user manager, a different failure ..."}],
 "evidence": {"decision": "answer",
  "citable": ["00192d073e84", "c73d42e81759", "cfcbb0b31d77"]}}
```

The question names only the symptom. Memory returns the fix that worked, a second case of the
same failure, and the neighbouring failure where that fix does not apply. The agent reads what was
learned the first time instead of repeating the diagnosis.

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
