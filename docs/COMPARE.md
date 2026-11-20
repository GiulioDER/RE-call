# RE-call compared with other memory layers

These projects solve different parts of agent memory, and several of them are further along than
RE-call. The useful question is not which is best but what each one does when a remembered fact
stops being true, and what it needs to run.

Descriptions of other projects come from their public repositories and documentation, checked on
2026-11-05. They change quickly; follow the links before relying on a detail. The sourced,
per-capability record is the [prior art comparison](PRIOR_ART.md).

This page is the maintained copy of the comparison published at `site/compare.html`.

## Which one fits

| Choose | When you want | What it needs |
|---|---|---|
| RE-call | Memory that tells the agent a result is superseded, shows where it came from, and refuses (ABSTAIN) when nothing in memory supports an answer. | A local SQLite file (Lite) or PostgreSQL with pgvector. No LLM call in the memory layer by default. |
| [Mem0](https://github.com/mem0ai/mem0) | An LLM that extracts memories from conversations for you, as a library, self-hosted or as a managed cloud service. | An LLM for extraction, plus a vector store. |
| [Graphiti](https://github.com/getzep/graphiti) (behind [Zep](https://www.getzep.com/)) | A temporal knowledge graph whose ingestion infers entities and relations and invalidates contradicted facts. | A graph database (Neo4j, FalkorDB, Kuzu or Amazon Neptune) and LLM calls at ingestion. |
| [Letta](https://github.com/letta-ai/letta) | A complete stateful agent runtime in which the agent edits its own memory blocks and searches archival memory. | The Letta server, self-hosted or on Letta Cloud. It is an agent layer, not a retrieval library. |
| [LangMem](https://langchain-ai.github.io/langmem/) | Memory management tools inside a LangGraph application. | Usually LangGraph and its store. |
| Plain pgvector or a vector database | Nearest-neighbour text, where your own code decides whether a result is current or trustworthy. | The database alone. |

## What happens to a fact that stopped being true

- RE-call marks it superseded when a newer memory declares that it replaces it, and returns the
  current one ahead of it. The author declares the replacement; RE-call does not infer it.
- Graphiti infers the contradiction with an LLM at ingestion and invalidates the older edge rather
  than deleting it.
- Mem0, since its 2026 algorithm, adds memories without updating or deleting them on the add path,
  and handles time at retrieval. Explicit update and delete remain in its API.
- Letta leaves it to the agent, which rewrites its own memory blocks.
- A plain vector store returns the stale text whenever it is the nearest match.

## What RE-call gives up

- Coverage of supersession. Declared replacements are precise but rare: on the reference corpus,
  2 of 792 memos declared one while 60 closed a decision only in prose. An LLM that infers
  replacements covers more and is sometimes confidently wrong; RE-call chooses precision.
- No entity reasoning. There is no knowledge graph of people and relations to reason over, which
  is where Graphiti is stronger.
- Abstention costs answers. Out of the box, default retrieval abstained on none of LoCoMo's 446
  adversarial questions. The shipped levers raise that to between 0.37 and 0.77 only by refusing a
  quarter to half of legitimate questions.

## The one paired measurement

On LoCoMo's 1,540 answerable questions, with the same GPT-4o reader and judge and matched retrieval
budgets, RE-call scored 48.4% judged answer accuracy against Mem0 OSS at 44.4%. RE-call ran this
test itself, and the direction depends on the reader: a later Sonnet 4.5 subset reverses it. The
method, artifact and limits are on the [benchmark scorecard](BENCHMARKS.md#locomo).

RE-call has not been compared head to head with Graphiti, Zep, Letta or LangMem, so this page makes
no score claim about them.

## Next

- [Install RE-call](https://re-call.dev/start.html), Lite or PostgreSQL.
- [How trusted search and the trust gate work](HOW_IT_WORKS.md).
- [Evidence and limits](EVIDENCE.md) of persistent memory.
