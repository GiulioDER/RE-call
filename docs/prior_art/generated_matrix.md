# RE-call prior art capability matrix

Generated from the canonical evidence records in `docs/prior_art/`.

This matrix records evidence, not absolute absence. `unknown` means the investigation is incomplete; `not_evidenced` means the reviewed sources did not establish the capability.

## System overview

This overview has one row per system and one column per capability group. Group cells aggregate the detailed capability records below; `partial` means the group is not completely evidenced.

| System | representation | write_path | retrieval | validity_and_revision | time | provenance | uncertainty | authority_and_scope | deletion_and_forgetting | action_feedback | security | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AgeMem | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` |
| AgentMemoryBench | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` |
| A-MEM | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| Agent-Memory Protocol | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` |
| Bad Memory | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` |
| ChronoMem | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| Dependency Guided Rollback Repair | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` |
| Graphiti | `partial` | `unknown` | `partial` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| HippoRAG | `unknown` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| LangMem | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| Letta | `partial` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| LongMemEval | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` |
| LongMemEval V2 | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` |
| Mem0 | `partial` | `partial` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| MemMachine | `partial` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| Memory R1 | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` |
| MemoryBank | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` |
| MemSecBench | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` |
| MERIT | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` |
| MindMemOS | `unknown` | `partial` | `unknown` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |
| MPBench | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` |
| Neo4j Agent Memory | `partial` | `partial` | `unknown` | `unknown` | `partial` | `partial` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` |
| Quipu | `partial` | `unknown` | `unknown` | `unknown` | `partial` | `partial` | `unknown` | `partial` | `partial` | `unknown` | `unknown` | `unknown` |
| RE-call | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `partial` | `partial` | `partial` | `unknown` | `unknown` | `unknown` | `unknown` |
| Reflexion | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` | `partial` | `unknown` | `unknown` |

## representation: Raw episodes

The system preserves source episodes or turns as retrievable memory.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `verified` | [src_letta_repository](https://github.com/letta-ai/letta) | Letta supports persistent agent memory through externally managed memory blocks and archival storage. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `verified` | [src_memmachine_paper](https://arxiv.org/abs/2604.04853) | MemMachine stores entire conversational episodes to preserve ground truth for later memory use. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `verified` | [src_quipu_repository](https://github.com/jeffhajewski/quipu) | Quipu preserves raw evidence and links derived memories back to the supporting messages. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## representation: Structured facts

The system represents memory as structured facts or attributes rather than only raw text.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `verified` | [src_mem0_paper](https://arxiv.org/abs/2504.19413) | Mem0 extracts and stores structured persistent memories from ongoing conversations. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## representation: Knowledge graph

The system represents entities and relations as a graph that can be queried.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `verified` | [src_graphiti_docs](https://help.getzep.com/graphiti/getting-started/overview) | Graphiti represents agent memory as a temporal knowledge graph of entities and relations. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `partial` | [src_mem0_paper](https://arxiv.org/abs/2504.19413) | Mem0 provides a graph based variant for representing relationships among conversational elements. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `verified` | [src_neo4j_agent_memory](https://github.com/neo4j-labs/agent-memory) | Neo4j Agent Memory represents agent memory as a knowledge graph with entity and relationship structure. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## write_path: Memory extraction

The system extracts persistent memory from conversations or agent traces.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `verified` | [src_langmem_docs](https://langchain-ai.github.io/langmem/) | LangMem provides memory management components for extracting and managing long term memory in LangGraph applications. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `verified` | [src_mem0_paper](https://arxiv.org/abs/2504.19413) | Mem0 uses an extraction and consolidation pipeline for ongoing conversations. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## write_path: Memory consolidation

The system merges, compresses, or organizes multiple memories during maintenance.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `verified` | [src_mindmemos_paper](https://arxiv.org/abs/2608.12428) | MindMemOS consolidates accumulated memories by merging redundant records and resolving conflicts. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `verified` | [src_neo4j_agent_memory](https://github.com/neo4j-labs/agent-memory) | Neo4j Agent Memory provides consolidation primitives for deduplicating entities. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## write_path: Agent authored memory

The agent can deliberately decide what to write or update in persistent memory.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `verified` | [src_agemem_paper](https://arxiv.org/abs/2601.01885) | AgeMem exposes memory operations as tool actions so the agent can decide when to store, retrieve, update, summarize, or discard information. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `verified` | [src_amem_paper](https://arxiv.org/abs/2502.12110) | A-MEM uses an agentic process to create structured memory notes, link new memories to historical ones, and evolve existing memory attributes. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `verified` | [src_letta_benchmark](https://www.letta.com/blog/letta-leaderboard/) | Letta agents can manage memory through tool calls as part of the agent runtime. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `verified` | [src_memory_r1_paper](https://arxiv.org/abs/2508.19828) | Memory R1 uses a learned Memory Manager to choose whether to add, update, delete, or leave memory unchanged. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## retrieval: Dense retrieval

The system retrieves memory using vector similarity.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `verified` | [src_mem0_paper](https://arxiv.org/abs/2504.19413) | Mem0 evaluates memory retrieval as part of a persistent memory architecture. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## retrieval: Lexical retrieval

The system retrieves memory using lexical or full text search.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## retrieval: Graph traversal

The system expands retrieval through graph relations or traversals.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `verified` | [src_graphiti_docs](https://help.getzep.com/graphiti/getting-started/overview) | Graphiti combines semantic, full text, and graph based retrieval. |
| HippoRAG | `verified` | [src_hipporag_paper](https://arxiv.org/abs/2405.14831) | HippoRAG combines a knowledge graph with Personalized PageRank to expand retrieval across related information for multi hop questions. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## retrieval: Query time context construction

The system constructs or expands the memory context at query time using the retrieved evidence.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `verified` | [src_memmachine_paper](https://arxiv.org/abs/2604.04853) | MemMachine expands nucleus retrieval matches with surrounding conversational context at retrieval time. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## validity_and_revision: Supersession

The system represents that a newer memory replaces an older memory.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `verified` | [src_recall_readme](https://github.com/GiulioDER/RE-call/blob/master/README.md) | RE-call uses declared supersession and validity metadata to demote stale memories during retrieval. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## validity_and_revision: Contradiction handling

The system detects or represents conflicting memories.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` | [src_langmem_docs](https://langchain-ai.github.io/langmem/) | The reviewed LangMem documentation is insufficient to establish its contradiction handling behavior. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `verified` | [src_mindmemos_paper](https://arxiv.org/abs/2608.12428) | MindMemOS resolves conflicting accumulated memory records during its consolidation process. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## validity_and_revision: Rollback

The system can restore or select an earlier memory state.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `verified` | [src_chronomem_paper](https://arxiv.org/abs/2607.27773) | ChronoMem snapshots whole memory states and maps natural language rollback requests to historical versions. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## time: Valid time

The system records when a fact was true in the world.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `verified` | [src_graphiti_docs](https://help.getzep.com/graphiti/getting-started/overview) | Graphiti records when a relation became valid and when it stopped being valid. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `verified` | [src_neo4j_agent_memory](https://github.com/neo4j-labs/agent-memory) | Neo4j Agent Memory tracks when facts become valid or invalid. |
| Quipu | `verified` | [src_quipu_repository](https://github.com/jeffhajewski/quipu) | Quipu stores current facts with temporal supersession so old values remain historical rather than silently replacing the record. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## time: Ingestion time

The system records when it learned or ingested a fact.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `verified` | [src_graphiti_docs](https://help.getzep.com/graphiti/getting-started/overview) | Graphiti records when it learned a fact and when it learned that the fact was no longer true. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## time: Point in time retrieval

The system can answer what was known or valid at a requested historical time.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## provenance: Source attribution

A returned memory can identify the source evidence supporting it.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `verified` | [src_neo4j_agent_memory](https://github.com/neo4j-labs/agent-memory) | Neo4j Agent Memory tracks where entities were extracted from and which extractor produced them. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `verified` | [src_recall_readme](https://github.com/GiulioDER/RE-call/blob/master/README.md) | RE-call returns provenance and source metadata with retrieved memory evidence. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## provenance: Transformation lineage

Derived memories link back through transformations to their source evidence.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `verified` | [src_quipu_repository](https://github.com/jeffhajewski/quipu) | Quipu records supporting evidence for derived facts and exposes provenance inspection for those derivations. |
| RE-call | `partial` | [src_recall_reasoning_graph](https://github.com/GiulioDER/RE-call/blob/master/docs/REASONING_GRAPH.md) | RE-call projects typed semantic entities and relations with supporting chunk identifiers and generation identity. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## provenance: Evidence receipts

The system returns a machine readable record of evidence used for a memory result.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `partial` | [src_quipu_repository](https://github.com/jeffhajewski/quipu) | Quipu exposes provenance inspection for derived memories, including the supporting evidence used for a memory result. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## uncertainty: Abstention

The system can return an explicit refusal when memory support is insufficient.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `verified` | [src_recall_readme](https://github.com/GiulioDER/RE-call/blob/master/README.md) | RE-call can return an explicit abstention when no valid result clears its calibrated threshold. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## uncertainty: Entailment checking

The system checks whether retrieved evidence actually supports the requested claim.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## uncertainty: Confidence calibration

The system calibrates a trust or confidence decision against labeled examples.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## authority_and_scope: User consent

The system makes persistent memory subject to an explicit user consent or control decision.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## authority_and_scope: Tenant scope

The system enforces tenant boundaries on memory storage and retrieval.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `verified` | [src_quipu_repository](https://github.com/jeffhajewski/quipu) | Quipu supports scoped retrieval so memory access can be constrained to the requested scope. |
| RE-call | `verified` | [src_recall_readme](https://github.com/GiulioDER/RE-call/blob/master/README.md) | RE-call includes tenant identifiers and row level security in its production boundaries. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## authority_and_scope: Multi agent handoff

Memory can be intentionally shared or handed from one agent to another with scope semantics.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## deletion_and_forgetting: Source deletion

A source memory can be deleted from the system.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `verified` | [src_memory_r1_paper](https://arxiv.org/abs/2508.19828) | Memory R1 trains a Memory Manager to perform structured DELETE operations on external memory entries. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## deletion_and_forgetting: Derived deletion propagation

Deleting a source removes or invalidates derived memories, graph edges, embeddings, and caches that depend on it.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `verified` | [src_dependency_rollback_paper](https://arxiv.org/abs/2608.10502) | Dependency Guided Rollback Repair traces memory to action dependencies, deactivates unsupported memory state, and selectively replays affected computation while preserving unaffected state. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `verified` | [src_quipu_repository](https://github.com/jeffhajewski/quipu) | Quipu propagates forgetting from raw evidence to derived memories and reports the resulting closure. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## deletion_and_forgetting: Selective forgetting

The system can forget selected memories without clearing unrelated memory.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `verified` | [src_agemem_paper](https://arxiv.org/abs/2601.01885) | AgeMem includes discard as an agent controlled operation for removing selected information from memory. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `verified` | [src_memorybank_paper](https://arxiv.org/abs/2305.10250) | MemoryBank selectively forgets or reinforces memories using elapsed time and relative significance. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## action_feedback: Outcome storage

The system stores observed results of agent actions as reusable memory.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `verified` | [src_merit_paper](https://arxiv.org/abs/2608.05906) | MERIT stores oracle verified repair corrections and observed unsuccessful directions in an online dual polarity episodic memory. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `partial` | [src_neo4j_agent_memory](https://github.com/neo4j-labs/agent-memory) | Neo4j Agent Memory stores reasoning traces, tool usage, and audit edges linking reasoning steps to entities. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## action_feedback: Outcome linked revision

An observed action result updates the memory claim that influenced that action.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## action_feedback: Policy revision

Remembered outcomes modify future agent behavior or policy selection.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `verified` | [src_reflexion_paper](https://arxiv.org/abs/2303.11366) | Reflexion stores verbal reflections derived from task feedback in episodic memory to improve later decision making. |

## security: Prompt injection resistance

The system limits memory content from changing agent instructions or tool authority through injection.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## security: Memory poisoning resistance

The system detects or limits malicious or misleading persistent memories.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## security: Egress control

The system controls which memory fields can leave the memory boundary for a model or tool.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `verified` | [src_amp_paper](https://proceedings.mlr.press/v317/wu26a.html) | The Agent-Memory Protocol defines deterministic redaction, purpose limited packing, and hydration operations to keep personal identifiers inside the user boundary. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## evaluation: Temporal evaluation

The system is evaluated on changing facts and historical validity.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `verified` | [src_longmemeval_paper](https://arxiv.org/abs/2410.10813) | LongMemEval evaluates long term memory systems on temporal reasoning and knowledge updates across timestamped interaction histories. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## evaluation: Conflict evaluation

The system is evaluated on contradictory memories or conflicting claims.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `not_evidenced` | [src_letta_benchmark](https://www.letta.com/blog/letta-leaderboard/) | The reviewed Letta benchmark sources do not establish a dedicated conflict resolution evaluation cell. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## evaluation: Action impact evaluation

The system is evaluated by whether memory improves downstream agent actions or outcomes.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `partial` | [src_longmemeval_v2_repo](https://github.com/xiaowu0162/LongMemEval-V2) | LongMemEval V2 evaluates whether compact memory evidence improves downstream agent question answering about environment state, workflows, and recurring failure modes. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## evaluation: Continual learning evaluation

The system is evaluated over an ordered sequence of tasks or episodes where memory must support transfer and forgetting.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `verified` | [src_agent_memory_bench](https://openreview.net/pdf?id=MSXbrNExax) | AgentMemoryBench jointly evaluates system and personal agent memory in a continual learning framework covering online learning, transfer, and forgetting. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `unknown` |  | No accepted claim in the corpus. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `unknown` |  | No accepted claim in the corpus. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `unknown` |  | No accepted claim in the corpus. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## evaluation: Memory security evaluation

The system or benchmark is evaluated on persistent memory poisoning, prompt injection, downstream consequences, or selective repair.

| System | Value | Evidence | Claim |
| --- | --- | --- | --- |
| AgeMem | `unknown` |  | No accepted claim in the corpus. |
| AgentMemoryBench | `unknown` |  | No accepted claim in the corpus. |
| A-MEM | `unknown` |  | No accepted claim in the corpus. |
| Agent-Memory Protocol | `unknown` |  | No accepted claim in the corpus. |
| Bad Memory | `verified` | [src_bad_memory_paper](https://arxiv.org/abs/2607.14611) | Bad Memory evaluates whether prompt injection payloads planted in persistent memory influence current and future agent sessions. |
| ChronoMem | `unknown` |  | No accepted claim in the corpus. |
| Dependency Guided Rollback Repair | `unknown` |  | No accepted claim in the corpus. |
| Graphiti | `unknown` |  | No accepted claim in the corpus. |
| HippoRAG | `unknown` |  | No accepted claim in the corpus. |
| LangMem | `unknown` |  | No accepted claim in the corpus. |
| Letta | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval | `unknown` |  | No accepted claim in the corpus. |
| LongMemEval V2 | `unknown` |  | No accepted claim in the corpus. |
| Mem0 | `unknown` |  | No accepted claim in the corpus. |
| MemMachine | `unknown` |  | No accepted claim in the corpus. |
| Memory R1 | `unknown` |  | No accepted claim in the corpus. |
| MemoryBank | `unknown` |  | No accepted claim in the corpus. |
| MemSecBench | `verified` | [src_memsecbench_paper](https://arxiv.org/abs/2607.27080) | MemSecBench evaluates the lifecycle of malicious memory from persistence through downstream consequence and selective repair. |
| MERIT | `unknown` |  | No accepted claim in the corpus. |
| MindMemOS | `unknown` |  | No accepted claim in the corpus. |
| MPBench | `verified` | [src_mpbench_paper](https://arxiv.org/abs/2606.04329) | MPBench evaluates memory poisoning attacks across multiple memory write channels and attack classes. |
| Neo4j Agent Memory | `unknown` |  | No accepted claim in the corpus. |
| Quipu | `unknown` |  | No accepted claim in the corpus. |
| RE-call | `unknown` |  | No accepted claim in the corpus. |
| Reflexion | `unknown` |  | No accepted claim in the corpus. |

## Value legend

`verified` means primary evidence directly supports the claim. `partial` means the evidence covers a narrower case. `not_evidenced` means the reviewed sources did not establish the capability. `unknown` means research is incomplete. `contradicted` means available evidence conflicts with the claim. `contested` is rendered when accepted claims for one cell conflict.

## Incomplete or unresolved claims

* `clm_langmem_revision_001` for `sys_langmem` and `validity_and_revision.contradiction` is `unknown`.
* `clm_letta_eval_001` for `sys_letta` and `evaluation.conflict_resolution` is `not_evidenced`.

## Conflicting evidence

No conflicting evidence is recorded.
