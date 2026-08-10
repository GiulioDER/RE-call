# Storyboard

Target: 45 seconds.

Audience: developers building agents who have seen vector search return plausible but stale memory.

Message:

1. Plain RAG returns the highest cosine stale memory.
2. RE-call returns trust state, not only similarity.
3. The current memory wins even with a lower cosine.
4. Unanswerable questions return abstention.
5. Install and try the local Postgres plus pgvector path.

Narrative beats:

| Original time | 2x cut | Beat |
|---|---|---|
| 0 to 8 | 0 to 4 | Introduce the problem: agent memory is not just nearest neighbor search. |
| 8 to 23 | 4 to 11.5 | Show plain vector retrieval choosing the stale rate-limit memory. |
| 23 to 45 | 11.5 to 22.5 | Run RE-call trusted search, where the stale hit is marked superseded. |
| 45 to 61 | 22.5 to 30.5 | Show an unanswerable query producing an abstention instead of a guess. |
| 61 to 76 | 30.5 to 38 | Show what the system returns: verdict, confidence, provenance, validity. |
| 76 to 90 | 38 to 45 | Close with install command and positioning. |

Short caption:

RE-call is trustworthy memory for AI agents: verdict, confidence, provenance, validity, or abstain.
