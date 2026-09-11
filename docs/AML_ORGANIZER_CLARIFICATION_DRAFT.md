# Draft for organizer review: model policy clarification

Status: draft only. Do not send without Giulio's approval.

Subject: Clarification for RE-call Hosted 1.0 Industry submission

Hello AML team,

We are preparing RE-call Hosted 1.0 for the Commercial Products board. The service uses exactly
`gpt-4o-mini` for every generative operation during Add and Search. Its non-generative retrieval
stack uses `voyage-4` embeddings and `voyage:rerank-2.5` reranking.

The participation material says the memory system's Add and Search model must be `gpt-4o-mini`,
while also saying the platform does not prescribe the database, index, embedding model, or internal
architecture. Could you confirm in writing that Voyage embeddings and reranking are permitted when
all generative Add and Search operations remain fixed to `gpt-4o-mini`?

We will disclose the complete fixed configuration and keep the submitted endpoint stable for at
least 30 days.

Thank you,
Giulio
