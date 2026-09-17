# Atomic fact current generation query scope clarification

Status: committed before document embedding or candidate retrieval.

The frozen 50 row file in the current generation shadow preregistration contains 22 answerable
rows with `relevant_files` source labels and 28 intentionally unanswerable controls with empty
`relevant_files`. The first runner launch exposed this already frozen shape and stopped during
input validation, before loading the embedder, building atomic views, embedding documents, or
running either retrieval arm.

This clarification does not change the file, hash, row population, arms, thresholds, or gates.
It fixes the denominators as follows:

1. Latency, determinism, candidate movement, and overlap use all 50 queries.
2. Gold source reach, rank one gold gains and losses, and labelled gold zero view coverage use
   only the 22 answerable rows.
3. The 28 unanswerable controls remain in the private per query artifact but do not receive an
   invented gold source and do not enter quality numerators or denominators.

As in the parent preregistration, every quality number from this previously consumed file remains
diagnostic only and cannot authorize serving or tune the candidate.
