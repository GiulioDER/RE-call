# Atomic fact Context 4 source sanitation clarification

Status: registered before document embedding and before any retrieval outcome.

The full parent alignment preflight registered in
`2026-09-16-atomic-fact-context4-implementation-clarification.md` next stopped on one included,
hash matched source. Its original bytes contain a NUL. Production removed that byte before parsing
because PostgreSQL text cannot store NUL, while the pilot parsed it and therefore produced a fact
that was not contained in the unchanged production parent.

The auxiliary builder now matches the production ingestion semantics by decoding verified bytes
as UTF-8 with optional BOM removal and stripping NUL before `parse_document`. Manifest size and
SHA256 verification remain over the unmodified source bytes. CRLF preservation from the earlier
clarification remains unchanged. Every resulting view must still occur inside its pinned parent or
the run stops before document embedding.

This corrects source reproduction only. It does not change the source universe, query pool,
lineage, candidate arms, rendering, metrics, thresholds, or decision rule.
