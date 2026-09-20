# Routed specialist corpus qualification

Status: frozen when committed. No live provider result or hosted retrieval result from this
candidate may be inspected before the commit that adds this record.

## Question

Can one logical AML corpus preserve a shared chunk identity and metadata layer while storing
Voyage Code 4, Voyage Context 4, and Voyage Multimodal 3.5 vectors in physically isolated
namespaces, routing before retrieval, and combining only rank positions through reciprocal rank
fusion?

This is an integration qualification. Final coding task success belongs to the separately
preregistered AMB run.

## Frozen candidate

The implementation candidate is RE-call commit
`3cc8f9a1ba0beb57239b131568fec9eece97defb`, variant `C7_routed_specialists`.

The primary index uses profile `voyage-code-4-v1`. The Context specialist uses profile
`voyage-context-4-v1`. The visual specialist uses profile `voyage-multimodal-3.5-v1`.

Every logical chunk written to Code4 and Context4 must have the same chunk id, source, text, and
metadata. Each embedding profile has its own derived tenant. No vector is averaged with another
model's vector. No cosine score from one model is compared with a cosine score from another.

The deterministic router is `conservative-specialist-router-v1`:

1. A query containing image content or an explicit visual term selects the multimodal route.
2. A text query with a code signal selects Code4.
3. A text query with an explicit conversational memory signal and no code signal selects
   Context4.
4. Ambiguous text selects Code4. This protects the measured CAMBench coding baseline.

Within Code4 or Context4, the selected dense ranking is fused with canonical BM25 by unweighted
RRF with constant 60. On the multimodal route, the text ranking and visual ranking are fused by
primary chunk id with unweighted RRF and constant 60.

## Prior evidence boundary

The 2026-09-20 Context4 coding experiment closed equal peer fusion on this corpus. Context4 had
30 of 34 source recall at 10 against Code4's 34 of 34, and mean reciprocal rank 0.5997 against
0.8464. The protected suffix did not meet its minimum activation count. This candidate therefore
does not fuse Context4 into coding queries.

The multimodal quality lane remains separate. This qualification checks wiring, identity,
preservation, and provider readiness. It does not claim that multimodal task quality improved.

## Frozen qualification corpus and probes

The coding replay uses the committed AMB corpus manifest with SHA 256
`58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814`, all 196 source sessions,
1,220 content only windows, and all 34 task prompts.

Three additional isolated probes are used:

1. One coding text memory and one code query containing a file name and pytest signal.
2. One conversational memory and one query asking what was decided in yesterday's meeting.
3. One text plus PNG memory and one screenshot query.

All provider calls on VPS2 run under
`/home/sentiment/recall-repos/.locks/embed.lock`. Before the first call, the lock and process list
must show no competing embedding run. Add concurrency is three workers. The hard document budget
is 500,000 input tokens per text model. The visual probe contains one image.

## Predictions

1. Startup readiness succeeds for all three registered profiles and `/health` returns ready.
2. `/version` reports `C7_routed_specialists`, Code4 primary, Context4 specialist, Multimodal 3.5,
   the frozen router, and routed rank fusion.
3. After text ingestion, the Code4 and Context4 namespaces contain exactly the same 1,220 chunk
   ids and the same logical chunk payloads.
4. The two text namespaces contain different vectors for at least 99 percent of shared chunks.
5. Every one of the 34 AMB task prompts selects the code route.
6. C7 returns the exact ordered C6 top 100 coding ranking for all 34 task prompts. Its source recall
   at 10 remains 34 of 34 and its mean reciprocal rank is at least 0.8464.
7. The conversational probe selects Context4 and returns its planted memory within the first
   three results.
8. The visual probe selects Multimodal 3.5, returns the planted parent within the first three
   results, and reconstructs the original ordered text and image parts byte for byte.
9. No query response contains a raw cross model cosine comparison or an averaged heterogeneous
   vector.

## Decision rules

The candidate qualifies for the AMB Task Solve run only if predictions 1 through 9 all pass.
Any missing specialist namespace, identity mismatch, wrong route, provider readiness failure,
or coding rank regression blocks Task Solve until a new preregistration.

The qualification does not authorize an official AML Smoke or official AML evaluation. Those
remain blocked pending the user's explicit approval.

<!-- results and append only corrections go below this line; everything above is frozen -->

## Append only safety apparatus amendment, before live calls

No live call from this candidate had run when this amendment was written. The routed behavior
remains the frozen semantic implementation at `3cc8f9a1`. Three direct descendant apparatus
commits are required for the live qualification:

1. `654bcd1d` serializes primary Code4, Context4, and Multimodal provider calls through the shared
   VPS2 flock, including constructor probes and readiness calls. `/version` must report
   `embedding_call_lock: true`.
2. `00f79ea7` permits isolated unit and runtime environment filenames, so private lock aware C6
   and C7 comparison services cannot rewrite or restart the public C6 service configuration.
3. `ecfeee1c` adds the machine checked nine gate driver
   `scripts/aml_c7_qualification.py`. Its release manifest binds the driver source. It refuses
   wrong variants, uses three Add workers, audits shared identities and vector difference read
   only, compares all 100 ordered ranks for all 34 coding prompts, runs the Context4 and visual
   probes, records cleanup status, emits no secret values, and records
   `official_aml_launched: false`.

The live driver requires the explicit `--execute-live-qualification` flag and dedicated C6 and
C7 endpoint credentials. Any false gate blocks AMB Task Solve. This amendment does not authorize
an official AML Smoke or official AML evaluation.

### Mutation restoration correction, before live calls

Commit `ecfeee1c` was created while the qualification test worker was temporarily applying its
documented negative mutations. No live call used that revision. Commit `728793ad` restores the
intended controls: dedicated C7 credential selection, exact ordered top 100 comparison, forbidden
embedding field detection, and secret safe exception receipts. The live qualification must use
`728793ad` or a direct descendant containing it.
