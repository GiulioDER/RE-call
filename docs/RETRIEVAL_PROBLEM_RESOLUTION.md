# Retrieval problem resolution

The five known retrieval problems are handled as separate engineering concerns. No single
profile or threshold can solve all five.

## Item 2 measurement

On 2026-09-07 I ran the current near miss harness with the pinned
cross-encoder/qnli-distilroberta-base judge, HashingEmbedder at 64 dimensions, and the shipped
ten case near miss set. The leave one out calibration used 20 answerable and 20 far gap queries.

| Arm | Near miss false confidence | Far gap false confidence | Answerable false abstain | Answerable MRR |
| --- | ---: | ---: | ---: | ---: |
| threshold | 0.50 | 0.20 | 0.50 | 0.350 |
| threshold plus entailment | 0.10 | 0.00 | 0.70 | 0.300 |
| entailment only | 0.70 | 0.10 | 0.05 | 0.867 |

The result supports enabling entailment only as a measured deployment choice. The stacked arm
substantially improves near miss and far gap rejection, but its answerable false abstention is too
high for an unqualified default. The entailment only arm preserves more answerable queries but
loses far gap protection. A larger, representative evaluation set is still required before
selecting a production threshold or default.

| Problem | Resolution | Evidence and remaining boundary |
| --- | --- | --- |
| Quality depends on corpus style and embedder | Treat corpus and embedder as calibration lineage. Require a matching calibration for trusted production reads, and report embedding, corpus, generation, and calibration identities. | Calibration does not make an unsuitable corpus suitable. Each corpus and embedder pair still needs representative evaluation. |
| Near misses and weak abstention | Open. The cross encoder entailment guard measured above has been removed: as an evidence admission stage on the AML Coding screen of 2026-09-19 it cut complete coverage from 0.92 to 0.42 and raised Search p95 from 0.5 s to 37 s. A deployment that still sets 'RECALL_ENTAILMENT' to a true value is refused at startup. | Calibrated thresholds remain the only shipped abstention signal. |
| Quality profiles cost latency and memory | Keep profile selection process scoped, clamp candidate and returned counts, bound concurrency and queue capacity, and measure every retrieval stage. | The committed quality benchmark peaked at 934.2 MiB RSS at offered concurrency 8. The measured policy is a 1.25 GiB RSS alert and an explicit SLO in [RETRIEVAL_SLO.md](RETRIEVAL_SLO.md). |
| Latency and concurrency are policy choices | Publish profile budgets and concurrency limits, record queue wait, total time, budget overruns, and shed requests, and expose p50, p95, and p99 histograms. | The quality policy is p95 ≤ 2,000 ms and p99 ≤ 2,200 ms for served warm requests, with four recommended offered concurrent requests, two running slots, and eight queued slots. |
| Legacy profile ambiguity | Make the canonical MCP response models authoritative and use 'null' when lineage is absent. A real legacy retrieval can still report 'legacy' from the retrieval diagnostics. | Existing callers can continue importing models from 'recall_mcp.service'; the service now reexports the canonical definitions. |

The entailment guard was designed as a separate decision stage after trust evaluation. It has
been removed; the Item 2 measurement above is kept as the record of what it did.
