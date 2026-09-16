# Atomic fact active serving

Decision: `PASS_ATOMIC_ACTIVE_PREDEPLOYMENT`.

The frozen 96-query current-generation regression passed before fusion with 16 exact-parent gains
and 11 gold-source gains, with zero losses in both comparisons. Through fresh production-mode MCP
processes, active serving gained six exact parents and seven gold sources at cutoff five, again with
zero losses. Cutoffs one and three also had zero losses.

The atomic stage measured p95 12.130 ms and p99 18.865 ms against gates of 20 ms and 45 ms.
Source-scoped and security-policy-scoped requests each preserved exact control parity on 12 of 12
queries. There were no errors, no trusted-response regressions, and no unexplained trust changes.
Seven rows served no evidence in either arm but changed abstention reason; the frozen offline receipt
proved a different rank-six parent on all seven, so these are candidate-explained changes under the
preregistered rule.

The current artifact contains 6,323 atomic views over 4,247 parents and is bound to generation
`gen_919991221e1045e69824baa1a9be4e30` and calibration
`cal_cf8477d2df95464a938a3bcceb097803`. Its size is 27,077,806 bytes, cold load is 107.442 ms, and
resident-memory growth is 52,334,592 bytes. The registry also contains the immediate rollback
generation artifact.

Missing, malformed, wrong-lineage, and missing-parent probes all refused while active. A pinned
process served the rollback generation from its own artifact. Production routing did not change
during measurement.

This decision authorizes disabled-first deployment and the activation plus rollback rehearsal. It
is not `READY_FOR_RELEASE` until those production checks pass.
