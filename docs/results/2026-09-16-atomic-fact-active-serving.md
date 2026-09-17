# Atomic fact active serving

Decision: `READY_FOR_RELEASE`.

The frozen 96-query current-generation regression passed before fusion with 16 exact-parent gains
and 11 gold-source gains, with zero losses in both comparisons. Through fresh production-mode MCP
processes, active serving gained six exact parents and seven gold sources at cutoff five, again with
zero losses. Cutoffs one and three also had zero losses.

The final production-path replay measured p95 12.701 ms and p99 36.903 ms against gates of 20 ms
and 45 ms.
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

Production now serves commit `d55f9723ac6ecf5580f9f1334f9e46408a4426aa` from
`/home/sentiment/recall-repos/atomic-active-prod-d55f9723` with
`RECALL_ATOMIC_RESCUE_MODE=active`. The database route and active generation did not change. A
fresh inherited-mode rollback rehearsal proved that `off` removes the atomic stage and that
restoring `active` restores it, with exact lineage in both processes. Database health, the
published calibration, and the current artifact binding were rechecked after activation.

The production refresh pipeline now builds and validates a generation-bound artifact after
certification and before promotion whenever active mode is configured. Artifact failure therefore
stops promotion. Its read-only check reports the active mode and current artifact readiness.

A fixed six-query diagnostic recovered five of six preregistered gain queries in repeated short
processes, while the authoritative 96-query replay recovered all six exact gains and seven source
gains with no losses. The same retrieval path showed bounded evidence-order variance when the old
snapshot was compared with itself. The release decision therefore remains bound to the frozen full
pool, not to that smaller diagnostic subset.
