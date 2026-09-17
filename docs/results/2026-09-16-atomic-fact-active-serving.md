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

Production now serves commit `00947aff86964d59e93863dbb34cccf3295e2788` from
`/home/sentiment/recall-repos/atomic-active-prod-00947aff` with
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

## Post-activation refresh verification

Measured 2026-09-17 by running the production memory refresh after adding the canonical production
handoff memo. The seven-step lifecycle built and validated 1,591 sources and 11,401 ordinary chunks
as generation `gen_6669297b3ec943c5bcdabe0051b39abf`. Carry-forward correctly refused because the
lineage reason changed from `explicit development build` to `hosted provider, production
admissible`. The actual chunking, model, dimension, profile ID, and profile fingerprint were
identical. Automatic recalibration on the stored certified query set published
`cal_4182117e50ff47169a734884a01303b8` and left the threshold at 0.4100.

The first artifact step failed closed before promotion because the external lifecycle script passed
`Nice=15` as a systemd scope property. Production continued serving the prior generation. The
script now applies niceness through `nice -n 15`, while retaining `MemoryMax=8G`,
`MemorySwapMax=0`, `CPUQuota=250%`, and four embedding threads. The corrected wrapper was exercised
successfully, the already certified candidate resumed without repeating generation embeddings, and
promotion completed.

The active artifact now contains 6,325 views over 4,249 parents, occupies 27,086,354 bytes, loads in
116.527 ms, and adds 52,355,072 resident bytes. All artifact construction limits passed. The active
lineage is pipeline `1c1be8a259dcd2949d0658ba4ec2e6637f89e235ee2a1feb527477c7eb4ed409`
and corpus `a6e0291a6944513216ba8879ff17316fa0725b78f4adebb408b25fa34f31f11e`.

After promotion, `index_memory_manifest.sh --check` reported the corpus unchanged, the embedding
lock free, active mode, and the current artifact ready. The serving symlink and resolved Python
module still point to `/home/sentiment/recall-repos/atomic-active-prod-00947aff`. The database holds
11,401 rows for the active generation. A fresh inherited-mode MCP process returned trusted evidence
with no failure code, exact current lineage, and an `atomic_rescue` stage. It retrieved the new
production handoff memo, proving that the persistent memory update is served. The single cold-process
atomic stage took 27.606 ms; this is a smoke observation, not a replacement for the frozen 96-query
latency distribution that made the release decision.
