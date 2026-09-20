# AML grounded graph merged live smoke

Date: 2026-09-20

Merged graph commit: `3b758912d1752208bc6ba5863db4ac8ccfafb1ce`

Variant: `G1_grounded_graph`

Endpoint: isolated VPS2 localhost experiment on port 18006. The public
`recall-aml.service` was not modified during this confirmation and remained active with its
original process and zero restarts. The isolated experiment service was stopped after the probe.

## Official-envelope simulation

The probe used the AML API guide's synchronous Add and Search request envelopes with one temporary
`user_id`, `top_k: 100`, and a cleanup step in `finally`. It was a compatibility and mechanism
smoke, not an AML quality evaluation or an official platform Smoke job.

| Signal | Result |
|---|---:|
| Add HTTP status | 200 |
| Search HTTP status | 200 |
| Raw chunks | 2 |
| Compiled chunks | 2 |
| Compiler fallback | false |
| Corpus status | ready |
| Authored relations | 2 |
| Eligible relations | 2 |
| Graph sidecar chunks | 2 |
| Graph attempted | true |
| Graph fallback | false |
| Relation hits | 2 |
| Candidate promotions | 0 |
| Returned only raw evidence | true |
| Top-100 membership preserved | true |

The two raw records were both inside the protected first-eight prefix, so zero promotions was the
expected safe result. The deterministic graph test covers the complementary case: an eligible tail
target can move only to rank 9, while the first eight results and full result membership remain
unchanged.

No AML evaluation credential was available on the host, so this receipt does not claim an official
Smoke or Full result.
