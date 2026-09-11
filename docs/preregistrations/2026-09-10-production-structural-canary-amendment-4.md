# Amendment 4: rerun after serving repair

Recorded after fixture ingestion and before calibration or effect measurement. This amendment
binds the locked canary to the newly built generation after the two production repairs.

Serving repair commit: `563fb73`.

Source repair commit: `8203c0c8`.

New generation: `gen_01e5d39a320848c2a72787424f32afd9`.

Use the exact 40 calibration labels from amendment 3, unchanged, with idempotency key
`structural-edge-canary-calibration-v4-20260910`. If certification fails, stop without publishing
or changing trust settings. If certification succeeds, publish and promote only this canary
generation, then rerun the original paired `off` and `one_hop` arms with the original query and
budgets. No query, ranking, relation shape, or budget changes are allowed.

The cleanup requirement remains unchanged: remove `a.md`, `b.md`, and `c.md` from the isolated
tenant and verify an empty inventory after measurement. Do not touch the `memory` tenant.
