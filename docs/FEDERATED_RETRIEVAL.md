# Bounded federated retrieval

`recall.federation.federate` is an opt in logical retrieval layer over isolated specialist
tenants. A `FederationLeg` binds one tenant to one generation, embedding profile, calibration,
route, and retrieval callback. The callback must return that tenant's already trust evaluated
`TrustedResult`.

The first leg is the primary control leg. Federation executes at most
`RECALL_FEDERATION_MAX_LEGS` legs with at most `RECALL_FEDERATION_MAX_CONCURRENCY` workers. Each
leg is independently checked for tenant, generation, embedding, certified calibration, trusted
state, and a clear tenant gap check. Invalid legs are omitted by default. Set
`RECALL_FEDERATION_INVALID_LEG=fail_closed` when omission is not acceptable.

Fusion uses reciprocal rank fusion over candidate identifiers only. A cosine or calibrated
confidence from one tenant is never compared with a value from another tenant. Candidates retain
their full lineage: tenant, generation, embedding profile, calibration, route, leg rank, fused
rank, and optional pipeline and corpus fingerprints.

The primary prefix is protected. Secondary candidates may enter only after the primary segment,
only when they are trusted and calibrated by their own leg, and only when their identifier and
source content are novel relative to the primary pool. `RECALL_FEDERATION_RESCUE_SLOTS` bounds
the secondary tail. A secondary leg cannot fill a missing or rejected primary leg.

Modes are deliberately explicit:

* `off`, the default, executes only the first leg and preserves the single tenant control path.
* `shadow` executes the bounded plan and records federation diagnostics, while `active` remains
  false for the caller.
* `active` permits a caller to serve `FederationResult.served`.

The serving adapter in `recall_mcp.federation_adapter` now consumes the versioned Workstream 1
route plan. It binds every selected leg to its own `StoreRegistry` store and embedding profile,
requires a complete certified generation identity, and passes each leg through its own trusted
retrieval callback. The adapter never discovers tenants from request input and never compares raw
scores across profiles.

The compatibility route remains single tenant. Configured federation requires authenticated store
registry access and resolved generation, profile, calibration, and trust identities. If those
identities are unavailable, serving fails closed instead of silently falling back to another leg.

## Handoff

The route planner puts the primary leg first, and the serving adapter binds each callback to its
own `StoreRegistry` store. Search and evidence responses include bounded federation diagnostics
inside `retrieval_plan`. Keep `RECALL_FEDERATION_MODE=off` for deployments without independently
certified specialist generations; use `shadow` to observe bounded fanout before an approved
promotion to `active`. No deployment or external release action is implied by this module.

Federation emits bounded process metrics for total and per leg latency, fanout, candidate counts,
and rejections. It does not add multimodal ingestion, a learned router, unconditional fanout, raw
cross tenant score fusion, or a multivector schema.
