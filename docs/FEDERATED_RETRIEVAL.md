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

The configured primary prefix is protected when `primary_prefix + rescue_slots <= result_k`;
invalid configurations are rejected. Secondary candidates may enter only after the primary segment,
only when they are trusted and calibrated by their own leg, and only when their identifier and
source content are novel relative to the primary pool. `RECALL_FEDERATION_RESCUE_SLOTS` bounds
the secondary tail; a rescue slot that no secondary candidate fills goes back to the primary leg.
A secondary leg cannot fill a missing or rejected primary leg.

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
In the serving adapter, `off` searches the caller's authenticated tenant even when a configured
route names another primary tenant, so `retrieval_plan` is advisory until the mode is `shadow` or
`active`. The primary leg is sized to cover the request's `k`. Under the default
`RECALL_FEDERATION_INVALID_LEG=omit`, a primary leg that raises keeps its own error type
(`fail_closed` wraps it in `FederationLegRejected`); in `shadow`, a primary leg that overruns the plan's latency budget falls back
to the control retrieval instead of failing the request.

## Handoff

The route planner puts the primary leg first, and the serving adapter binds each callback to its
own `StoreRegistry` store. `recall_search` responses include bounded federation diagnostics
inside `retrieval_plan`; in `active` mode its `served` list gives each served chunk's tenant,
generation, embedding profile and calibration, because a rescued hit is served inside the
primary result. `recall_evidence` always runs single tenant (federation off), because evidence
cards persist through the primary store's provenance controller, so it carries no federation
diagnostics. Keep `RECALL_FEDERATION_MODE=off` for deployments without independently
certified specialist generations; use `shadow` to observe bounded fanout before an approved
promotion to `active`. No deployment or external release action is implied by this module.

Federation emits bounded process metrics for total and per leg latency, fanout, candidate counts,
and rejections. It does not add multimodal ingestion, a learned router, unconditional fanout, raw
cross tenant score fusion, or a multivector schema.
