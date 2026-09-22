# Workstream 2 handoff: request aware retrieval plans

Workstream 1 adds `recall.retrieval_plan.RetrievalPlan` and
`RetrievalPlanResolver`. The resolver is deterministic and consumes only request metadata:
`scope`, `source`, `modality`, an optional explicit `route_id`, and conservative query signal
tokens. It returns one primary physical tenant and, only for an ambiguous request, one rescue
tenant. Each selected leg carries a bounded item limit. The plan also carries the allowed tenant
set, fanout limit, latency bound, route id, policy version, and selection reason.

The stable executor boundary is:

```python
plan = resolver.resolve(
    current_tenant=logical_tenant,
    query=query,
    request_scope=scope,
    source=source,
    modality=modality,
    route_id=route_id,
)
plan = registry.validate_retrieval_plan(plan)
for leg in plan.selected_legs:
    # Workstream 2 owns bounded execution and result assembly.
    consume(leg.tenant, leg.limit, leg.identity)
```

`StoreRegistry.validate_retrieval_plan` rejects tenants outside its allowlist, rejects active
routes whose generation is not servable, rejects runtime embedding-profile mismatches, and
requires a certified calibration when a generation store exposes calibration resolution. It
returns the plan with tenant, generation, embedding profile, calibration, trust, and provenance
identity fields populated where the serving boundary can resolve them; otherwise calibration and
trust remain deferred to that existing serving boundary.

The MCP tools `recall_search` and `recall_evidence` accept optional `scope`, `modality`, and
`route_id` fields. Their additive `retrieval_plan` response field exposes the route id, reason,
selected legs, limits, fanout, latency bound, identities, and bounded federation diagnostics when
the federation layer is enabled. The serving adapter keeps the compatibility route single tenant,
and only configured certified plans may execute bounded specialist fanout.

Enable configuration with `RECALL_RETRIEVAL_PLANS_JSON`, for example:

```json
{
  "version": 1,
  "default_route": "memory",
  "routes": [
    {
      "id": "memory",
      "primary_tenant": "memory",
      "rescue_tenant": "re-call-code-gen",
      "allowed_tenants": ["memory", "re-call-code-gen", "re-call-docs"],
      "scopes": ["memory"],
      "modalities": ["text"],
      "primary_limit": 5,
      "rescue_limit": 2,
      "latency_budget_ms": 900
    }
  ]
}
```

Leave the setting unset for the compatibility plan, which selects only the current tenant and
preserves existing callers. Invalid versioned configuration fails during settings bootstrap.
Federation is separately opt in with `RECALL_FEDERATION_MODE=shadow` or `active`; it requires
authenticated per-tenant stores and complete certified generation identities.

No quality or latency measurement was added. The available RE-call memory search endpoint failed
with an application error in this session, so no prior-work claim was inferred from that outage.
The closed hypothesis search did return negative evidence against rank disagreement triggers and
weighted score fusion; that evidence supports keeping this layer deterministic and score free.
