# Multimodal tenant release handoff

Workstream 3 is implemented as release integrator groundwork. No deployment, migration, merge,
or official quality evaluation was performed.

Implemented surfaces:

* `recall.multimodal` provides the isolated tenant config, bounded media provenance, sidecar and
  query contracts, authorization and byte budget refusals, and the Voyage adapter.
* `recall.embedding_registry` registers `voyage-multimodal-3.5-v1` and carries its identity into
  the runtime embedder.
* MCP settings expose the feature flag and budgets. `scripts/session-mcp.sh` keeps the specialist
  off by default and adds it only with `RECALL_MCP_INCLUDE_MULTIMODAL=1`.
* Focused tests cover digest linkage, media and response budgets, tenant isolation, disabled
  defaults, and generation profile identity. Their docstrings record deliberate red mutations and
  restored green behavior.

Integrator gates before activation:

1. Provision a controlled object store and retention and erasure policy for the multimodal tenant.
2. Create a tenant specific generation and calibration using the registered profile. Do not reuse
   text specialist generations or calibration artifacts.
3. Configure provider egress, credentials, access policy, and budgets in the deployment environment.
4. Run the relevant tests, lint, typing, schema readiness, generation validation, and calibration
   gates on the target checkout. Keep MM2 classified as operational evidence only.
5. Run a separately approved and preregistered quality evaluation before making any quality claim.
