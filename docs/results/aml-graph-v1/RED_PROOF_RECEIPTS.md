# AML grounded graph red proof receipts

These focused mutations were applied to production code, the named node was run, and the mutation
was restored before the green run.

| Test node | Production target | Plausible mutation | Intended failure |
|---|---|---|---|
| `tests/test_aml_hosted.py::test_grounded_graph_links_only_verbatim_server_resolved_evidence` | `recall_aml.graph.attach_grounded_relations` | Invert the exact quote comparison so supported spans are rejected | `relations` was empty instead of the one server-resolved `references` edge |
| `tests/test_aml_hosted.py::test_graph_promotion_protects_prefix_and_preserves_raw_membership` | `recall_aml.graph.GRAPH_PROTECTED_PREFIX` | Set the protected raw prefix from 8 to 0 | `raw_11` displaced `raw_0` at rank 1 |
| `tests/test_aml_hosted.py::test_graph_rejects_cross_session_relation_without_changing_raw_order` | `recall_aml.graph.promote_grounded_raw` | Remove the source-session equality gate between the compiled seed and raw target | A foreign sidecar promoted `raw_11` into rank 9 instead of preserving the raw order |
| `tests/test_aml_hosted.py::test_graph_variant_is_raw_identical_when_no_grounded_relation_exists` | `recall_aml.graph.promote_grounded_raw` | Reverse the baseline when the sidecar has no hits | The graph arm no longer matched the raw control at rank 1 |
| `tests/test_aml_hosted.py::test_graph_sidecar_failure_returns_the_exact_raw_ranking` | `recall_aml.service.HostedService.search` | Re-raise a sidecar exception instead of invoking `HostedRetriever.graph_fallback` | Search raised `RuntimeError` instead of returning the raw control ranking |
| `tests/test_aml_hosted.py::test_graph_variant_add_search_endpoint_uses_isolated_authored_sidecar` | `recall_aml.variants.GRAPH_VARIANTS` | Disable `graph_sidecar` on `G1_grounded_graph` | Corpus status omitted the sidecar and its authored relation, and Search did not report graph activation |
| `tests/test_aml_hosted.py::test_grounded_graph_vps2_setup_uses_a_distinct_store_and_generation` | `scripts/aml_experience_vps2_setup.sh` | Pre-change deployment selector without a `G1_grounded_graph` case | The exact graph variant case and its isolated table, generation, and app-root assertions were absent |
| `tests/test_aml_hosted.py::test_vps2_experiment_port_can_be_isolated_from_the_public_service` | `scripts/aml_experience_vps2_setup.sh` | Pre-change fixed `18004` experiment port | The launcher could not select an unused test port beside the public service |

All three failures reached the intended assertion. There were no collection, import, fixture,
timeout, or network failures.
