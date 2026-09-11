# RE-call Hosted 1.0 red proof receipts

These receipts were produced on 2026-09-11 before accepting the corresponding green tests. Each
red run changed a production symbol to a plausible defective implementation. Import, collection,
network, and setup failures do not appear here.

## Exact tenant isolation

Test node: `tests/test_aml_hosted.py::test_add_is_immediately_searchable_and_exactly_tenant_isolated`

Production symbol: `recall_aml.identity.tenant_for`

Mutation: replace the SHA256 derived tenant with one process wide constant.

Observed assertion failure: the Search result for `user-a` contained `other tenant secret` written
under `user-b`.

Green restoration: `tenant_for` again returns `aml_` plus the SHA256 digest of the exact user ID.

## Durable and concurrent Add idempotency

Test node: `tests/test_aml_hosted.py::test_idempotent_replay_is_single_write_and_conflict_is_rejected`

Production symbol: `recall_aml.service.HostedService.add`

Mutation: bypass `Repository.get_receipt` and always treat the replay as new.

Observed assertion failure: `persist_calls` was 2 instead of 1 for two concurrent identical Adds.

Green restoration: receipt lookup occurs inside the request keyed lock before compilation or
persistence.

## Supersession filtering

Test node: `tests/test_aml_hosted.py::test_packer_honors_supersession_deduplication_budget_and_top_k`

Production symbol: `recall_aml.retrieval.pack_evidence`

Mutation: make the superseded record exclusion branch unreachable.

Observed assertion failure: packed IDs were `new, old` instead of only `new`.

Green restoration: nonhistorical queries exclude candidate IDs named by supported supersession
references.

## Required generation model

Test node: `tests/test_aml_hosted.py::test_openai_compiler_treats_prompt_injection_as_data_and_uses_fixed_model`

Production symbol: `recall_aml.config.GENERATION_MODEL`

Mutation: change `gpt-4o-mini` to `gpt-4o`.

Observed assertion failure: the captured provider request model was `gpt-4o` rather than
`gpt-4o-mini`.

Green restoration: every compiler and facet request uses the fixed `gpt-4o-mini` constant.

## API authentication

Test node: `tests/test_aml_hosted.py::test_http_contract_auth_version_health_delete_and_validation`

Production symbol: `recall_aml.app._authenticated`

Mutation: return true before checking Bearer or `X-Api-Key` credentials.

Observed assertion failure: an unauthenticated Add returned HTTP 422 after reaching validation,
rather than HTTP 401 at the authentication boundary.

Green restoration: credentials are checked with constant time comparison before body processing.

## Benchmark source join

Test node: `tests/test_recall_hosted_adapter.py::test_adapter_uses_public_add_search_and_builds_read_only_mcp`

Production symbol: `adapters.recall_hosted.adapter.RecallHostedAdapter.search`

Mutation: derive the benchmark source path from the product's opaque `source` field instead of the
exact returned `session_id`.

Observed assertion failure: the ranked list had zero joinable documents instead of the two distinct
session paths returned by the fake hosted product.

Green restoration: document identity comes from exact `session_id` and duplicate records from one
session collapse at their first product rank.

## Full evaluation promotion threshold

Test node: `tests/test_aml_promotion_gate.py::test_promotion_gate_fails_closed_for_each_missing_or_bad_gate`

Production symbol: `scripts.aml_promotion_gate.decide`

Mutation: lower the net successful cell threshold from 8 to 7.

Observed assertion failure: a result with 7 net successful cells was promoted when the test
required refusal.

Green restoration: promotion requires at least 8 net successful cells and every other frozen gate.

## Context budget selection

Test node: `tests/test_aml_promotion_gate.py::test_context_selector_chooses_smallest_arm_within_one_point_of_best`

Production symbol: `scripts.aml_select_context.select_budget`

Mutation: widen the permitted coverage gap from one percentage point to two.

Observed assertion failure: the selector chose 7,000 characters for coverages 0.70, 0.80, and
0.82, while the preregistered one point rule requires 9,000.

Green restoration: the eligibility threshold is the best observed coverage minus 0.01.

## Hosted quality candidate width

Test node: `tests/test_aml_hosted.py::test_hosted_quality_is_a_real_fixed_product_profile`

Production symbol: `recall.profiles.HOSTED_QUALITY_PROFILE`

Mutation: reduce `candidate_k` from the frozen width 100 to 99.

Observed assertion failure: the resolved product profile reported candidate width 99 instead of
100.

Green restoration: `hosted-quality` resolves to the immutable 100 candidate profile.

## Supersession beyond the retrieved pool

Test node: `tests/test_aml_hosted.py::test_packer_filters_supersession_declared_outside_candidate_pool`

Production symbol: `recall_aml.retrieval.pack_evidence`

Mutation: ignore the tenant wide set returned by `explicit_superseded_chunk_ids` and inspect only
edges inside the retrieved candidates.

Observed assertion failure: the sole obsolete candidate was returned even though its successor's
stored edge named it outside the candidate pool.

Green restoration: the packer seeds its exclusion set from the tenant wide structural scan, then
adds edges visible inside the pool.

## Raw evidence segmentation

Test node: `tests/test_aml_hosted.py::test_raw_messages_are_segmented_without_losing_order_or_content`

Production symbol: `recall_aml.service.RAW_SEGMENT_CHARS`

Mutation: raise the segment ceiling from 6,000 to the 200,000 character request field ceiling.

Observed assertion failure: a 6,010 character message produced one embedding input instead of two.

Green restoration: raw content is split at 6,000 characters with ordered segment metadata, and the
test reconstructs the original content byte for byte from the stored chunk texts.

## OpenRouter credential selection

Test node: `tests/test_aml_hosted.py::test_hosted_settings_read_openrouter_key_not_legacy_openai_key`

Production symbol: `recall_aml.config.HostedSettings.from_env`

Mutation: populate `openrouter_api_key` from `OPENAI_API_KEY` instead of `OPENROUTER_API_KEY` while
both variables contain distinct sentinels.

Observed assertion failure: the runtime setting contained `legacy-key-must-not-win` instead of
`openrouter-key`.

Green restoration: the hosted settings read only `OPENROUTER_API_KEY` for generative requests and
leave the legacy OpenAI variable untouched.

## Provider-qualified generation model

Test node:
`tests/test_aml_hosted.py::test_openrouter_compiler_treats_prompt_injection_as_data_and_uses_fixed_model`

Production symbol: `recall_aml.config.GENERATION_MODEL`

Mutation: remove the OpenRouter provider prefix, changing `openai/gpt-4o-mini` to the bare
`gpt-4o-mini` identifier.

Observed assertion failure: the captured request contained `gpt-4o-mini` instead of
`openai/gpt-4o-mini`.

Green restoration: compiler and facet requests use the fixed OpenRouter identifier
`openai/gpt-4o-mini`.

## OpenRouter transport endpoint

Test node:
`tests/test_aml_hosted.py::test_openrouter_client_uses_fixed_compatible_endpoint_and_disables_sdk_retries`

Production symbol: `recall_aml.config.OPENROUTER_BASE_URL`

Mutation: route the OpenAI-compatible client to `https://api.openai.com/v1` instead of OpenRouter.

Observed assertion failure: the captured client configuration contained the OpenAI API URL rather
than `https://openrouter.ai/api/v1`.

Green restoration: the hosted generator client uses the fixed OpenRouter endpoint and keeps SDK
retries disabled so the compiler's bounded retry policy remains the sole retry owner.

## Pre-admission readiness state

Test node:
`tests/test_aml_hosted_preflight.py::test_prepare_allows_only_admission_values_to_be_pending`

Production symbol: `scripts.aml_hosted_preflight.inspect_environment`

Mutation: mark every configuration value required during the preparation phase, including the two
values that become available only when admission opens.

Observed assertion failure: preparation reported `ready` false instead of true while only
`RECALL_AML_DATABASE_URL` and `RECALL_AML_API_KEY` were absent.

Green restoration: preparation reports those two values as `pending_admission`; all provider and
release identity values remain mandatory.

## Launch readiness fails closed

Test node:
`tests/test_aml_hosted_preflight.py::test_launch_requires_every_runtime_value_and_does_not_accept_openai_substitution`

Production symbol: `scripts.aml_hosted_preflight.inspect_environment`

Mutation: require only the preparation values during launch and leave both admission values
optional.

Observed assertion failure: launch reported `ready` true after `RECALL_AML_DATABASE_URL` was
removed.

Green restoration: launch requires every preparation and admission value.

The same node received a second mutation that accepted `OPENAI_API_KEY` when
`OPENROUTER_API_KEY` was empty. The intended assertion failed because launch incorrectly reported
ready for the `OPENROUTER_API_KEY` case. The restored implementation never reads the legacy key.

## Preflight secret redaction

Test node: `tests/test_aml_hosted_preflight.py::test_preflight_output_never_contains_configuration_values`

Production symbol: `scripts.aml_hosted_preflight.inspect_environment`

Mutation: add each raw configuration value to its returned check object.

Observed assertion failure: the serialized preflight result contained a supplied secret sentinel.

Green restoration: every check contains only its variable name, required flag, and presence state.

## Published AML request shapes

Test node:
`tests/test_aml_hosted.py::test_official_aml_requests_accept_unix_milliseconds_and_choice_array`

Production symbols: `recall_aml.models.Message.parse_aml_timestamp` and
`recall_aml.models.SearchRequest.options`

Baseline: commit `4e276a107fdc0dac81afd12ba9ea9211d44e8a54` accepted only a strict datetime
for `messages[].timestamp` and required `options` to be an internal object.

Observed assertion failure: the exact Unix millisecond Add request returned HTTP 422 instead of
200. The same baseline also rejects AML's optional array of answer choices.

Green restoration: the public boundary accepts an integer Unix millisecond timestamp and an
optional list of choice strings, while rejecting loose boolean, float, or arbitrary string
timestamp coercion.

## Published AML Add response identity

Test node: `tests/test_aml_hosted.py::test_official_aml_add_response_echoes_required_identity`

Production symbol: `recall_aml.models.AddResponse`

Baseline: commit `4e276a107fdc0dac81afd12ba9ea9211d44e8a54` omitted `success`, `user_id`,
and `session_id` from Add responses.

Observed assertion failure: the required echoed Add identity was not a subset of the HTTP response.

Green restoration: Add echoes the four required contract fields. Product diagnostic fields remain
additive.

## Published AML Search content field

Test node: `tests/test_aml_hosted.py::test_official_aml_search_response_uses_content_field`

Production symbols: `recall_aml.models.SearchItem.content` and
`recall_aml.retrieval.pack_evidence`

Mutation: rename the public output field and constructor argument from `content` to `memory`, the
pre-fix product behavior.

Observed assertion failure: the returned evidence item contained `memory` and the required
`content` key was absent.

Green restoration: every Search item exposes its stored evidence under `content`.

## Registered attribution ladder

Test node:
`tests/test_aml_hosted.py::test_registered_variants_match_the_preregistered_single_feature_ladder`

Production symbol: `recall_aml.variants.VARIANTS`

Mutation: disable compilation in `A1_compiler`, making it behaviorally identical to `A0_raw`.

Observed assertion failure: the second arm reported `(False, False, False, False)` instead of the
registered `(True, False, False, False)` treatment tuple.

Green restoration: the executable registry names the seven locked arms and adds exactly the
compiler, facets, reranker, then the three packing budgets.

## Raw baseline stage isolation

Test node: `tests/test_aml_hosted.py::test_a0_raw_bypasses_compiler_facets_and_reranker`

Production symbol: `recall_aml.service.HostedService.add`

Mutation: execute the compiler branch unconditionally, including for `A0_raw`.

Observed assertion failure: the deliberately failing compiler invoked the deterministic fallback,
so `compiled_count` was one instead of zero.

Green restoration: A0 persists only raw chunks and does not call the compiler, facet planner, or
reranker.

## Unpacked arm output

Test node: `tests/test_aml_hosted.py::test_unpacked_variant_returns_more_than_product_pack_limit`

Production symbol: `recall_aml.service.HostedService.search`

Mutation: execute A4 evidence packing unconditionally for every arm.

Observed assertion failure: an A0 retrieval containing thirteen stored chunks returned twelve,
proving the A4 pack limit had leaked into the raw baseline.

Green restoration: A0 through A3 render full retrieval order after structural supersession
filtering. Only A4 uses bounded packing.

## Served variant identity

Test node: `tests/test_aml_hosted.py::test_http_contract_auth_version_health_delete_and_validation`

Production symbol: `recall_aml.app.create_app`

Mutation: omit the service variant from `/version`.

Observed assertion failure: `version.get("variant")` returned `None` instead of
`A4_pack_7000`.

Green restoration: `/version` exposes the exact active attribution variant beside the immutable
product and retrieval identities.

## Live model readiness

Test node:
`tests/test_aml_hosted.py::test_live_readiness_probes_every_model_stage_used_by_the_served_variant`

Production symbol: `recall_aml.readiness.verify_model_readiness`

Mutations, run separately: replace the embedding call with a correctly sized zero vector; omit the
compiler facet call; replace the reranker call with the input list.

Observed assertion failure for each mutation: the corresponding deliberately failing provider
escaped the probe, so the expected `RuntimeError` was not raised.

Green restoration: startup makes one bounded live call to the embedder and to each compiler or
reranker stage enabled by the served variant. A bad vector shape or reranker identity also refuses
readiness.

## Cross-process duplicate Add serialization

Test node: `tests/test_aml_hosted.py::test_duplicate_add_is_serialized_across_service_instances`

Production symbol: `recall_aml.service.HostedService.add`

Mutation: bypass the repository request lock and rely only on each service instance's in-memory
lock.

Observed assertion failure: both service instances reached compilation with no prior record, so
the compiler recorded `[0, 0]` instead of one invocation at `[0]`.

Green restoration: Add acquires the repository lock before reading the durable receipt and holds it
through compilation, persistence, and receipt recording.

## PostgreSQL advisory lock lifetime

Test node: `tests/test_aml_hosted.py::test_postgres_operation_lock_is_held_across_the_protected_body`

Production symbol: `recall.store.PgVectorStore.operation_lock`

Mutation: replace blocking `pg_advisory_lock` with `pg_try_advisory_lock` while ignoring its boolean
result.

Observed assertion failure: the ordered SQL assertion saw `pg_try_advisory_lock` where the blocking
lock must be acquired before the protected body.

Green restoration: one borrowed connection acquires the blocking advisory lock, holds it across the
protected body, unlocks the same signed key, then returns the connection.

## Release checkout immutability

Test node:
`tests/test_aml_release_manifest.py::test_verify_repository_rejects_a_dirty_tracked_checkout`

Production symbol: `scripts.aml_release_manifest.verify_repository`

Mutation: remove the Git status check after verifying the expected commit.

Observed assertion failure: the modified tracked file was accepted, so the expected `RuntimeError`
was not raised.

Green restoration: release generation refuses tracked and untracked checkout changes after proving
that `HEAD` is the exact requested commit.

## Release artifact content binding

Test nodes:
`tests/test_aml_release_manifest.py::test_manifest_binds_artifact_bytes_and_excludes_secret_values`
and `tests/test_aml_release_manifest.py::test_sha256_file_streams_exact_bytes`

Production symbol: `scripts.aml_release_manifest.sha256_file`

Mutation: hash the artifact path string instead of reading artifact bytes.

Observed assertion failures: the digest did not equal the expected content digest and two different
wheel payloads at the same path produced the same digest.

Green restoration: the manifest streams every artifact byte into SHA256, so a content change alters
the release identity.

## Immutable release receipt

Test node: `tests/test_aml_release_manifest.py::test_write_manifest_refuses_overwrite`

Production symbol: `scripts.aml_release_manifest.write_manifest`

Mutation: remove the destination existence guard before writing the manifest.

Observed assertion failure: the second write silently replaced the first receipt, so the expected
`FileExistsError` was not raised.

Green restoration: release manifest creation refuses an existing output path.

## Search fallback telemetry

Test node:
`tests/test_aml_hosted.py::test_search_fallback_headers_are_truthful_without_changing_the_aml_body`

Production symbols: `recall_aml.service.HostedService.search` and
`recall_aml.app.create_app`

Mutation: publish zero for both Search fallback headers even after the facet planner and reranker
raised and their deterministic fallback paths served the request.

Observed assertion failure: `X-Recall-Facet-Fallback` was `0` instead of `1`.

Green restoration: the service carries the two request-local fallback states to response headers
while Pydantic excludes them from the official AML JSON body.

## Search planner latency containment

Test node:
`tests/test_aml_hosted.py::test_search_facets_have_one_short_attempt_while_add_retains_bounded_retries`

Production symbol: `recall_aml.compiler.OpenAICompiler._json`

Mutation: give facet planning the compiler's three attempt policy instead of its dedicated one
attempt policy.

Observed assertion failure: the failing facet provider was called three times instead of once.

Green restoration: Search facets make one request with a two second timeout and no retry sleep,
while Add retains three bounded attempts with eight second request timeouts.

## Facet prompt release identity

Test node: `tests/test_aml_hosted.py::test_http_contract_auth_version_health_delete_and_validation`

Production symbols: `recall_aml.compiler.facet_prompt_digest` and
`recall_aml.app.create_app`

Mutation: omit the facet planner prompt digest from `/version` while retaining only the compiler
prompt digest.

Observed assertion failure: reading `facet_prompt_digest` raised `KeyError`.

Green restoration: `/version` and the release manifest bind separate SHA256 identities for the
compiler and facet planner prompts.

## Registered final population gate

Test node:
`tests/test_aml_promotion_gate.py::test_promotion_gate_fails_closed_for_each_missing_or_bad_gate`

Production symbol: `scripts.aml_promotion_gate.decide`

Mutation: remove the exact 34 task population check while leaving all measured outcome thresholds
unchanged.

Observed assertion failure: a 12 task artifact was promoted instead of refused.

Green restoration: Full promotion requires exactly 34 tasks, 102 paired cells, seeds 0 through 2,
zero invalid cells, and verified paired identities.

## Provider budget promotion gate

Test node:
`tests/test_aml_promotion_gate.py::test_promotion_gate_fails_closed_for_each_missing_or_bad_gate`

Production symbol: `scripts.aml_promotion_gate.decide`

Mutation: remove the provider spend ceiling from the mechanical gate.

Observed assertion failure: an artifact reporting 450.01 US dollars was promoted instead of
refused.

Green restoration: the gate reserves the final 50 dollars of the 500 dollar budget by refusing
promotion above 450 dollars of recorded spend.

## Cross-chunk compiler context order

Test node: `tests/test_aml_hosted.py::test_prior_session_records_are_read_in_ingest_order`

Production symbol: `recall.store.PgVectorStore.chunks_for_source`

Baseline: prior session chunks were ordered only by content-derived hash ID.

Observed assertion failure: the SQL contained `ORDER BY id` instead of
`ORDER BY indexed_at, id`.

Green restoration: the compiler receives earlier session records in stable ingestion order, with
ID used only as the deterministic tie break inside one transaction timestamp.

## Relevance core before session diversity

Test node:
`tests/test_aml_hosted.py::test_packer_keeps_multi_record_task_evidence_before_diversity_fill`

Production symbol: `recall_aml.retrieval.pack_evidence`

Baseline: place every first item from a new session before every repeated session item.

Observed assertion failure: eleven irrelevant sessions displaced `target-beta`, the second exact
fact from the best matching session, from the twelve item pack.

Green restoration: the strongest two thirds of the item budget remain in relevance order, then
the remaining tail prefers unseen sessions before repeated sessions.

## Raw rescue pack compatibility

Test node:
`tests/test_aml_hosted.py::test_every_raw_segment_fits_the_smallest_registered_pack_budget`

Production symbol: `recall_aml.service.RAW_SEGMENT_CHARS`

Baseline: raw content segments were capped at 6,000 characters while the smallest registered A4
pack was capped at 5,000 characters.

Observed assertion failure: the rendered raw chunk was 6,020 characters and could never pass the
pack budget check.

Green restoration: raw content segments are capped at 4,500 characters, leaving room for the
maximum role and timestamp prefix inside every registered context budget.

## Compiler claim grounding

Test node:
`tests/test_aml_hosted.py::test_compiler_removes_unsupported_outcome_validation_and_event_time`

Production symbol: `recall_aml.compiler.OpenAICompiler.compile`

Baseline: accept outcome and validation fields when only the record's evidence quote was verified.

Observed assertion failure: the unsupported `the deployment succeeded` outcome remained stored
instead of becoming an empty field.

Green restoration: nonempty outcome and validation text must occur verbatim in the supplied
messages, and an event time must equal a supplied message timestamp. Unsupported fields are
removed while grounded parts of the record remain useful.

## Provider JSON event time parsing

Test node:
`tests/test_aml_hosted.py::test_compiler_accepts_a_supported_event_time_from_provider_json`

Production symbol: `recall_aml.compiler.OpenAICompiler.compile`

Baseline: validate the parsed provider mapping in strict Python mode, where a JSON datetime arrives
as a string and cannot satisfy the strict `datetime` field.

Observed assertion failure: Pydantic raised a datetime validation error before the supported record
could be returned.

Green restoration: compiler payload validation uses Pydantic's strict JSON mode, which accepts the
JSON datetime representation, then retains it only when it equals an input message timestamp.

## Sixteen request p95 latency

Test node:
`tests/test_aml_hosted.py::test_concurrency_p95_counts_the_slowest_of_sixteen_requests`

Production symbol: `scripts.aml_hosted_verify.percentile`

Baseline: use a zero-based floor of `(n minus 1) times p` for the percentile index.

Observed assertion failure: p95 of request latencies 1 through 16 was reported as 15 instead of
16.

Green restoration: the external gate uses the nearest rank definition, so p95 of sixteen requests
includes the slowest request.

## Thirty minute concurrency soak

Test node:
`tests/test_aml_hosted.py::test_concurrency_soak_repeats_sixteen_by_sixteen_until_duration`

Production symbol: `scripts.aml_hosted_verify.verify_concurrency`

Mutation: break after the first concurrent Add and Search burst, reproducing the previous quick
check behavior.

Observed assertion failure: the duration-bound fixture reported one cycle instead of two and only
sixteen requests per operation instead of thirty-two.

Green restoration: the verifier repeats independent 16 by 16 bursts until the requested duration
has elapsed, defaulting to the registered thirty minutes, and reports actual duration and counts.

## PostgreSQL restart and deletion integration

Test node:
`tests/test_aml_hosted.py::test_postgres_add_replay_restart_search_and_tenant_delete`

Production symbol: `recall_aml.storage.PgHostedRepository.persist`

Mutation: report the materialized chunk count without embedding or upserting the chunks.

Observed assertion failure: after service reconstruction, Search returned no item containing
`ExactRestartEvidence`.

Green restoration: the real shared pool PostgreSQL path embeds and atomically upserts raw evidence,
persists the replay receipt across service reconstruction, and deletes one tenant without removing
the peer tenant.
