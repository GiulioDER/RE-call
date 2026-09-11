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
