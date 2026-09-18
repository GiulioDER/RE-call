# AML coding memory matrix red proof receipts

Date: 2026-09-18

No benchmark measurement had started. Each receipt below ran with exactly three pytest workers
against a temporary production-code mutation. The mutation was restored with `apply_patch` before
the corresponding green run.

## Hosted behavior mutations

Command node ids:

1. `test_coding_matrix_uses_registered_context4_identity`
2. `test_learned_sparse_is_an_added_leg_and_queries_use_query_encoding`
3. `test_task_conditioned_packing_changes_kind_priority_without_gold_labels`
4. `test_c2_retains_raw_evidence_beside_grounded_procedure_memory`

Mutations changed the embedding identity to `voyage-4`, omitted learned sparse queries, ignored
the task type in packing, and made C2 compiled-only. Pytest reported four intended failures:
`voyage-4 != voyage-context-4-v1`, zero learned sparse calls instead of two, feature packing chose
the repair record, and C2 returned zero raw chunks. After restoration, the same four node ids
reported `4 passed in 27.63s`.

The next mutation omitted sparse sidecar persistence and replaced the C4 planner class with
`unknown`. The exact tests
`test_repository_persists_sparse_sidecars_before_add_can_acknowledge` and
`test_c4_uses_query_only_task_plan_and_surfaces_routing_class` failed with incomplete coverage
`dense=1 sparse=0` and `unknown != bugfix`. Both passed in the subsequent full hosted suite.

## Benchmark and selector mutations

The replay mutation accepted an absent task routing header as `unknown`.
`test_replay_refuses_missing_task_routing_telemetry` failed because no `RuntimeError` was raised.
The retrieval selector mutation overwrote its computed winner with C0.
`test_selector_applies_every_incremental_retrieval_gate_in_order` failed with
`C0_raw_lexical != C4_task_pack`.

The screen selector received the same winner-overwrite mutation.
`test_screen_selector_uses_success_then_frozen_retrieval_tiebreakers` failed with
`C0_raw_lexical != C4_task_pack`.

The final confirmation selector was mutated from the frozen eight net win threshold to nine.
`test_final_selector_requires_eight_net_wins_and_complete_admission` failed at exactly eight net
wins, proving the boundary is exercised rather than merely reported.

Finally, the pilot mutation removed `xs-` from explicit task selection and restored the unsafe
hosted adapter limits of 16 concurrent Add calls, 45 seconds for Add, and 10 seconds for Search.
The two exact contract tests failed on the missing prefix and `16 != 1`. The mutation was restored
before the green harness run.

## Corpus cache mutations

The sparse backfill mutation called the dense passage embedder before writing the SPLADE sidecar.
With exactly three workers,
`test_sparse_backfill_reuses_existing_dense_corpus_without_embedding_it_again` failed on the
sentinel assertion `dense corpus must not be reembedded during SPLADE backfill`. After restoration,
the same node passed in 15.99 seconds.

The hosted adapter mutation ignored `AMB_RECALL_HOSTED_REUSE_CORPUS=1` and followed its ordinary
Delete plus Add path. With exactly three workers,
`test_adapter_reuses_dense_corpus_and_backfills_only_sparse` failed because the observed calls
were Delete and Add instead of the single sparse backfill request. After restoration, the same
node passed in 17.13 seconds.

The empty-cache mutation accepted zero dense chunks and zero sparse chunks as complete coverage.
`test_sparse_backfill_refuses_an_empty_dense_corpus` failed because no exception was raised. The
matching adapter and replay mutations removed the positive sparse count requirement;
`test_adapter_refuses_to_reuse_an_empty_corpus` and
`test_replay_refuses_to_reuse_an_empty_corpus` both failed because no exception was raised. All
three mutations were restored before the green suite.

## Sparse backfill transport mutation

The first live C1 preparation showed that the general 180 second request window cannot contain a
2,284 chunk CPU SPLADE backfill. The mutation set both dedicated sparse backfill transports back
to 180 seconds. With exactly three workers,
`test_sparse_backfill_transport_outlives_the_observed_cpu_envelope` failed with
`180.0 >= 7200.0`, and `test_adapter_uses_the_dedicated_sparse_backfill_timeout` failed with
`180.0 == 7200.0`. The selector fixtures also refused the changed timeout identity. The mutation
was restored before the green run.

## Green receipts

1. RE-call hosted suite: 53 passed, 1 database-only skip, in 15.62 seconds.
2. AMB replay, selection, and pilot contracts: 36 passed, 5 environment skips, in 18.76 seconds.
3. Python Ruff checks passed in both worktrees.
4. Bash syntax checks passed for the setup, replay, screen, and final orchestration scripts.
5. Mypy passed over all 215 RE-call source files.
6. The amended AMB replay, adapter, and selector suite passed 27 tests with exactly three workers;
   Ruff and the frozen JSON parse check also passed.
