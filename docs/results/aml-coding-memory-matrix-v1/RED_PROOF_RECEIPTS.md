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

## Green receipts

1. RE-call hosted suite: 51 passed, 1 database-only skip, in 28.11 seconds.
2. AMB replay, selection, and pilot contracts: 25 passed, 5 environment skips, in 16.88 seconds.
3. Python Ruff checks passed in both worktrees.
4. Bash syntax checks passed for the setup, replay, screen, and final orchestration scripts.
