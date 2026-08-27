# Preregistration: original model query construction

Date: 2026-08-25

## Question

Can RE-call improve memory recovery by asking the original agent to restate what memory it needs before retrieval, while deterministic controls prevent model text from becoming evidence?

## Frozen population

Use the same `agent-ab-skill-001` population as the earlier formulation gap work:

* 15 frozen miss sessions.
* 31 frozen hit controls.
* The original user prompt and first recorded memory query are fixed inputs.
* The governing memo labels are used only for scoring after retrieval, never in prompts or query construction.

The prior query side expansion result was 3 of 15 rescues and is a historical comparator, not a new control arm.

## Arms

1. `baseline`: the original recorded memory query.
2. `original_loop`: after the initial retrieval, ask the original model for a bounded task frame and one revised query. Retrieve the revised query once. The original model sees the user prompt and bounded trusted evidence, but no gold label.
3. `pyramid`: ask for the same task frame, validate its schema, generate up to three query candidates across literal, intent, anchor, and decomposition forms, then retrieve accepted candidates. Run one further original model challenge only when the controller reports a gap or zero new trusted items.

All model outputs are proposals. Only retrieved chunks that pass the normal tenant, generation, calibration, and trust checks may count as evidence.

## Control rules

* Maximum two construction rounds.
* Maximum three candidate queries per round.
* Maximum one original model challenge per round.
* Maximum 2,000 characters per query.
* Duplicate and zero novelty queries are rejected.
* Parent chunk references must point to trusted evidence from the same retrieval generation.
* A task frame is never cited and never passed to the answer model as evidence.
* The controller stops when the original model says the memory need is satisfied, or when the round budget is exhausted.

## Primary metrics

* Governing memo recovery at top 5 on the 15 misses.
* Rescue count relative to baseline.
* Hit retention on the 31 controls.

Secondary metrics are query novelty, accepted and rejected candidates, new trusted chunks, original model calls, retrieval calls, latency, and token cost.

## Decision rule

Prefer `pyramid` if it rescues at least 5 of 15 misses, retains at least 30 of 31 controls, and uses no more than two retrieval rounds per session. Prefer `original_loop` only if it reaches the rescue bar and pyramid does not. Otherwise keep the current path and treat the result as evidence that the missing vocabulary must be introduced at index time or through a separate memory representation.

## Apparatus gate

The live comparison is invalid if the original prompt is not preserved, if retrieval generations differ within a session, if the trusted evidence binding is missing, or if the calling model cannot receive the challenge and return a frame. These cases are reported separately and do not count as retrieval failures.
