# Codex and RE-call paired benchmark

This document is a template. Copy it to the run artifact directory and complete it before
invoking `run_paired`. Do not fill result fields until the paired run has finished.

## Claim under test

Primary claim:

> On the fixed task set below, RE-call changes task success while reducing or increasing measured
> model token use and wall time by the stated deltas.

The result is only valid for the exact model, prompt, repository snapshot, task manifest, and
resource limits recorded here.

## Fixed configuration

| Field | Value |
|---|---|
| Run identifier | `TODO` |
| Task manifest and hash | `TODO` |
| Repository revision | `TODO` |
| Model and version | `TODO` |
| Sampling parameters | `TODO` |
| System prompt revision | `TODO` |
| Working directory policy | `isolated per arm` |
| RE-call memory snapshot or tenant | `TODO` |
| Host and resource limits | `TODO` |
| Pair concurrency | `1` unless justified here: `TODO` |
| Repetitions per task | `TODO` |

The `recall_on` and `recall_off` arms must receive the same task input, model configuration,
prompt, repository snapshot, and resource limits. Only RE-call access may differ.

## Primary endpoints

Compute paired deltas as `recall_on - recall_off`:

1. Task success rate, where higher is better.
2. Total model tokens, where lower is better.
3. Wall time in milliseconds, where lower is better.

Secondary endpoints are input tokens, output tokens, model turns, tool calls, RE-call call count,
RE-call latency, system cost, evaluator cost, abstention, trust verdicts, and Ragas scores.
Missing values remain null and are excluded from that metric's paired calculation.

## Quality evaluation

Ragas version and evaluator model:

`TODO`

Selected metrics and exact configuration:

`TODO`

Ragas evaluator usage is reported separately from the application under test. Deterministic
RE-call retrieval, trust, supersession, and abstention measures remain the primary memory layer
measures.

## Exclusions and stopping rules

Record every failed case. A task is excluded from paired metric summaries when either arm fails or
is missing. Do not replace a missing measurement with zero. Record any excluded task IDs and the
reason in the final artifact.

No stopping rule, task removal, prompt change, model change, or metric change may be introduced
after inspecting the results. Any exploratory rerun receives a new run identifier and
preregistration.

## Artifacts

The run must preserve:

1. This completed preregistration.
2. The task manifest and its hash.
3. Raw JSONL `SessionRecord` rows.
4. The paired summary and summary code revision.
5. Model configuration and evaluator configuration.
6. Failed cases, incomplete task IDs, and environment information.
