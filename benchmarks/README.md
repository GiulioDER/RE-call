# RE-call benchmark map

This directory contains benchmark harnesses, pre-registrations, audit notes, and review documents.
Published result tables live in `results/`; benchmark code and protocols live here.

## Main paths

| Path | Purpose |
|---|---|
| [run.py](run.py) | LOCOMO memory benchmark runner. |
| [systems.py](systems.py) | RE-call and comparator adapters. |
| [pipeline.py](pipeline.py) | Generator, judge, aggregation, and shared records. |
| [latency.py](latency.py) | Isolated memory-layer latency measurement. |
| [enterprise_rag.py](enterprise_rag.py) | EnterpriseRAG-Bench answer-file runner for public leaderboard submission. |
| [agent_ab/](agent_ab/) | Paired application-level benchmark: one agent, RE-call on and off. |
| [claim_gate.py](claim_gate.py) | Published-number gate for result documents. |
| [ladder/](ladder/) | Answerability ladder benchmark. |
| [beam/](beam/) | BEAM harness integration and related probes. |
| [labelling/](labelling/) | Human arbitration data and scoring utilities. |
| [finetune/](finetune/) | Experimental embedding fine-tuning study support. |
| [../run_lme_s.sh](../run_lme_s.sh) | Crash-tolerant LongMemEval S runner kept at repository root because it anchors paths beside its output directory. |

## Protocol documents

| Document | Purpose |
|---|---|
| [PREREGISTRATION.md](PREREGISTRATION.md) | Rules fixed before the main RE-call versus Mem0 memory benchmark. |
| [archive/preregistrations/README.md](archive/preregistrations/README.md) | Archived preregistrations for follow-up benchmark arms. |
| [REVIEW.md](REVIEW.md) | Adversarial review of the LOCOMO article claims. |
| [SUITE-DESIGN.md](SUITE-DESIGN.md) | Evaluation suite design and benchmark tracks. |
| [../docs/VISIBILITY_BENCHMARKS.md](../docs/VISIBILITY_BENCHMARKS.md) | Public leaderboard submission track and artifact contract. |
| [EXPERIMENT-CONVENTION.md](EXPERIMENT-CONVENTION.md) | Prior-work search convention for new experiments. |
| [VOYAGE_REFERENCE.md](VOYAGE_REFERENCE.md) | Voyage model reference and experiment recommendations. |

## Result documents

Use these before quoting a number:

| Document | What it contains |
|---|---|
| [../results/RESULTS.md](../results/RESULTS.md) | Complete measured tables. |
| [../results/FINDINGS.md](../results/FINDINGS.md) | Interpretation, limits, and negative results. |
| [../results/ARTIFACTS.md](../results/ARTIFACTS.md) | Artifact-to-configuration map. |
| [../results/WITHDRAWN.json](../results/WITHDRAWN.json) | Known withdrawn figures. |

## Benchmark discipline

Every new benchmark should state the claim it can falsify, the prior work searched, the fixed
parameters, the artifacts it writes, and the limits that would make the result non-comparable.
Post-hoc analysis belongs in review notes, not in a pre-registration.

Before quoting a result in the README or docs, cite the result document and confirm the number is
covered by the claim gate or by a committed artifact.
