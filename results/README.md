# Results Map

This directory is the evidence record behind the README and [docs/EVIDENCE.md](../docs/EVIDENCE.md).
It is not the product entry path. Start with the summaries, then open raw artifacts only when you
need to audit or reproduce a number.

| Document | Use it for |
|---|---|
| [FINDINGS.md](FINDINGS.md) | Interpretation, limits, negative results, and corrected claims. |
| [RESULTS.md](RESULTS.md) | Complete published tables. |
| [ARTIFACTS.md](ARTIFACTS.md) | Mapping from quoted numbers to committed artifacts. |
| [WITHDRAWN.json](WITHDRAWN.json) | Machine readable list of withdrawn public figures. |
| [CLAIMS_BASELINE.json](CLAIMS_BASELINE.json) | Claim gate baseline used by CI. |

| Directory | Scope |
|---|---|
| [head_to_head/](head_to_head/) | RE-call versus comparator memory benchmark artifacts. |
| [locomo/](locomo/) and [locomo_rerank/](locomo_rerank/) | LOCOMO retrieval and reranking artifacts. |
| [gap/](gap/) | Embedder gap and abstention threshold studies. |
| [ladder/](ladder/) | Answerability ladder verdicts and compact artifacts. |
| [mtrag/](mtrag/) and [mtrag_generation/](mtrag_generation/) | MTRAG probes and generation summaries. |
| [promotion/](promotion/) | Generation promotion and parity artifacts. |
| [scale/](scale/) and [scale-pressure/](scale-pressure/) | Scale and pressure measurements. |
| [store_latency/](store_latency/) | Store latency measurements. |
| [wrrf/](wrrf/) | Weighted RRF follow-up artifacts. |

Raw logs and per-question JSONL payloads are intentionally excluded by `.gitignore`. Published
numbers must be backed by committed summaries or compact artifacts that the claim gate can inspect.
