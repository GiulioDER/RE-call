# Operating Modes

RE-call is deliberately configurable because memory deployments differ by legal boundary, hardware,
latency target, quality bar, and cost model. The default path keeps retrieval local. Hosted
embedding, reranking, and stricter serving profiles are opt-in choices after measurement.

| Mode | Use when | Trust policy | Egress | Typical configuration |
|---|---|---|---|---|
| Local demo | You want to try the product surface on the bundled corpus. | Development, with degraded confidence marked clearly. | None with FastEmbed. | `RECALL_TRUST_MODE=development`, `RECALL_EMBEDDER=fastembed`. |
| Local production | You need private memory over a controlled corpus. | Strict, with a generation-bound calibration. | None with FastEmbed and local artifacts. | `RECALL_ENV=production`, `RECALL_RETRIEVAL_PROFILE=fast`, serving and migration DSNs split. |
| Quality production | Answer quality is worth higher latency or local model cost. | Strict, with a calibration fitted for that embedder and corpus. | None with local artifacts. Possible egress if a hosted embedder is selected. | `RECALL_RETRIEVAL_PROFILE=quality`, pinned reranker path and digest. |
| Hosted embedder | Legal and privacy review permits external embedding calls. | Strict or development, depending on whether calibration has been promoted. | Query and corpus text may leave the environment. | Select the hosted embedder and document the approved egress path. |
| Evaluation | You are reproducing published evidence or comparing corpus options. | Explicit benchmark policy or development mode. | Depends on the selected embedder. | Use `benchmarks/`, `results/`, and a run-specific configuration record. |

Selection rule: start local, calibrate on the real corpus, measure the failure modes, then opt into
hosted embedding or reranking only when the measured gain pays for egress, latency, and operating
cost.
