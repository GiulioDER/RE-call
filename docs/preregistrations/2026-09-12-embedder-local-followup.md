# Preregistration: Voyage 4 versus local follow up embedders

Status: protocol locked before measurement on 2026-09-12.

## Objective

Measure whether locally hosted embedding models improve gold evidence retrieval over the current
Voyage 4 control on the same frozen LOCOMO benchmark used by the committed Voyage 3 versus Voyage 4
comparison. This is a retrieval only experiment. It does not measure generated answer quality,
graph expansion, reranking, latency, or provider cost.

## Frozen inputs

* Dataset: `locomo10.json`, 10 conversations and 1,986 source questions.
* Dataset SHA256: `79FA87E90F04081343B8C8DEBECB80A9A6842B76A7AA537DC9FDF651EA698FF4`.
* Source runner: `scripts/run_locomo_embedder_comparison.py` from the Voyage comparison branch.
* Retrieval: same RE-call indexing and retrieval path, fresh isolated table per arm, `k=5`,
  `candidate_k=20`, and depth curve `1,3,5,10,20`.
* Gold labels: answerable LOCOMO categories 1 through 4, using exact evidence turn identifiers.
  Category 5 remains excluded from gold retrieval scoring.
* Control: `voyage:voyage-4`.
* Treatments:
  * `fastembed:jinaai/jina-embeddings-v3`
  * `fastembed:intfloat/multilingual-e5-large`
  * `fastembed:mixedbread-ai/mxbai-embed-large-v1`

Each treatment is compared independently with Voyage 4. The same dataset and question identities
must be present in both arms. No reranker is enabled, so this isolates first stage embedding quality.

## Predictions

1. Voyage 4 will remain the best pooled hit@5 arm. None of the local treatments is predicted to
   exceed Voyage 4 by more than 2 percentage points.
2. Jina Embeddings v3 is predicted to be the strongest local treatment, because it is multilingual,
   retrieval task aware, and 1,024 dimensional.
3. Multilingual E5 large is predicted to improve over the old local bge-small baseline but remain
   below Voyage 4 on pooled hit@5.
4. mxbai embed large is predicted to be competitive with multilingual E5 on English questions but
   not to exceed Voyage 4 on pooled hit@5.
5. If any treatment improves hit@5, the improvement is expected to come primarily from moving gold
   turns into ranks 1 through 5, not from recovering many questions absent from the top 20.

## Primary gate

For each treatment independently, the gate is passed when the paired bootstrap 95 percent interval
for treatment minus Voyage 4 hit@5 excludes zero and the point estimate is at least 2 percentage
points. Results are exploratory across the three treatment comparisons and are not pooled into a
single multiplicity adjusted claim.

## Secondary outcomes

Report pooled and per category hit rate at 1, 3, 5, 10, and 20, complete evidence coverage, MRR of
the first gold turn, paired rescues and regressions, and miss attribution by first gold depth:
1 through 5, 6 through 10, 11 through 20, or absent by 20.

## Operational constraints

Run on VPS2, one embedding or indexing process at a time, with the existing bounded environment.
Do not mutate the production corpus. If a model cannot construct or complete within the installed
resource envelope, record it as an operational failure and do not substitute a different model.

