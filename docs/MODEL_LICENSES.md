# Model licences

RE-call is **MIT**. It ships **no model weights**. Every model below is downloaded from
HuggingFace by the user at runtime, so RE-call redistributes nothing and this document is
guidance for adopters rather than a licence obligation of the package.

⚠️ Not legal advice. The MIT/Apache reading is uncontroversial; anything involving a
non-commercial model and a commercial product is a question for a lawyer.

📌 Every licence in this document was read off its model card on 2026-08-06, not inferred. The
same applies to any model added later — see the last section.

## Default stack — all permissive, safe to ship

| role | model | licence | notes |
|---|---|---|---|
| dense embedder | `BAAI/bge-small-en-v1.5` | MIT | default via `FastEmbedEmbedder` |
| learned sparse | `prithivida/Splade_PP_en_v1` | Apache-2.0 | `recall.sparse.DEFAULT_MODEL` |
| reranker (pinned default) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 | `recall.rerank.DEFAULT_RERANKER_MODEL` |
| reranker (stronger) | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | +0.010 to +0.017 nDCG@5 over MiniLM, ~25x compute |

**Nothing in this table restricts commercial use.** Every measured RE-call result on MTRAG comes
from this stack.

## Opt-in only — NON-COMMERCIAL, and a notice does not fix that

| model | licence | why it is registered |
|---|---|---|
| `naver/splade-v3` | **cc-by-nc-sa-4.0** | the checkpoint MTRAGEval rank 3 used. ⚠️ also a GATED repo: needs an approved HuggingFace account |
| `naver/splade-cocondenser-ensembledistil` | **cc-by-nc-sa-4.0** | ungated substitute for the above; MRR@10 38.3 vs the default's 37.22 |

🔑 **`CC-BY-NC-SA-4.0` bundles three separate things, and only one is an attribution problem:**

- **BY** — attribution. This is the part a credit line satisfies.
- **NC** — NonCommercial. A **use restriction**. No notice, credit or acknowledgement anywhere in
  your repository grants commercial rights. You either do not use it commercially, or you obtain
  different terms from the rights holder.
- **SA** — ShareAlike. Derivatives inherit the licence.

⚠️ **The restriction follows the OUTPUTS, not only the weights.** Encoding a corpus with an NC
model and serving commercial search from the resulting index is commercial use of that model,
even though no weights ship in the product. Whether the stored vectors are themselves "adapted
material" under SA is legally unsettled, which is an additional reason not to build a product on
one.

## How the code enforces this

`recall/sparse.py`:

- `KNOWN_MODELS` maps every supported checkpoint to its licence. An **unrecorded** model is
  refused by `from_pretrained`, so a new checkpoint cannot be used until someone looks its licence
  up and writes it down.
- Anything not `apache-2.0` requires `accept_noncommercial_license=True`, passed explicitly in
  code. The check runs **before** the download, so an accidental non-commercial dependency costs
  an error rather than 500MB and a licence problem discovered later.
- `DEFAULT_MODEL` is the Apache-2.0 one, and a test asserts that, so a future edit cannot quietly
  make a non-commercial model the default that everyone installs without choosing it.

## Research artifacts

The 2026-08-06 benchmark archive
(`/var/lib/recall-benchmarks/2026-08-06-mtrag-splade-learned-sparse/` on VPS2) contains
`vectors_v3.tar.gz`: a corpus encoded with `naver/splade-cocondenser-ensembledistil`.

⛔ **Research only. It was never loaded and never contributed to any published figure.** Do not
build a served index from it. Every number in that archive comes from the Apache-2.0 default.

## Adding a model

1. Look up the licence on the model card. Do not guess it, and do not infer it from a sibling
   model in the same organisation — `naver/splade-v3` and `Splade_PP_en_v1` are both SPLADE and
   have different licences.
2. Add it to `KNOWN_MODELS` with that licence string.
3. If it is not `apache-2.0` (or another permissive licence), say so here and expect callers to
   pass the opt-in flag.
4. Note whether the repo is **gated**. `naver/splade-v3` returns 401 without an approved account,
   which is a separate obstacle from its licence and is not detectable from the licence field.
