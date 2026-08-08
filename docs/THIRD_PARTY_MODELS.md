# Third-party models

RE-call itself is MIT. The checkpoints it can load are not all MIT, and two of them carry
obligations that survive into anything published with them. This file is the repo-level discharge
of the Attribution term; the machine-readable source of truth is `recall.sparse.KNOWN_MODELS`, and
`tests/test_learned_sparse.py` fails if a model is recorded there but missing here.

The per-result credit line is emitted by `recall.sparse.attribution_notice()` and written into each
benchmark artifact's `sparse_attribution` field, so the obligation travels with the number rather
than living only in this file.

## Shipped default

### `prithivida/Splade_PP_en_v1`

- **Creator**: Prithivi Da
- **Source**: <https://huggingface.co/prithivida/Splade_PP_en_v1>
- **Licence**: `apache-2.0` — <https://www.apache.org/licenses/LICENSE-2.0>
- **Changes**: none. Loaded unchanged and used for inference only, to produce term weights.

This is RE-call's default (`recall.sparse.DEFAULT_MODEL`) and the only one usable commercially. It
is what every shipped number should be read against.

## Benchmark-only, non-commercial

Both models below are **CC BY-NC-SA 4.0**, which permits research and benchmark reproduction and
forbids commercial use. Neither is shipped with RE-call, neither is the default, and reaching
either requires passing `accept_noncommercial_license=True` explicitly. `SpladeEncoder.from_pretrained`
refuses them otherwise, so the choice cannot be made by accident or by a default someone changed.

### `naver/splade-cocondenser-ensembledistil`

- **Creator**: Naver Corporation
- **Source**: <https://huggingface.co/naver/splade-cocondenser-ensembledistil>
- **Licence**: `cc-by-nc-sa-4.0` — <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- **Changes**: no modification to the model. Weights loaded unchanged and used for inference only.
  The top-k pruning RE-call applies acts on this run's **output vectors**, not on the checkpoint.

Used as the "is our checkpoint the weak link?" control, because it is stronger than the default
(MS MARCO MRR@10 38.3 vs 37.22) while sharing the same BERT MLM architecture and 30522 vocabulary,
so it is a drop-in for the same encoder.

### `naver/splade-v3`

- **Creator**: Naver Corporation
- **Source**: <https://huggingface.co/naver/splade-v3>
- **Licence**: `cc-by-nc-sa-4.0` — <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- **Changes**: no modification. Inference only.

A **gated** repository: access needs an approved HuggingFace account, not merely the licence flag.
`splade-cocondenser-ensembledistil` is the ungated substitute.

## The four terms, and how each is discharged

| Term | How |
|---|---|
| **Attribution** | Creator, source link, licence link and a statement of changes, recorded in `KNOWN_MODELS`, rendered by `attribution_notice()`, and written into every artifact's `sparse_attribution`. Nothing here implies Naver endorses RE-call. |
| **NonCommercial** | Enforced in code, not by policy: `from_pretrained` refuses a non-`apache-2.0` checkpoint unless the caller opts in by name. These models are excluded from the shipped default and from any commercial path. Benchmark reproduction and research are the only uses made of them. |
| **ShareAlike** | Binds adaptations. RE-call does not redistribute these checkpoints or the sparse vectors derived from them; the vectors live in a local sidecar table and are not published. **Should those vectors ever be distributed, they must carry CC BY-NC-SA 4.0**, and `attribution_notice()` says so in the artifact. |
| **No additional restrictions** | RE-call adds no term, DRM, or technical measure that would restrict what the licence permits. The `accept_noncommercial_license` flag gates *our* default behaviour to prevent an accidental licence violation; it does not restrict anyone's rights under the licence, and the code is MIT. |

## When adding a model

Record it in `recall.sparse.KNOWN_MODELS` with all five fields and add a section here. An
unrecorded licence is refused at load time rather than assumed permissive, which is the point: the
failure mode this prevents is a non-commercial checkpoint becoming a default nobody chose.
