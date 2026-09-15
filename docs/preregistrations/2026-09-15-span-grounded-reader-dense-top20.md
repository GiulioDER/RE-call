# Span grounded reader over the original dense top 20

Status: predicted, not yet measured.

Registered 2026-09-15 before implementing the experiment harness, generating any reader score,
or inspecting any row level reader outcome.

## Question

The consumed empty base cohort already contains the frozen exact answer span somewhere in the
original dense top 20 for 10 of 15 answerable rows. Dense rank one contains it for only three rows,
and generic cross encoder and ColBERT selectors have not produced a safe first result. Can a
question answering reader choose an answer bearing chunk and a literal supported quote from the
fixed original pool, while returning null on matched unanswerable controls?

This differs from the failed guarded spare slot extractive experiment. That experiment used source
agreement to decide whether a newly proposed source should be appended to retrieval. This
experiment adds no source and generates no candidate. It reads only the already retrieved original
dense top 20 and selects at most one of those chunks.

This is a development screen on consumed data. It cannot authorize serving or a claim on unseen
data.

## Frozen population and lineage

Use the same 30 empty base rows from the completed query anchor holdout, split into 15 answerable
rows and 15 matched unanswerable controls. Reuse the already captured ColBERT collection without
issuing new retrieval calls. The query pool SHA256 is
`6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`, the holdout result SHA256 is
`58b883af8d1527ef137913762f9a3a456198c8f818d8286bfc87a248c4d3fa5b`, and the collection SHA256 is
`6cc9a85025a91bbea7841766bbf5a90200458ed8aa638bf0215c000f79ac099f`.

The captured lineage is generation `gen_2ccf2130f6c64d99a11a6bcb6f929dd8`, calibration
`cal_e50dac493112488ea5e7cf79d86c0099`, pipeline fingerprint
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus fingerprint
`f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312`.

The apparatus must reproduce the original dense gold source counts at ranks 1, 3, 5, 10, and 20
as 6, 6, 7, 8, and 10, and exact span counts as 3, 3, 5, 8, and 10. It must reproduce 14 answerable
rows and zero controls passing the existing corpus supported three anchor gate. Any mismatch is an
apparatus failure without a reader quality interpretation.

## Frozen reader

Use `deepset/roberta-base-squad2` at revision
`adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`. The model card identifies it as an English
Transformers question answering model trained on SQuAD 2.0 and licensed CC BY 4.0. Load the
safetensors weights with `AutoTokenizer` and `AutoModelForQuestionAnswering`. Record the exact
Transformers and Torch versions in the score artifact.

For every row, score the complete original question against each of the exact 20 stored chunk
texts. Tokenize question and context pairs with `truncation="only_second"`, maximum length 512,
document stride 128, overflowing windows enabled, and offset mappings enabled. Run the model in
evaluation and inference mode.

For each window, define the null score as the start logit plus end logit at the classification
token. Enumerate only spans whose start and end tokens belong to the context, whose end is not
before the start, whose length is at most 96 tokens, and whose character offsets form a nonempty
literal substring of the stored chunk. Define margin as span score minus null score. Retain the
highest margin span for each chunk, then the highest margin span across all 20 chunks. Resolve an
exact margin tie by original dense rank, then earliest character start. Return null when the best
global margin is at most zero. Otherwise return that chunk ID and the exact substring named by its
offsets.

The primary candidate is `anchor_gated_reader`: apply the reader result only when the existing
three anchor safety gate marks the row eligible, and return null otherwise. Also report
`raw_reader` descriptively without the gate so the model's native null behavior is visible. Gold
sources, expected answerability, and frozen answer spans must not appear in the VPS2 reader input or
affect scoring.

The candidate may promote its selected chunk to the first evidence position while preserving the
relative order and stored dense cosine of every other original candidate. It must not widen the
pool, use ColBERT order, mix logits with dense scores, generate text, or tune a null threshold.

All reader inference must run on VPS2 under the shared embedding lock and bounded process controls.
Check the lock and competing processes before starting. No reader inference may run on the
workstation.

## Measurements

For `raw_reader` and `anchor_gated_reader`, report answerable selections, control activations,
selected gold source rows, selected chunks containing the full frozen exact span, and returned
quotes that are literal substrings of the selected chunk. Report exact normalized quote equality
and SQuAD normalized token F1 against the frozen answer span as descriptive answer measures.

For the primary candidate, report how many selected chunks differ from dense rank one, how many of
those changes gain or lose a gold source, and how many gain or lose an exact bearing chunk. Report
the unchanged top 20 membership invariants.

Private output may contain row identifiers, questions, chunk text, raw logits, margins, selected
quotes, and gold annotations but must remain outside the repository. Public output contains only
aggregate counts and model identity.

## Prediction and decision

I predict `anchor_gated_reader` will select a gold source for at least 7 of 15 answerable rows and
an exact bearing chunk for at least 5 of 15, improving both dense rank one baselines of 6 and 3. It
must activate on zero controls, return only literal substrings, preserve all original top 20
memberships, produce at least two exact gains relative to dense rank one, and produce zero exact
losses relative to dense rank one.

Proceed to a fresh source disjoint reader validation only if every prediction passes. Otherwise do
not tune the model, null threshold, maximum answer length, window size, stride, anchor gate, pool,
or score combination on these consumed rows. On failure, close generic off the shelf reader
selection on this cohort and revisit corpus and gold construction or train a task specific selector
on source disjoint development data.
