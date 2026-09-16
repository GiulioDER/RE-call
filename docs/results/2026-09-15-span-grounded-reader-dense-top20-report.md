# Span grounded reader over dense top 20 result

Measured 2026-09-15 on the consumed 30 row empty base development cohort. The result is complete
and failed the frozen promotion gate. The preregistered plan is
[2026-09-15-span-grounded-reader-dense-top20.md](../preregistrations/2026-09-15-span-grounded-reader-dense-top20.md).

## Verdict

`STOP_OFF_THE_SHELF_READER_ON_THIS_COHORT`.

The raw reader selected an answer on 8 of 15 answerable rows and activated on 9 of 15 matched
unanswerable controls. It selected a gold source for 3 answerable rows and an exact bearing chunk
for 2. Its native SQuAD 2.0 null margin therefore does not transfer safely to this memory corpus.

The primary `anchor_gated_reader` returned null for every ineligible row, which removed all control
activations. It still selected only 3 gold sources and 2 exact bearing chunks, below the original
dense rank one baselines of 6 and 3. It made five nontrivial selections, produced one gold gain and
four gold losses, produced zero exact gains and one exact loss, and matched no complete frozen
answer span. Its mean selected quote token F1 was 0.166.

Every returned quote was a literal substring of its selected chunk and all 30 candidate memberships
were preserved. The failure is model relevance, not grounding or pool corruption.

## Interpretation

An off the shelf SQuAD 2.0 reader is the wrong selector for these generated memory questions and
paragraph sized answer spans. The model both over answers matched controls and prefers plausible
short local phrases over the chunk containing the complete recorded fact. The existing anchor gate
solves the first problem by rejecting controls before inference, but it supplies no positive
evidence that the reader chose the right chunk.

Do not tune the null threshold, maximum answer length, windowing, checkpoint, anchor gate, pool, or
score mixture on these consumed rows. The generic reader lane is closed together with generic cross
encoder and ColBERT first result selection.

The next highest ROI mechanism is a task specific query to evidence selector trained from
automatically generated, source disjoint extractive examples. Positives should be the exact bearing
chunk. Hard negatives should come from the same query's dense top 20, and matched absent identifier
questions should train the explicit null class. The first step should be a no inference gold shape
and split audit across the existing 250 paired extractive pool to establish answer length,
template, source, and chunk boundary distributions before freezing a model or training objective.
This can show whether the available automatic gold supports a source disjoint train and validation
design without spending another holdout.

A lower ROI alternative is to index structured facts as independent heading conditioned chunks.
That can raise the present 10 of 15 top 20 exact ceiling, but it requires a new generation,
embeddings, lineage, and calibration, and it risks adding redundant chunks before selection quality
has been solved.

## Integrity and execution

The reader was `deepset/roberta-base-squad2` at pinned revision
`adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`, using Transformers 5.15.0 and Torch 2.13.0. The
registered model card licence is CC BY 4.0. Reader inference ran only on VPS2 under the shared
embedding lock, an 8 GB memory cap, zero swap, a 250 percent CPU quota, four Torch threads, and
lowered priority. It completed 30 rows and 600 question chunk pairs in 442.4 seconds.

The private reader input contained only query IDs, questions, chunk IDs, dense ranks, and chunk
text. It contained no gold source, exact span, expected answerability, answer span, or eligibility
field. Its SHA256 is `1f81521f1a56114fa27c97bc346231131d67eb0470cf5f0c35569c5332581258`.
The reader output SHA256 is
`ffb25600a22de2a9439daa4562a850c10d3d3027acf598a88b2b2972738d9056`.

The first local report attempt was stopped before aggregation because the reused collection
correctly omitted the frozen answer text. The harness was repaired to join each answer span from
the already hashed query pool by query index only after model scoring. A behavioral red proof used
query zero for every row and failed on row one before the correct join was restored. No model
parameter or reader output changed.

The private row level result remains outside the repository under
`C:\Users\gde00\.codex\evals\query-anchor-2026-09-15\span-grounded-reader\development.json`,
SHA256 `d97c1b08915077c82b0f9b10be05a89595f76f0666fd6d5e335982eb5b348e40`.
