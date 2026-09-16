# Task specific dense hard negative selector result

Measured 2026-09-16 on the preregistered 33 row validation split. The protocol is
[2026-09-16-task-specific-dense-hard-negative-selector.md](../preregistrations/2026-09-16-task-specific-dense-hard-negative-selector.md).

## Verdict

`STOP_TASK_SPECIFIC_SELECTOR_VALIDATION`.

The validation gate failed, so the sealed 29 row internal test was not uploaded to VPS2, scored, or
summarized.

## Result

Original dense rank one selected an exact bearing chunk for 16 of 33 rows and a gold source for 23
of 33. The pinned generic MiniLM selected 15 exact chunks and 17 gold sources. The task specific
model selected 15 exact chunks and 18 gold sources.

Relative to dense rank one, the trained selector changed 21 first results. It made five exact gains
and six exact losses, for changed selection exact precision 0.238. It made five gold gains and ten
gold losses, for changed selection gold precision 0.333. Every candidate membership was preserved.

The trained model did not fail through a no op. Its sampled weight moved by
`0.00046992721036076546`, and its ordering differed from the base model on all 33 validation pools.
Training improved generic MiniLM gold rank one by one row, but it did not recover the stronger dense
ordering and did not improve exact rank one at all.

Dense exact reach was 16, 22, 24, 26, and 26 at ranks 1, 3, 5, 10, and 20. Dense gold reach was 23,
27, 28, 28, and 28. The trained selector exact reach was 15, 19, 20, 23, and 26, while gold reach
was 18, 26, 28, 28, and 28. It therefore damaged the early ranks while preserving only the fixed
top 20 ceiling.

## Interpretation

Real dense negatives satisfied the explicit reentry condition left by the earlier failed BM25
negative fine tune, but they did not make MiniLM a reliable first result selector. Binary relevance
training on 186 automatic questions and 744 hard negatives learned a different ordering, not a
better one.

One plausible mechanism is that the automatic source title templates and whole chunk labels teach
coarse topical plausibility more readily than exact recorded fact selection. This is an inference,
not an established causal result. The measured conclusion is narrower: this pinned model, data,
objective, and real dense negative recipe does not improve exact or gold first result quality.

Do not tune epochs, learning rate, batch size, negative count, maximum length, score mixing, or the
MiniLM checkpoint on these consumed rows. The generic and task specific MiniLM selector lane is
closed for this pool.

## Way forward

Selection has now failed with a generic cross encoder, ColBERT MaxSim, an extractive reader, and a
task specific cross encoder. The next highest ROI work moves upstream to corpus representation.
The measured dense top 20 ceiling still misses seven exact chunks and five gold sources in only 33
rows, while every learned selector tested has damaged rank one. A structured fact chunk view can
put the recorded fact and its heading or field label in one independently retrievable unit before
asking another model to reorder candidates.

The next phase should first run a no embedding construction audit over the same source disjoint
positive pool. It should measure how many field and heading facts can become nonoverlapping,
heading conditioned fact chunks, how much duplicate text they add, and whether each frozen span is
preserved. Only if that audit passes should a new Context 4 generation be built and compared on a
fresh source disjoint retrieval set.

## Integrity and execution

The model was `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`, trained for two frozen epochs on VPS2 with Torch
2.13.0, Sentence Transformers 5.7.0, and Transformers 5.15.0. It ran under the shared model lock,
an 8 GB memory cap, zero swap, a 250 percent CPU quota, four Torch threads, and lowered priority.

The private collection SHA256 is
`6ef22d973553b847bee46b270139c795886ee54e96b13a8bd03d718d8ad6e4c7`. The private validation
score SHA256 is `f0e2bf2f96c25d69e5251b7088201d7027b9da296ab8535eacf2b7693ab23391`.
The saved model digest is
`1d216200fa3cba914cd19f3be5dceea80f8a7139b5a4212fca42b213181e3546`. Row level artifacts and
weights remain private.

