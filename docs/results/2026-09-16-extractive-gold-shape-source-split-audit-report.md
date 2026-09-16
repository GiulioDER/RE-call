# Extractive gold shape and source split audit result

Measured 2026-09-16 on the consumed 250 answerable plus 250 matched control extractive pool. The
frozen protocol is
[2026-09-16-extractive-gold-shape-source-split-audit.md](../preregistrations/2026-09-16-extractive-gold-shape-source-split-audit.md).

## Verdict

`GO_POSITIVE_SELECTOR_ONLY`.

The positive ranking supervision passed every frozen gate. The control supervision failed every
leakage gate and must not be used to train or evaluate an explicit null class.

## Positive supervision

All 250 answerable rows had an available source with the frozen SHA256, and every frozen span was
found in exactly one reconstructed default chunk at its recorded ordinal. There were 250 unique
source paths, 250 unique source content hashes, and 250 unique normalized questions. Exact source
path and source content overlap with the later query anchor comparison pool were both zero.

The deterministic source content split produced 186 train, 34 validation, and 30 internal test
answerable rows. Each split contains all three construction families. Source paths, source content
hashes, and normalized questions have zero overlap across every split pair.

The pool contains 132 fallback, 60 heading, and 58 field examples. Answer spans range from 44 to
320 characters, with a median of 192. They range from 8 to 57 words, with a median of 27. The
corpus mix is 178 `sentiment-agent` sources and 72 `recall` sources.

## Control leakage

The frozen absent identifier rule and the introductory phrase rule each achieved 1.000 balanced
accuracy over all 500 rows. `according` and `item` occur in every control and no answerable query.
These rows measure whether a system detects the synthetic construction, not whether it recognizes
natural absence. Training a null class on them would create a misleadingly easy success.

## Limitations found

One answer span hash is shared between validation and internal test even though the sources and
questions are disjoint. The next model experiment should remove every repeated answer hash from
all evaluation splits before training or scoring, with the exclusion frozen in its preregistration.

The minimum span to chunk boundary clearance is zero for at least half the pool, and 184 of 250
spans have less than 40 characters of clearance. This does not invalidate a whole chunk relevance
label because every complete span is present, but it does make the pool a poor basis for training a
short span reader. The next experiment should rank complete chunks rather than predict answer
offsets.

This is a data feasibility result on consumed development material. It is not evidence that a task
specific selector improves retrieval, and it does not authorize serving.

## Next experiment

Collect the current production dense top 20 for only the deduplicated answerable rows. Train a
small query and chunk cross encoder on train sources using the exact bearing chunk as the positive
and same query dense competitors as hard negatives. Use validation only for a single frozen model
selection rule, then compare dense rank one with selector rank one once on the internal test split.

Do not include the matched absent identifier controls. Null handling remains unchanged until a
separate natural unanswerable set exists. This next run is the explicit reentry condition left open
by the earlier failed MiniLM fine tune, which used BM25 negatives rather than real retrieval pool
negatives.

## Integrity

The result artifact is
`docs/results/2026-09-16-extractive-gold-shape-source-split-audit.json`. It contains aggregate counts
only. The audit used no model, embedding, retrieval, or external inference.

