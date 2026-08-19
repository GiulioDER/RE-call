# ATM answer cap second addendum

Status: preregistered before resuming answer generation.

The resumed run with a 2,048 token ceiling completed three additional answers after the first
checkpoint, reaching 18 answers, then stopped when the next answer reached the ceiling. The
retrieval checkpoint for that question was preserved and the truncated answer was rejected.

## Change

The resumed run changes only the answer output ceiling from 2,048 to 4,096 tokens. Retrieval,
reranking, evidence selection, model, reasoning request, prompt, checkpoint files, and official
judge configuration remain unchanged.

The completed Voyage index is reused with `--reuse-index`. The 18 valid existing predictions are
preserved. The resumed run starts at the first question without an answer checkpoint.

## Prediction

I predict that the 4,096 token ceiling will avoid another output ceiling failure on the remaining
questions and that the resumed run will produce a complete prediction file for all 1,013 questions.
I predict that the answer score will not decrease solely because of this ceiling change.

## Result

No resumed answer generation measurement has been run at the time this addendum was committed.
