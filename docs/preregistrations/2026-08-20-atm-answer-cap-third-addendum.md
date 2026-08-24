# ATM answer cap third addendum

Status: preregistered before resuming answer generation.

The 4,096 token run completed 722 answers and then stopped when the next answer reached the
configured ceiling. OpenRouter counts reasoning tokens inside the completion ceiling. With medium
effort allocating approximately half of the ceiling to reasoning, difficult questions can leave
too little space for the final answer.

## Change

The resumed run changes only the answer output ceiling from 4,096 to 8,192 tokens. The requested
reasoning effort remains `medium`. Retrieval, reranking, evidence selection, model, prompt, and
official judge configuration remain unchanged.

The completed Voyage index is reused with `--reuse-index`. The 722 valid existing predictions are
preserved. The resumed run starts at the first question without an answer checkpoint.

## Prediction

I predict that the 8,192 token ceiling will avoid another output ceiling failure on the remaining
questions and that the resumed run will produce a complete prediction file for all 1,013 questions.
I predict that the answer score will not decrease solely because of this ceiling change.

## Result

No resumed answer generation measurement has been run at the time this addendum was committed.
