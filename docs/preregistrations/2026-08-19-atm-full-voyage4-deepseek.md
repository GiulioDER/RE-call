# ATM full run, Voyage 4 large and DeepSeek V4 Pro

Status: preregistered before the first full answer generation measurement.

## Configuration

This run uses the full ATM Bench split with 1,013 questions and the text descriptions already
used by the retrieval measurement.

The RE call pipeline is fixed as follows:

1. Embedder: `voyage:voyage-4-large`.
2. Retrieval: dense plus lexical hybrid fusion.
3. Reranker: `voyage:rerank-2.5`.
4. Voyage candidate pool: `candidate_k=25`. This is the largest tested pool that completed under
   the observed Voyage project limit of 2,000,000 rerank tokens per minute.
5. Answer context: the top 10 reranked items, bounded to 8,192 characters, approximately 2,048
   input tokens.
6. Answerer: OpenRouter model `deepseek/deepseek-v4-pro`.
7. Reasoning: requested effort `medium`. DeepSeek documents this request as the high reasoning
   tier for V4 Pro.
8. Answer output ceiling: 1,024 tokens. A completion that reaches the ceiling is an invalid run
   result and must be rerun with a larger ceiling before scoring.
9. Checkpointing: retrieval and answer JSONL files are written after every completed question.
10. Official judge: `gpt-5-mini`, using the official ATM prompt and a 600 token output ceiling.

The answer prompt requires the shortest complete answer, preserves exact facts and list members,
uses only retrieved evidence, and refuses unsupported claims. The runner does not expose the
ground truth answer to the retriever or answerer.

## Predictions

Before the full answer run, I predict the following directional result against the already measured
MiniLM plus lexical plus Voyage rerank retrieval reference:

1. Voyage 4 large will improve question level retrieval at 10 or remain within 0.03 absolute of
   the reference.
2. Voyage 4 large will improve complete evidence recall on the full split or remain within 0.03
   absolute of the reference.
3. The answer score will be lower than retrieval question recall because some questions require
   multiple evidence items and exact completeness.
4. The hard split will remain materially lower than the full split.
5. The DeepSeek answer stage will complete all 1,013 questions without a missing prediction when
   resumed from checkpoints after transient provider failures.

## Result

No full answer generation measurement has been run at the time this record was committed.
