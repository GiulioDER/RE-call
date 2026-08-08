"""The producer for `recall.promotion`'s gate.

`recall/promotion.py` implements `evaluate_retrieval_promotion` completely and, until this
package, nothing built a `RetrievalGateInput` outside its own test — so no promotion decision
could exist. This package is that producer, in five parts:

``manifest``   frozen question ids and input hashes, fixed before any candidate result exists
``records``    the closed per-question record schema every corpus writes into
``adapters``   LOCOMO, PEPs, the Answerability Ladder, LongMemEval and MT-RAG, reduced to
               frozen questions
``run``        corpus-agnostic scoring of one arm into a resumable ledger
``aggregate``  outcomes, safety metrics, the gate input, and the decision artifact

Resume comes from `recall.eval.resume`, which is the single factored-out mechanism rather than a
fourth one; its docstring records why `recall/eval/gap_run.py` was deliberately left on its own.
"""
