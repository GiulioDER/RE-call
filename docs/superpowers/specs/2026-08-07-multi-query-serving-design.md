# Serving multi-query fusion in `HybridRetriever`: design

Status: **APPROVED DESIGN**, 2026-08-07. Implementation plan follows separately.

Turns a measured benchmark result into a serving capability. The result:
`/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/`.

## What the measurement actually says

Fusing the current turn with a concatenation of prior turns (`mq_nested2_nogold`) beats the
single-query control on **both** axes, under two cross-encoders 25x apart in size:

| arm | reranked nDCG@5 (MiniLM) | reranked nDCG@5 (BGE) | R@100 |
|---|---|---|---|
| `mq_last` (control) | 0.3775 | 0.3930 | 0.7377 |
| **`mq_nested2_nogold`** | **0.3858** | **0.4047** | **0.8220** |

Paired: **+0.0084** nDCG@5 (MiniLM, p=0.0002, Holm-significant), **+0.0117** (BGE), and
**+0.0842** R@100. No gold, no LLM, no API, no extra model.

⚠️ **The gain is CONDITIONAL on reranking.** Raw, this arm is **worse**: −0.0447 nDCG@5, which
tripped three preregistered ranking vetoes. The cross-encoder repairs the damage the concatenated
query does to the ranking. That conditionality is the single most important fact in this document
and the design enforces it rather than documenting it.

⚠️ **Measured at `candidate_k=100` on MTRAG-human dev.** RE-call's default is 20. Other settings
are untested, not merely different.

## Architecture

`HybridRetriever` gains ONE public method. `search()` is untouched.

```python
def search_fused(
    self, query: str, history: Sequence[str], k: int = 5, source: str | None = None
) -> RetrievalResult
```

Both methods delegate to a new private `_retrieve_legs(query, source)` returning per-leg ID
rankings plus hits. That seam is the only structural change to existing code, and it is what stops
the two paths drifting:

- `search()` = `_retrieve_legs` → inner RRF → rerank whole pool → truncate. **Unchanged behaviour.**
- `search_fused()` = `_retrieve_legs` x2 → inner RRF each → **outer RRF** → **cap at 100** →
  rerank → truncate → re-score.

**Fusion is opt-in by DATA, not by flag.** No `history`, no fusion. Every one of the dozen-plus
existing `search()` call sites is unaffected, and there is no new default to argue about.

**`recall_mcp` is out of scope.** Exposing fusion over MCP means adding a history parameter to a
public tool surface with its own auth, limits and query-length contract. That deserves its own
spec. This design makes the capability available to the library.

## The fusion, and every constant's provenance

1. **Inner fusion**, per variant: RRF over `[dense, splade]` at k=60. Bit-identical to `search()`.
2. **Outer fusion**: RRF over the two per-variant rankings, k=60, **equal weights**, nested.
   Contrast T1 found nested and flat indistinguishable on R@100 with flat nominally ahead, but the
   arm that was measured and won is the nested one. Shipping the nominally-better arm from a
   non-significant contrast is reading noise.
3. **`FUSED_RERANK_POOL_CAP = 100`**, applied BEFORE reranking.
4. Rerank once, truncate to `k`, re-score (below).

🔑 **Step 3 is not a tidy-up, it is a measured requirement.** The whole-pool secondary found that
reranking a 547-candidate pool **lost 0.0513 R@100** where reranking 200 **gained 0.0226**. The
cross-encoder degrades as the pool widens, which
`closed-hypothesis-recall-rerank-pool-interaction-2026-08-05` recorded as "not more to select
from, more rope". Multi-query fusion roughly doubles the pool. Without the cap it hands the
reranker twice the rope and the measured gain reverses.

## The score contract, and the store method it requires

`search()` guarantees: **each hit's score is its true dense cosine against the query.** That is
load bearing, not cosmetic. `recall/trust.py:292` thresholds on it and `trust.py:536` feeds it to
`cal.confidence(hit.score)`, a CALIBRATED confidence. A cosine measured against the concatenated
history, pushed through a calibration fitted on cosines against the query, yields a number that
silently means something else.

Three of the four legs already comply: both sparse legs accept `vec=qvec_last` for reporting, and
`last`'s dense leg is against the query by construction. **Only `full`'s dense leg is on the wrong
basis** — `query_dense` ranks by and reports against the same vector, with no reporting override.

**Resolution: add `PgVectorStore.cosines_for(ids, vec) -> dict[str, float]`** and re-score the
final `hits[:k]` against `qvec_last` before returning. That is <= `k` rows (default 5) on a
primary-key lookup: one small round trip, and it makes the contract IDENTICAL rather than similar.

Rejected alternative: drop `full`'s dense leg, fusing `last`(dense+splade) + `full`(splade). It
needs no store change and it is **not the arm that was measured**. Trading a measured
configuration for an unmeasured one to avoid an engineering cost is the substitution this whole
line of work exists to prevent.

## History handling

The concatenation rule lives in a module-level function, testable in isolation: prior turns
newline-joined in order, `strip_speaker` applied per line, reproducing the benchmark's `full`
variant exactly. RE-call owns the rule so that two installations cannot send different
concatenations and both call it the measured configuration.

**`FUSED_HISTORY_MAX_CHARS = 4096`**, matching `recall_mcp.service.MAX_QUERY_CHARS` so the library
and the server agree on what an over-long query is. Measured against the CONCATENATION, which is
built inside RE-call and therefore never passes MCP's own check on the incoming query.

**Over-budget histories are REFUSED, never truncated.** A truncated history is a configuration the
benchmark never tested, served under the measured configuration's name. Refusing is the same
principle as `resolve_reranker` refusing an unparseable flag rather than reading it as "off": an
operator who asked for the measured behaviour and quietly got something else has no way to notice.

⚠️ **The encoder truncates at 512 tokens regardless of this budget**, so a history beyond roughly
2,000 characters is already being silently clipped by the tokenizer before it reaches the index.
That clipping was PRESENT IN THE MEASUREMENT (MTRAG's later turns concatenate well past the
window), so it is part of the measured system rather than a defect this design introduces. The
4096 budget therefore bounds the request, not the encoded query, and the two limits are different
things. An operator reading `FUSED_HISTORY_MAX_CHARS` must not conclude that 4,000 characters of
history all reach the retriever.

## Errors

All `ValueError`, all naming the actual and expected state:

| condition | why it raises rather than degrades |
|---|---|
| no reranker configured | the gain is conditional on reranking; without one this arm is measurably WORSE than `search()` |
| `history` empty | callers wanting single-query behaviour should call `search()`, not get it silently |
| concatenation over budget | see above; refuse, never truncate |

## Diagnostics

- `candidate_pool_size` reports the **realised** fused pool before the cap, not the configured
  `candidate_k`. The benchmark had to learn this distinction the hard way: `pool_bound()`
  overstated the realised pool by 3x on 307 of 777 queries, and no score in the output revealed it.
- `stage_ms` gains `history_retrieval` and `outer_fusion`, so the doubled retrieval cost is
  visible rather than buried in one number.
- `gap_warning` is computed from **`last`'s dense candidate scores only**. It answers "does the
  corpus hold an answer to what the user asked", and `last` IS what the user asked. Computing it
  over both variants would let a strong match on stale earlier context suppress a gap warning on
  the current question: the honesty guard failing in the dangerous direction.
- `RetrievalResult.query` carries the original query, not the concatenation.

`candidate_k` stays a free operator knob. Forcing 100 would be choosing for the operator; the
docstring states plainly that the measured figures come from `candidate_k=100` with a reranker.

## Testing

Every test written to survive mutation: removing the behaviour must turn the suite red. That
standard is not theoretical. Twice in the work behind this design I wrote guards that could not
fire, and both times only a mutation check caught it.

| test | what it protects |
|---|---|
| **benchmark parity** | committed fixture of leg rankings, expected order from `multiquery.fuse_arm(mq_nested2_nogold)`. Asserts serving reproduces the benchmark exactly. **The "served system IS the measured system" gate.** |
| three refusals | each asserts the message names the actual state |
| score basis | a hit surfaced ONLY by history must return the cosine against the QUERY. Fails without `cosines_for`; pins the fix, not the symptom |
| gap-warning direction | history matches strongly, query does not, warning must still fire |
| degenerate invariant | `search_fused(q, [q])` orders identically to `search(q)` **when the fused pool is at or under the cap**; nested RRF over two copies is order-preserving. ⚠️ At `candidate_k=100` the pool reaches 200 and the cap DOES bind, so the two legitimately differ there; the test pins the sub-cap case and asserts the divergence above it rather than pretending the invariant is unconditional |
| pool cap binds | a fused pool over 100 hands the reranker exactly 100 |
| `search()` unchanged | existing tests pass untouched, plus one asserting identical output across the `_retrieve_legs` refactor |
| `cosines_for` | store test against real Postgres, alongside `test_store_*` |

## Scope

**In:** `search_fused`, `_retrieve_legs`, the concatenation function, `FUSED_RERANK_POOL_CAP`,
`cosines_for`, diagnostics, tests.

**Out:** MCP exposure (own spec); any change to `search()`'s behaviour; LLM reformulations
(withdrawn on evidence: contrast C3 found the gold rewrite an LLM would approximate contributes
nothing once reranked); flat topology; tuning `candidate_k`.

## Known limits

- Measured on **MTRAG-human dev only**, at `candidate_k=100`, with a reranker. Never on MTRAG-UN.
- The effect is small in absolute terms (+0.0084 nDCG@5) though robust across two rerankers, and
  it is bought with roughly **2x the retrieval cost** plus mandatory reranking (~1,050 ms/query on
  CPU per `recall_mcp/service.py`). Whether that trade is worth it is an operator decision, which
  is why this is opt-in by data and refuses rather than defaults.
- The benchmark's `history` was MTRAG's own conversation concatenation. Real histories differ in
  length and shape, and no bound beyond the refusal was tested.
