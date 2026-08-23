# Document expansion and evidence bundles

Status: supported optional answer selection, measured on the real EnterpriseRAG corpus and the
fixed fixture. The default retrieval ordering remains unchanged.

## Purpose

Chunk retrieval is a useful first address, but a question may require two sections from the same
document. The document expansion path handles that case without widening the global candidate pool
or copying a trust verdict from one chunk to another.

## Retrieval path

```text
query
  -> ordinary calibrated retrieval
  -> identify the best source documents
  -> rerun the same query inside those sources
  -> merge unique chunks
  -> run the normal trust evaluation over every chunk
```

Expansion is adaptive by default. It triggers only for relational queries such as causal,
comparative, temporal, or status questions. It is bounded by the number of source documents and
the number of chunks requested per source.

The public entry point is:

```python
from recall.retriever import DocumentExpansionPolicy
from recall.trust import trusted_search

result = trusted_search(
    store,
    embedder,
    "why did the rollout change?",
    document_expansion=DocumentExpansionPolicy(enabled=True),
)
```

The LangChain and LlamaIndex factories accept the same policy, so the trust boundary stays in one
place when RE-call is used inside a framework adapter:

```python
from recall.integrations.langchain import RecallRetriever
from recall.retriever import DocumentExpansionPolicy

retriever = RecallRetriever.from_store(
    store,
    embedder,
    document_expansion=DocumentExpansionPolicy(enabled=True),
)
```

The command line equivalent is:

```text
recall search "why did the rollout change?" --expand-documents --evidence
```

## Bundle selection

`EvidencePolicy(bundle_mode="document")` ranks source documents by their first trusted retrieval
position, then emits selected chunks in document order. Allocation gives each selected document one
chunk before giving a document a second chunk. This preserves cross document coverage while still
allowing distant sections from one strong document into the context.

The default bundle mode remains `retrieval`, so existing callers retain their ordering and token
semantics.

## Structural expansion and answer slots

`StructuralExpansionPolicy` adds ordinal neighbors around retrieved sections and the highest ordinal
section in each selected document. The terminal section is useful for exception and outcome cases,
but this is still bounded retrieval structure, not heading aware parsing.

`AnswerSlot` provides a deterministic lexical requirement for a bundle. Supplying slots selects a
representative trusted chunk for each requirement and abstains when a required slot is absent. It
does not fill unused capacity with ordinary neighbors, which prevents a misleading partial passage
from being reintroduced after the requirements are satisfied.

`EvidencePolicy(selection_mode="beam", answer_slots=...)` evaluates bounded candidate bundles by
slot coverage, trust quality, and document diversity. Beam selection is opt in and costs more CPU
than the representative slot selector.

## Supported serving path

Answer slot selection is promoted as a supported optional serving feature. It is available through
the public `recall.AnswerSlot` and `recall.EvidencePolicy` exports and through the LangChain and
LlamaIndex `evidence` methods:

```python
from recall import AnswerSlot, EvidencePolicy

policy = EvidencePolicy(
    bundle_mode="document",
    answer_slots=(
        AnswerSlot("decision", ("approved", "exception"), min_matches=2),
        AnswerSlot("owner", ("owner",)),
    ),
)
bundle = retriever.evidence("what was decided and who owns it?", policy=policy)
```

When any required slot is absent, the bundle abstains with `answer_slot_gap`. Callers should treat
that as a refusal to generate an answer, not as an empty successful result. The default policy has
no answer slots, so existing callers are unchanged. Beam selection remains opt in because the real
corpus measurement found no slot recall gain over representative slot selection and much higher
latency. A certified, generation bound calibration is still required before relying on trust
decisions in production.

## Safety boundary

Expansion happens before `recall.trust.evaluate`. Every newly retrieved chunk therefore receives its
own cosine, confidence, validity, and supersession verdict. Only `ok` chunks can enter the bundle.
Expansion does not alter tenant, generation, calibration, or corpus binding.

## Measured fixture

The preregistered fixture contains 12 cases. The first measurement increased required evidence set
recall on distant section cases from 0 of 4 to 4 of 4, and overall recall from 7 of 12 to 12 of 12.
The unanswerable false positive count stayed at 1 in both arms. This is a fixture result, not a
claim about a general corpus. A database backed latency measurement remains necessary before
considering a default promotion.
