# Atomic fact auxiliary retrieval view construction audit

Status: predicted, not yet measured.

Registered 2026-09-16 before implementing the audit harness or calculating any aggregate from the
frozen pool.

## Question

Can the existing memory corpus support a bounded auxiliary retrieval view that embeds one complete
fact at a time, while preserving the original chunk as the only evidence returned to callers?

This is not the existing section contextualization feature. Section contextualization embeds the
complete ordinary chunk with document title, heading hierarchy, and source path. The candidate here
adds a second retrieval record for each eligible paragraph. Its embedding text contains the
document title, nearest heading hierarchy, optional explicit fact field, and the complete paragraph.
The record points to its ordinary parent chunk. Retrieval must deduplicate on that parent and return
the unchanged parent text, identity, provenance, trust metadata, and citation.

No model, embedding, retrieval, or external inference runs in this phase.

## Frozen inputs

The input pool is
`docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json`, SHA256
`66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`. Only its 250 answerable
rows are used. The controls are excluded.

The source roots are frozen as:

1. `recall` at
   `C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory`
2. `sentiment-agent` at
   `C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory`

Every source must still match the SHA256 frozen in the pool. Ordinary parent chunks are reconstructed
with the default `recall.index.chunk_text` policy.

## Frozen construction

Strip only a valid leading YAML frontmatter block. Parse the remaining Markdown in source order.
Maintain the nearest Markdown heading hierarchy from levels one through six. A heading itself is not
a fact.

Each nonheading paragraph is eligible when its whitespace collapsed form:

1. is between 40 and 320 characters inclusive
2. contains at least eight words under `\b[\w'-]+\b`
3. contains no HTTP URL, fenced code marker, or Markdown table pipe
4. occurs exactly once in the whitespace collapsed source
5. maps wholly into exactly one ordinary parent chunk

An explicit field paragraph begins with an optional Markdown list marker and optional bold marker,
then one of `FACT`, `VERDICT`, `RESULT`, `OUTCOME`, `DECISION`, `APPLY`, `WHY`, `STATUS`,
`OBJECTIVE`, or `ROOT CAUSE`, followed by a colon. For such a paragraph, the field label is stored
separately and the complete value after the colon is the fact content. Other eligible paragraphs
use the complete paragraph as fact content.

The auxiliary embedding text is rendered in this exact order, omitting empty fields:

```text
title: <document title>
section: <heading one > heading two>
field: <explicit field label>
content: <complete fact content>
```

Document title uses the production precedence in `recall.context.document_title`. Headings and
fields are normalized only by control character removal, trimming, and whitespace collapse. No
query text, gold label, answer span, source path, filename, synthetic identifier, model output, or
retrieval result may enter the view.

Each view records its source and parent chunk ordinal only for the audit. A future implementation
must store an opaque parent chunk identity and deduplicate candidates on it before trust evaluation
and evidence assembly.

## Measurements

Report:

1. source integrity failures and ordinary chunk reconstruction failures
2. total sources, ordinary chunks, auxiliary views, and sources with zero views
3. auxiliary views per source and per parent chunk at minimum, p50, p90, p95, and maximum
4. auxiliary row growth, defined as auxiliary views divided by ordinary chunks
5. rendered character growth, defined as all auxiliary embedding text characters divided by all
   ordinary chunk characters
6. title, heading, and explicit field coverage
7. rendered view length and fact content length distributions
8. exact normalized fact duplicates within a source and across sources
9. exact normalized rendered view collisions within a source and across sources
10. the count of views whose rendered text exceeds 800 characters
11. for every frozen answer span, the count of auxiliary views containing it, whether the unique
    matching view maps to the frozen parent ordinal, and whether its content is complete
12. gold coverage by the existing `extractive_field`, `extractive_heading`, and
    `extractive_fallback` construction families

All distributions use nearest rank quantiles. Normalization is Unicode NFKC, case folding, then
collapsing all whitespace to one ASCII space.

## Frozen gates

Return `GO_BUILD_ATOMIC_FACT_AUXILIARY_SHADOW` only if all of the following hold:

1. all 250 sources exist and match their frozen SHA256
2. all 250 frozen answer spans occur in exactly one auxiliary view
3. every unique gold view maps to the frozen ordinary parent ordinal
4. every gold construction family has 100 percent coverage
5. every source has at least one auxiliary view
6. zero rendered views exceed 800 characters
7. exact normalized fact duplicates within a source are at most 1 percent of views
8. exact normalized rendered view collisions across sources are at most 1 percent of views
9. auxiliary row growth is at most 2.0 times the ordinary chunk count
10. rendered character growth is at most 1.5 times ordinary chunk characters
11. the p95 auxiliary views per parent chunk is at most 4 and the maximum is at most 8

Otherwise return `STOP_ATOMIC_FACT_AUXILIARY_SHADOW` and name every failed gate. Do not tune the
eligibility bounds, rendering, or gates on these consumed sources.

Passing this audit authorizes only a shadow implementation and a new Context 4 generation. The
retrieval comparison must use a fresh source disjoint exact span set and must report both parent
deduplicated gold source and exact span reach. It does not authorize serving.

## Prediction

I predict the audit will preserve all 250 gold spans and stay within the row and character growth
bounds. The most likely failure is parent crowding in dense memos. If that gate fails, the fact view
is not safe to build as a flat auxiliary index because near duplicate facts could occupy too much of
the candidate budget.
