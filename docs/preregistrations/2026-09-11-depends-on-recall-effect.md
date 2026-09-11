# Preregistration: authored `depends_on` recall effect

Date: 2026-09-11

Status: locked before measurement

## Objective

Test whether projecting an existing authored `recall_graph.depends_on` declaration into the
semantic graph can recover a gold evidence item that the same retrieval misses without graph
expansion.

## Population and construction

Use a deterministic in memory corpus of paired documents. Each dependent document declares one
`recall_graph.depends_on` target. The target document is the gold evidence for the paired query.
Distractor documents are included so the baseline top five contains the dependent document but
not the prerequisite. The query, chunk IDs, file labels, metadata, and deterministic scores are
fixed before measurement.

The graph is built by `build_semantic_graph`. The treatment goes through the real
`_expand_semantic_graph` serving function with `graph_expansion=one_hop`; the control uses the
same trusted retrieval with `graph_expansion=off`. Both arms have the same five item evidence
budget and one added graph item maximum. No query gold label is supplied to graph construction.

## Primary outcomes

Report paired question level:

* gold evidence hit@5;
* mean reciprocal rank of the first gold item;
* rescue rate among control misses;
* evidence precision at five items;
* graph relation activation, candidate discovery, and candidate admission;
* whether any control query regresses under treatment.

The implementation passes the mechanism gate only if the dependency relation is authored,
points to the declaring chunk, resolves the canonical target path, activates in one hop, and
rescues at least one control miss. This is a mechanism test, not a production quality claim.

## Prediction and interpretation

I predict a positive hit@5 delta on the paired dependent queries and no change on unrelated
controls. A positive result demonstrates that the new typed edge is usable by retrieval. It does
not establish benefit on the production corpus until a rebuilt generation contains exercised
dependency edges and the same paired protocol is run there.

## Invalid run

Invalidate the run if the arms use different chunks, scores, query order, graph generation
identity, context budget, or graph relation source. Do not report a production recall number from
this isolated corpus.
