# Experimental Reasoning Release Notes

Status: experimental and opt in.

Reasoning remains an optional layer over trusted retrieval. It is provider neutral by default,
citation constrained, and not a general reasoning claim. A response may assemble evidence, trace
proposal guided exploration, or request review, but it does not certify unsupported model
inference as corpus truth.

## Migration Notes

CLI:

* Reasoning stays behind explicit `recall reasoning` commands.
* Existing `recall search`, `recall evidence`, indexing, calibration, and generation commands keep
  their retrieval behavior.
* New diagnostics may include `provider_metadata`, with provider id, model id, nullable model
  revision, token counts, latency, and nullable monetary cost.

MCP tools:

* Reasoning tools remain opt in and additive.
* MCP search trust semantics do not change.
* Proposal output is review data. It is not trusted metadata unless a reviewed promotion record
  exists.

Serialized fields:

* `ReasoningDiagnostics.provider_metadata` is an array of provider metadata records.
* Benchmark artifacts include top level `provider_metadata` and `cost_claims`.
* Monetary benchmark claims are rejected when provider metadata omits a model revision or cost.
* Raw inference proposals and promoted facts are separate serialized concepts.

## Limitations

Reasoning is citation constrained. Answers must cite trusted evidence bundle chunk ids. Proposal
ids, inferred relationship ids, and untrusted hits are not valid answer citations.

Provider metadata is best effort where providers expose it. Token counts and latency may be
available while model revision or monetary cost is null. Null cost is not a cost claim.

Promotion requires reviewer identity, source generation identity, proposal evidence ids,
timestamp, and an audit note. Only promoted facts produced from reviewed and accepted proposals are
trusted metadata.

## Evaluation Posture

Do not broaden beta from these fields alone. Before broader beta, run heldout evaluation with
nearest neighbor, shuffled edge, removed edge, and unsupported inference controls. Report citation
precision, unsupported claim rate, correct abstention, false abstention, proposal precision and
recall, provider failure rates, token counts, latency, and monetary cost only when the provider
metadata contract can audit the claim.
