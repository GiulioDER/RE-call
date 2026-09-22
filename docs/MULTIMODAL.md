# Isolated multimodal retrieval

The multimodal specialist is a separate physical retrieval tenant, not a second vector attached to
the text tenants. Its fixed identity is:

| Boundary | Value |
|---|---|
| Tenant | `re-call-multimodal` |
| Embedding profile | `voyage-multimodal-3.5-v1` |
| Provider model | `voyage-multimodal-3.5` |
| Declared dimension | 1,024 |
| Default state | Disabled |

The existing control plane remains the authority for the physical table, generation, profile, and
calibration binding. A multimodal generation must be built, validated, calibrated, and promoted
under this tenant exactly as any other generation. Equal dimensions never make the text tenants
interchangeable, and no raw cosine score or vector is compared across tenants.

## Stored contract

`recall.multimodal.build_media_ref` validates an image admission and returns a
`MediaObjectRef`. The vector store receives only bounded metadata: SHA256 content digest, media
type, byte size, controlled `s3://` or `file://` object reference, source linkage, authoritative
timestamp, access policy, OCR, caption, entities, location, and region or page references. Binary media,
base64 data URLs, and unbounded payloads are not stored in vector metadata or vector rows.

`MultimodalSidecar` associates that reference with bounded text. The sidecar can carry OCR and
operator supplied captions, but it does not replace the original object. The original object is
returned only as a controlled reference after tenant and principal authorization and within the
response byte budget. Missing authorization, an unknown object scheme, an overlarge upload, or an
overlarge response refuses the operation.

Provider image bytes are transient request material. `MultimodalQuery` builds the provider input
without adding the input bytes to persisted metadata. Original media must be retained in the
deployment owned object store according to its retention and erasure policy, with the digest kept
as the linkage key.

## Activation

The feature is off unless all of the following are explicit:

```text
RECALL_MULTIMODAL_ENABLED=1
RECALL_MULTIMODAL_OBJECT_ROOT=s3://approved-bucket/multimodal/
RECALL_MULTIMODAL_TENANT=re-call-multimodal
RECALL_MULTIMODAL_EMBED_PROFILE=voyage-multimodal-3.5-v1
VOYAGE_API_KEY=...
```

The fixed tenant and profile cannot be renamed through environment configuration. Admission,
response, and item budgets are set with `RECALL_MULTIMODAL_MAX_MEDIA_BYTES`,
`RECALL_MULTIMODAL_MAX_RESPONSE_BYTES`, and `RECALL_MULTIMODAL_MAX_ITEMS`. The session MCP file
does not include this specialist unless `RECALL_MCP_INCLUDE_MULTIMODAL=1` is set.

## Evidence boundary

The MM2 result is operational wiring evidence only. It supports the specialist's isolation,
provenance, bounded media handling, and opt in behavior. It does not establish official retrieval
quality, recall, or exact image preservation. Any quality evaluation requires a new preregistration,
its own generation and calibration, and explicit approval.
