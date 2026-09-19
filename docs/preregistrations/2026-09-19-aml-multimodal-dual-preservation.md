# AML multimodal dual preservation experiment

Date: 2026-09-19

## Purpose

This experiment tests whether preserving original image evidence improves the AML Multimodal
memory path, and whether an independent Voyage multimodal retrieval leg improves evidence recall
beyond caption-based text retrieval. It does not use private AML evaluation data and it does not
launch an official Full evaluation.

## Frozen contract

The participant API receives source-ordered messages whose content is either text or an ordered
array of text and inline Base64 image parts. Multimodal Search queries use the same content shape.
Returned memories may also contain ordered text and image parts. JPEG, PNG, and WebP are accepted.
The implementation will enforce 10 MiB decoded per image, 30 MiB decoded per Add request, and
30 MiB decoded across one Search response.

The source image remains the authoritative evidence. Captions, OCR, and embeddings are retrieval
aids and must not replace the original image or lose the relationship between surrounding text and
the image part.

## Frozen arms

1. `MM0_caption`: accept ordered multimodal input, retrieve using only supplied text, and return
   text evidence. Image bytes are deliberately not retained by this arm.
2. `MM1_preserve`: use the same text retrieval as `MM0_caption`, but durably retain the original
   images and return the source-ordered text and image evidence.
3. `MM2_dual`: retain and return the same evidence as `MM1_preserve`, add a dedicated
   `voyage-multimodal-3.5` candidate leg, and fuse text and multimodal rankings with deterministic
   reciprocal-rank fusion.

All arms preserve the same `user_id` isolation boundary. Search returns memory evidence only and
never a generated final answer. No arm may use question labels, gold answers, or gold evidence IDs
during Add or Search.

## Frozen predictions

1. `MM1_preserve` will improve answer quality over `MM0_caption` on questions whose required fact
   is visible in the source image but absent or incomplete in supplied text.
2. `MM2_dual` will improve evidence Recall at 10 over `MM1_preserve` on image-only and mixed
   text-image queries.
3. `MM2_dual` will not reduce text-only evidence Recall at 100 relative to `MM1_preserve`.
4. The dual arm will cost more Add and Search latency, but Search p95 will remain below the AML
   request timeout and every response will remain within the published decoded-media limit.

## Frozen measurements

The compatibility stage records:

1. Ordered text-image-text round-trip equality.
2. Accepted and rejected media types, decoded image sizes, and aggregate request sizes.
3. Idempotent replay and conflicting replay behavior.
4. Immediate searchability, tenant isolation, and deletion of text, visual, and media sidecars.
5. Search response decoded-media size and stable rank order.
6. Add and Search latency and Voyage usage metadata.

The quality stage records, overall and separately for text-only, image-only, and mixed queries:

1. Evidence Recall at 10 and 100.
2. Complete evidence recall at 10 and 100.
3. Mean reciprocal rank.
4. Final answer score under the same fixed Answer configuration.
5. Add latency, Search latency, persisted media bytes, and response media bytes.

## Frozen gates

The quality stage is authorized only after every compatibility invariant passes. `MM1_preserve`
is retained only if it has no contract, isolation, deletion, or response-budget failure.
`MM2_dual` is retained only if it improves image-only or mixed-query Recall at 10, does not reduce
text-only Recall at 100, and produces a positive final-answer result rather than only a retrieval
proxy gain.

If a test apparatus defect is found, preserve the failed artifact, append the defect and repair
below the frozen marker, prove the repair red then green, and use a fresh immutable result path.
Do not edit the registered text.

## Frozen implementation boundary

Original images are content-addressed by SHA256 and stored once per tenant. Message manifests keep
source order and refer to those media identifiers. Text retrieval chunks never embed Base64 image
data. The multimodal vector sidecar is stored separately from the existing text-vector space so
incompatible embedding spaces cannot be mixed accidentally.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

Not measured yet.
