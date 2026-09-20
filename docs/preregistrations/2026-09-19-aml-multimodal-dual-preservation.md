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

### Implementation compatibility result, 2026-09-20

The local compatibility implementation passed all fifteen focused tests in
`tests/test_aml_multimodal.py`:

```text
15 passed in 10.59s
```

The checks cover the ordered Add and Search wire shape, the expanded bounded HTTP body, accepted
and rejected media, decoded per-image and aggregate limits, SHA256 deduplication, absence of Base64
from primary text chunks, exact ordered reconstruction, Voyage provider request shape, both MM2
embedding calls, tenant isolation, the response-media budget, startup readiness, and deletion of
primary, media, and vector tenants. MM0 keeps only supplied text and no image bytes. Both
preservation variants also retain and return a text-only memory in its original scalar shape.

The two wire tests were observed red against committed pre-fix `b95d34bf12dc4220267513d42ca01ecf8d33b427`
with HTTP 422 at their intended HTTP 200 assertions. Eleven further mutation proofs were observed at
the exact assertions recorded in the test docstrings: restoring the 2,000,000-byte body cap,
bypassing signature validation, removing WebP, bypassing decoded-size checks, breaking media dedup,
reversing manifest order, bypassing document multimodal embedding, reversing the response-media
comparison, bypassing the multimodal startup probe, and changing Voyage document mode to query
mode. The MM0 mutation that copied image Data URIs into text failed the no-`data:image` corpus
assertion.

Both parameterized text-only preservation cases were also observed red against the pre-repair
implementation: retrieval found the text chunk, but reconstruction found no manifest and the exact
`response.data[0]` assertion raised `IndexError`. Routing all MM1 and MM2 Add records through the
same manifest path made both cases green without changing MM0 or any coding variant.

This is a compatibility result only. No quality dataset has been run, no Voyage quality gain has
been measured, and no AML hosted evaluation has been launched. The frozen quality gate therefore
remains closed.

### Live Voyage compatibility probe, 2026-09-20

A bounded live probe sent one ordered text-plus-image document and the same multimodal query to
`voyage-multimodal-3.5` with provider retries disabled. The updated VPS2 credential authenticated,
and both calls returned exactly one 1,024-dimensional vector:

```text
documents=1 document_dim=1024 query_dim=1024 model=voyage-multimodal-3.5
```

Two non-scientific setup failures preceded that result. The local repository `.env` still carried
an invalid Voyage key and failed authentication, so it was not used. The first VPS2 request used a
hardcoded one-pixel PNG fixture that Voyage rejected as corrupt. Regenerating the one-pixel PNG
through Pillow, while keeping the same inline Base64 request shape and model parameters, produced
the successful result above. No credential, image payload, or embedding value was logged or
committed, and the temporary local and VPS2 probe files were removed.

This proves provider acceptance and dimensional compatibility only. It does not measure retrieval
quality, latency distribution, cost, or final answer score, so it does not open the quality gate.
