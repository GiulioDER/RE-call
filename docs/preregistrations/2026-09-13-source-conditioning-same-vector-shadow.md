# Pre-registration: same vector source conditioning shadow

**Date:** 2026-09-13  
**Status:** predicted, not yet measured  
**Training artifact:** `docs/results/2026-09-13-live-source-scoped-expansion.json`  
**Training artifact SHA256:** `531dfa31c4344df020ebda9be46e2b09365b0dd9c918aff6835f8c5b4e1bd666`  
**Model artifact:** `docs/results/2026-09-13-source-conditioning-model.json`  
**Model artifact SHA256:** `fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f`  
**Validation generation:** `gen_808e6c2592494eebabf144eabb21f838`  
**Validation calibration:** `cal_bc65614ff5ba4471850d53384381d7d5`  
**Pipeline:** `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`  
**Corpus:** `079a63744f2a9ce44aae344a06f57c0f3d959499b674f66c95f3cfc16680e58a`  
**Query set SHA256:** `06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f`  
**Fact labels SHA256:** `45c38731e138f6aed425635b78ee79b692e142f5ef040e0efffeb042ad47186b`

## Motivation

The preceding registered protocol launched independent off and shadow MCP processes. Its final
attempt returned `REPAIR` because one of 50 public lists differed at the admission boundary. The
off process itself returned four items for `memory-001` in one attempt and two in the next. Full
evidence and reproduction are in
`docs/results/2026-09-13-source-conditioning-cross-process-repair.md`.

This protocol measures both arms from one request and therefore one hosted query vector. It does
not change the model, alpha, threshold, gold set, generation, calibration, or safety rules.

## Validation protocol

Launch one generation pinned MCP process with source conditioning in shadow mode at sample rate
one and the private retrieval leg and source admission audits enabled. Run the committed 50 query
set once.

For each request:

1. The baseline is the ordinary public `trusted_evidence.items` list.
2. The candidate is independently recomputed from the private dense, sparse, and full trust pool
   diagnostics. Those diagnostics are produced from the same query vector as the baseline.
3. The emitted candidate chunk hashes and count must match the independent recomputation.
4. The shadow diagnostic must emit hashes for the baseline chunks it received. They must match the
   public baseline identifiers in exact order.
5. The shadow must report `status=ok`, the registered artifact fingerprint, and the exact serving
   lineage. Every request must contain a nonempty internal shadow span.

The shadow remains diagnostic only. No active selector exists, no route changes, and no service is
restarted.

## Predictions

1. Candidate hash parity and public baseline hash linkage will each hold on all 50 requests.
2. No shadow request will error, and all 50 internal span values will be present.
3. The candidate will gain at least one complete query or one essential fact over its same vector
   public baseline without losing on the other metric.
4. Candidate false answers will not exceed baseline, will remain at most two of 28 controls, and
   labeled context precision will be at least 0.55.
5. The 50 requests will finish in under 12 minutes.

Prior cross process attempts already showed a candidate direction of 18 complete queries and 20
facts against 17 and 19 in their separate off process, with one false answer in both arms. Those
are disclosed prior observations, not predictions for the same vector baseline.

## Decision rule

`BUILD SAMPLED SHADOW` only when all five predictions pass. This licenses review of a low rate,
opt in diagnostic deployment. It does not license active selection.

`REPAIR` when hash linkage, artifact identity, lineage, diagnostics, or timing receipts fail.

`GATE` when mechanics and safety pass but no gold gain appears.

`CLOSE` when complete queries or facts decline, false answers increase, or precision falls below
0.55.

## Reproduction command

After the implementation and this preregistration are committed to the detached VPS2 checkout:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-shadow-<commit>'
$env:RECALL_SOURCE_COMMIT='<commit>'
python -u scripts/run_live_source_conditioning_same_vector.py `
  --generation-id gen_808e6c2592494eebabf144eabb21f838 `
  --artifact docs/results/2026-09-13-source-conditioning-model.json `
  --output docs/results/2026-09-13-live-source-conditioning-same-vector.json
```

<!-- frozen_above -->

## Result

Not measured yet.
