# C9: the atomic stage built inside Add, verified live

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

C8 (`C8_routed_specialists_grounded_graph`) carries an atomic rescue stage that reads a file
artifact built by hand after ingest (`scripts/build_aml_atomic_rescue_artifact.py`) and bound to
the served corpus fingerprint. The official AML flow calls Add, then Search, with no pause, and
requires each chunk to be persisted and searchable before its HTTP 200 response (live contract,
read 2026-09-23 at https://agentmemoryleaderboard.ai). Every Add changes the fingerprint, and no
one can run the builder in between, so under the official flow C8's stage falls back on every
query. The live C8 reference measured exactly that with no artifact: 187 of 187 attempted,
0 active, 187 fallback.

C9 (`C9_routed_specialists_grounded_graph_atomic`) is C8 with one change: `recall_aml.atomic_views`
builds the `micro` views of each Add request from its own raw windows, embeds them with each scope's
embedder (Code4, and the Context4 specialist) and stores them in an isolated tenant per scope,
before Add returns. Search reads the nearest views by an exact dense query and applies the same
selection and placement. Its defaults are mode `active` and placement `fused` (after fusion, so the
fused top five cannot change; user decision 2026-09-23).

Does C9 activate on every query under the official Add then Search flow with no manual step, does
its selection reproduce the artifact selection, and does the fused placement keep the top five?

## Frozen apparatus

- Host: **VPS3 only.** VPS2 serves the official Multimodal Full run and is not touched in any way.
- Service: an isolated C9 instance built with the C8 reference recipe
  (`~/atomizer-c8/service-setup.sh` on VPS3), changed only in: its own database `atomizer_c9`, its
  own port, variant `C9_routed_specialists_grounded_graph_atomic`, **no** artifact root, **no**
  atomic calibration or pipeline variables, and a **fresh, empty** embedding cache, so every vector
  is computed the way an official run computes it. Served commit: the head of branch
  `claude/aml-atomizer-add` that carries this record, printed by `/version` and recorded in the
  result.
- Corpus: CAMBench coding manifest SHA-256
  `58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814`, 196 sessions, 1,220
  windows, loaded by `load_frozen_corpus` in `scripts/aml_c7_qualification.py`.
- Harness: `scripts/c8_atomizer_live_receipt.py` at `origin/claude/atomizer-c8-study`, the same
  script that produced the C8 live receipt; `ingest` with one worker, then `search` on the dev split
  (153 probes) plus the 34 task prompts, 187 queries in total, the same set the C8 receipt used.
- One ingest, three search passes, the service restarted between passes with only the environment
  changed:

  | pass | environment | purpose |
  |---|---|---|
  | `c9-off` | `RECALL_ATOMIC_RESCUE_MODE=off` | control |
  | `c9-dense` | `RECALL_ATOMIC_RESCUE_PLACEMENT=dense` | parity with the C8 live `on` pass, which used dense placement |
  | `c9-fused` | nothing overridden | the official configuration |

- Parity baseline: the C8 live `on` rows on VPS3 (`~/atomizer-c8/private/`), served commit
  `3953e69d`, artifact `micro`, dense placement.
- Budget: USD 1.00 for this record, inside the program's USD 10 cap.

## Predictions

Written before any C9 service exists. My own record is that I over-predict effect sizes by two to
four times (eleven of twelve earlier predictions too high), so the quality prediction is
deliberately small.

1. **Activation.** In `c9-dense` and `c9-fused`: 187 of 187 queries attempted, active and candidate
   available, 0 fallback, with no artifact file anywhere on the host and no builder run. In
   `c9-off`: 0 attempted. Every Add returns 200 (196 of 196).
2. **View store.** The Code4 view tenant holds **11,749** views, the count the C8 `micro` artifact
   builder produced from the same windows, and the Context4 view tenant holds the same number. The
   raw tenant holds 1,220 windows and no view rows.
3. **Selection parity.** `c9-dense` top-8 ids equal the C8 live `on` top-8 ids on **at least 180 of
   187** queries. Allowed sources of difference: pgvector versus NumPy float32 cosine at ties, the
   Context4 view grouping (per request here, 64-view batches in the builder; 2 of 187 queries
   routed there in the C8 receipt), and graph records recompiled by gpt-4o-mini.
4. **Fused invariant.** `c9-fused` top-5 ids equal `c9-off` top-5 ids on **187 of 187** queries.
5. **Quality direction** (probes, exact gold at rank 8, paired against `c9-off`): gains 2 to 5,
   losses 1 to 4, net 0 to +2. Ranks 1 to 5: no change at all, by item 4.
6. **Latency.** Search p95 in `c9-fused` exceeds `c9-off` by 20 to 150 ms (one extra exact scan
   over about 11,749 rows of 1,024 dimensions).
7. **Spend** under USD 1.00: the gpt-4o-mini anchor compiler about USD 0.15, Voyage about USD 0.4.

## What falsifies it, and what follows

- Any fallback, or fewer than 187 active, in `c9-fused`: C9 does not do what it exists to do. No
  official run with it.
- Any top-5 difference in item 4: the fused placement is wrong. No official run until fixed.
- Parity below 180 of 187: the view-store selection diverges from the measured artifact selection;
  explain every divergent query before any official run.
- Items 5 to 7 inform the decision but cannot block it on their own: item 5 is a small sample, and
  the quality evidence for `micro` is the round-3 confirm and the memory-tenant check.

## Result

Not yet measured.
