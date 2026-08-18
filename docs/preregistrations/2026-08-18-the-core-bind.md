# Pre registration: does the promote/serve bind actually reproduce?

**Date:** 2026-08-18   **Status:** predicted, not yet measured

## The question

`docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` was written on top of a reported observation that nobody in
that work reproduced:

> building, validating and promoting a generation under `RECALL_ENV=development` left the MCP server
> reporting `INDEX_NOT_READY` forever; re-probing the same corpus with `RECALL_ENV=production` moved
> it to `CALIBRATION_MISSING`, i.e. the generation was suddenly visible.

Every code site that *explains* that behaviour was confirmed. The behaviour itself was taken on
trust and used as the premise of a merged design. This tests it.

**Q1.** With one promoted generation over one corpus, does a strict search refuse with
`INDEX_NOT_READY` under `RECALL_ENV=development` and `CALIBRATION_MISSING` under
`RECALL_ENV=production`, with nothing else changed?

**Q2.** Does it still hold on today's master? The reported observation predates #370, #381 and #383,
all of which touched the identity and fingerprint machinery in between.

## What I predict

**The claim reproduces exactly, and I can name the mechanism rather than only the outcome.**

- Under **development**, `recall/cli.py:2082` selects a plain `PgVectorStore`. That class has no
  `generation_binding` method, so in `trusted_search` the binding stays `None`, the gate finds no
  `generation_id`, and it overrides the failure code to `INDEX_NOT_READY` regardless of calibration
  state.
- Under **production**, a `GenerationStore` is selected, `generation_binding()` returns the promoted
  generation, the override does not fire, and `code_for_status("missing")` yields
  `CALIBRATION_MISSING`.

So: `development -> INDEX_NOT_READY`, `production -> CALIBRATION_MISSING`, both as strict refusals,
on the same database, same tenant, same promoted generation, changing only the environment variable.

**Q2: it still holds.** None of #370, #381 or #383 touched store selection or the gate's
override, so the bind should be untouched by them.

## What would falsify this

- Either environment producing a code other than the one predicted.
- Both environments producing the **same** code, which would mean the bind does not exist and the
  design's central framing is wrong.
- `development` reaching a *non refusal*, which would mean the corpus was servable all along.
- Any need to change something other than `RECALL_ENV` between the two probes. The claim is that the
  environment variable alone flips it; if a second change is required, the claim as written is false
  even if the codes match.

⚠️ **I am predicting the outcome I already argued for in a merged design, which is the worst
position to measure from.** The falsifiers above are therefore stated as codes, not as "the bind
exists". If both probes return the same code, that is a falsification I have to publish against my
own merged work.

## How it will be measured

One session database, schema at 384 dim, `fastembed` bge-small, corpus `recall/eval/corpus`
(22 files, already used elsewhere in this series).

1. `recall manifest inventory` then `recall manifest create` to get a `file://` manifest.
2. `recall generation build --unverified-development`, then `generation validate`, then
   `generation promote --unsafe-development-promotion`. All three necessarily under
   `RECALL_ENV=development`, because production refuses a local manifest and refuses
   `allow_unverified`. That refusal pair is itself part of the reported bind.
3. Probe a strict search twice, changing **only** `RECALL_ENV`, and record the failure code each
   time. `RECALL_TRUST_MODE` is left unset so strict is in force and a refusal carries its code.
4. Confirm the generation really is `active` in `recall_tenant_state` before probing, so a null
   result cannot be mistaken for the bind.

**Apparatus check, before believing either probe:** confirm the two probes actually take different
store classes. If both take the same class the experiment measures nothing, and identical codes
would be the expected result rather than a falsification.

## What I already know

- `promote()` refuses outright under production (`recall/generations.py:796`), so step 2 cannot be
  done in the environment that step 3 needs for the second probe. That asymmetry is the bind.
- The gate overrides to `INDEX_NOT_READY` whenever the binding carries no `generation_id`, which is
  what makes the development probe's code independent of calibration state.
- `recall/readiness.py:110` is a **different** entry point and is not on the search path, so it must
  not be used as the probe.

## Confounds I can name now

A single corpus and a single embedder. The probe uses the CLI rather than the MCP server; both
select the store from the same `RECALL_ENV` test, but the original report was against the server, so
this reproduces the mechanism rather than the exact deployment. If the codes match the prediction it
is strong evidence for the mechanism and slightly weaker evidence for the server specifically.
