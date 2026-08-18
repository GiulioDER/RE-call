# Pre registration: does the promote/serve bind actually reproduce?

**Date:** 2026-08-18   **Status:** measured 2026-08-18. **Every registered prediction
confirmed.** The bind reproduces end to end on master at `bda88122`: a generation can be
promoted only in the mode that cannot serve it. Predictions and falsifiers below are unedited;
the result is appended below the horizontal rule.

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

---

## Result

Measured 2026-08-18 on master at `bda88122`, end to end through the shipped CLI, on a throwaway
session database at 384 dim with `fastembed` bge-small over `recall/eval/corpus` (22 files).

**The claim reproduces exactly, including the mechanism, and every registered prediction holds.**

### ⚠️ Two citations above the rule moved before this was even opened as a PR

The registered sections are left byte identical, because a prediction is a historical record. #390
landed between writing and pushing and moved two of the three:

| Cited above | Was | Is now |
|---|---|---|
| `recall/cli.py:2082` | the `generation_mode = ...` store selection | `recall/cli.py:2199` |
| `recall/generations.py:796` | the `UnsafePromotion` message | `recall/generations.py:849` |

`recall/readiness.py:110` is unmoved. No claim changed; only line numbers did.

🔑 **This is the sixth such drift in one working session**, and the interval this time was under an
hour, from writing a prediction to opening its pull request. It is the reason the design states
requirements as behaviour rather than as the presence of a field, and the reason a CI check that
resolves every `path:line` against its anchor is filed as follow up work.

### The two halves of the bind

| Step | `RECALL_ENV=development` | `RECALL_ENV=production` |
|---|---|---|
| build / validate / promote | **succeeds** (22 objects, 22 chunks, promoted) | ⛔ `UnsafePromotion: generation promotion is unavailable in production until certification gates land` |
| strict search, same corpus | ⛔ **`INDEX_NOT_READY`** | ⛔ **`CALIBRATION_MISSING`** |

The generation was genuinely live before either probe: `active_generation_id =
gen_247764a2a6184239969821d8b15caa61`, `state = active`, 22 rows in `recall_chunks_v1`. So the
development refusal is not a null result dressed up as the bind.

**Only `RECALL_ENV` changed between the two probes.** Same database, same tenant, same promoted
generation, same embedder, same query, same command.

### Apparatus check

Registered as required before believing either probe, because two probes taking the *same* store
class would make identical codes the expected result rather than a falsification:

```
RECALL_ENV=development  -> PgVectorStore
RECALL_ENV=production   -> GenerationStore
```

Different classes, so the experiment could discriminate. It also did discriminate, since the two
codes differ.

### Verdict against the prediction

| Registered prediction | Measured | Verdict |
|---|---|---|
| `development` refuses `INDEX_NOT_READY` | `INDEX_NOT_READY` | **confirmed** |
| `production` refuses `CALIBRATION_MISSING` | `CALIBRATION_MISSING` | **confirmed** |
| only `RECALL_ENV` need change between probes | nothing else changed | **confirmed** |
| the bind still holds on today's master | holds at `bda88122` | **confirmed** |
| mechanism: no `generation_binding` on `PgVectorStore` forces the override | store classes differ as predicted | **confirmed** |

### What this settles, and what it does not

⛔ **It settles the premise the merged design was built on**, which until now had been confirmed
only at the level of the code sites that explain it. A generation can be promoted **only** in the
mode that cannot serve it, and served **only** in the mode that cannot promote it. Both refusals
were observed in one run.

⚠️ **It does not reproduce the original report's exact surface.** That report was against the MCP
server; this is the CLI. Both select the store from the same `RECALL_ENV` test and both call
`trusted_search`, so this is strong evidence for the mechanism and slightly weaker evidence for the
server specifically. The registered confound stands as written.

One corpus, one embedder, one machine. Nothing here measures whether the codes would differ on a
corpus with a calibration present, which is a different experiment.
