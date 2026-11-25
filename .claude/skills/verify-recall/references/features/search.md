# Search and the trust gate

Search ranks the store's chunks for a question, and the trust gate decides whether any of them may
be served: certified answer, abstention, or a refusal with a code.

## Sub-features

- `search-strict`: the default; an uncertified store is refused with a code on stderr, exit 1.
- `search-development`: `RECALL_TRUST_MODE=development` retrieves anyway and marks every result
  `[DEGRADED:<code>]`, exit 0.
- `search-certified`: a confident answer or an abstention. Needs a certified calibration.

## How to get to it (user POV)

Run `recall search "<question>"` against the store, optionally with `RECALL_TRUST_MODE=development`.

## Driving it with control_recall

- **Strict refusal.** Run `search --run <run> "how long do pricing snapshot cache entries live"`.
  `exit` is 1, `code` is `CALIBRATION_UNCERTIFIED` on a fresh run (`CALIBRATION_STALE` after a
  refresh changed the corpus), `rows` is empty.
- **Development ranking.** Add `--development`. `exit` is 0, `code` is the degradation code, and
  `rows` lists up to five results with `state`, `conf`, `cos` and `source`.
- **Certified answer and abstention.** Unreachable with `hashing`. Prerequisite: a run launched with
  a real embedder where embedding is allowed. The vetted queries for that run are in
  `recall/quickstart.py` `DEMO_QUERIES`; the third must abstain.

## Gotchas

- The strict refusal is printed to stderr and the degraded answer to stdout; `control_recall`
  reads both for `code`.
- `conf` is not `cos`: a superseded row can have the highest cosine and still lose (see supersession).
- Development mode applies a fixed 0.5 demonstration threshold bound to nothing; its `state`
  values (`ok`, `low_confidence`) say nothing about a certified store.
