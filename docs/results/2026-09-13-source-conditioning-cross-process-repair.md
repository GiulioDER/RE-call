# Source conditioning cross process validation repair

**Measured:** 2026-09-13T17:46:00Z  
**Registered verdict:** `REPAIR`  
**Final attempt source commit:** `784288ea7e1f1ef0b21728017d6292887c6e685d`  
**Final attempt artifact:** `docs/results/2026-09-13-live-source-conditioning-shadow-attempt-3.json`  
**Final attempt SHA256:** `57cd5afb5e7e9fd4bdd02647dd28f3a5b4858ba41b1434ccf23f8a81b8c0b2e0`

## Outcome

The fitted global alpha 0.08 model retained its gold direction on the refreshed Context 4 corpus,
but the registered two process parity test is not a stable validation design. Each process asks the
hosted embedder for its own query vector. The off arm itself changed at the admission boundary
across attempts: `memory-001` returned four public items in attempt two and two public items in
attempt three. The final attempt therefore had 49 of 50 public identity and order matches even
though all 50 shadow candidate hash comparisons matched and no shadow computation errored.

The final attempt reached 18 of 22 complete answerable queries and 20 of 25 essential facts for the
candidate, versus 17 and 19 for the independently embedded off arm. Both arms answered one of 28
unanswerable controls. Candidate labeled context precision was 0.6122. These quality numbers are
directional evidence, not the final paired estimate, because the two arms did not share a vector.

## Why the next protocol changes

The shadow candidate already uses the exact query vector that produced the public baseline inside
its own request. The sound comparison is therefore the public baseline and private candidate from
that one request. This removes hosted embedding variation without changing retrieval or replaying a
manufactured vector. A baseline hash receipt will bind the diagnostic to the public evidence list.

The separate process test did prove useful mechanical facts. Across the final attempt, candidate
hash recomputation matched 50 of 50 requests, shadow errors were zero, and the 100 requests finished
in 199359.453 ms. Those measurements do not authorize deployment because the registered public
parity gate failed.

## Reproduction

The final attempt was run with:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-shadow-0368ccba'
$env:RECALL_SOURCE_COMMIT='784288ea7e1f1ef0b21728017d6292887c6e685d'
python -u scripts/run_live_source_conditioning_shadow.py `
  --generation-id gen_808e6c2592494eebabf144eabb21f838 `
  --artifact docs/results/2026-09-13-source-conditioning-model.json `
  --output docs/results/2026-09-13-live-source-conditioning-shadow.json
```

The cross attempt baseline variation is reproduced from the preserved artifacts with:

```powershell
$code = @'
import json
from pathlib import Path
a=json.loads(Path('docs/results/2026-09-13-live-source-conditioning-shadow-attempt-2.json').read_text(encoding='utf-8'))
b=json.loads(Path('docs/results/2026-09-13-live-source-conditioning-shadow-attempt-3.json').read_text(encoding='utf-8'))
for ar,br in zip(a['rows'],b['rows']):
    ai=[(x['chunk_id'],x['verdict']) for x in ar['baseline_items']]
    bi=[(x['chunk_id'],x['verdict']) for x in br['baseline_items']]
    if ai != bi:
        print(ar['query']['id'], len(ai), len(bi), ai, bi)
'@
$code | python -
```
