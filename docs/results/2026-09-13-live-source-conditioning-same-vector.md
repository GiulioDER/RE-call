# Live same vector source conditioning result

**Measured:** 2026-09-13T17:58:29.805756+00:00  
**Registered verdict:** `BUILD SAMPLED SHADOW`  
**Generation:** `gen_808e6c2592494eebabf144eabb21f838`  
**Calibration:** `cal_bc65614ff5ba4471850d53384381d7d5`  
**Pipeline:** `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`  
**Corpus:** `079a63744f2a9ce44aae344a06f57c0f3d959499b674f66c95f3cfc16680e58a`  
**Source commit:** `4121b4b8da5d0c66273b4a52c5fe76ee9002210b`  
**Remote checkout:** `/home/sentiment/recall-repos/source-conditioning-shadow-0368ccba`  
**Raw artifact:** `docs/results/2026-09-13-live-source-conditioning-same-vector.json`  
**Raw artifact SHA256:** `0ff000bb0447023eaaf28925535cd00bceaa650dbc679821393502c666b3e6c5`

## Outcome

The versioned global alpha 0.08 source model retained its gold improvement on the refreshed Context
4 production corpus when both arms used the exact same query vector. The candidate improved complete
answerable queries from 17 of 22 to 18, and essential fact coverage from 19 of 25 to 20. It did not
lose any complete query or fact. False evidence answers remained one of 28 controls.

Labeled context precision improved from 0.4872 to 0.6122. Candidate hash recomputation and public
baseline hash linkage each matched all 50 requests. All 50 internal timing receipts were present,
no shadow request errored, and the run finished in 142461.045 ms.

| Arm | Complete | Facts | Source hits | Gold items | Total items | Precision | False answers |
|---|---:|---:|---:|---:|---:|---:|---:|
| public same vector baseline | 17/22 | 19/25 | 19/22 | 38 | 78 | 0.4872 | 1/28 |
| source conditioned candidate | 18/22 | 20/25 | 19/22 | 60 | 98 | 0.6122 | 1/28 |

## Where the gain came from

Only `memory-017`, “should I propose paid API runs for recall”, changed fact completeness. The
public baseline returned three items, including two from the correct source, but did not contain
enough labeled terms to cover `budget_rule`. Source conditioning returned five items, including
four chunks from `sentiment-agent/redacted-3aa583eee8b4`, and covered the
fact. The correct source had model support 0.603281. Its two newly useful chunks had raw cosine
0.510587 and 0.464188, and adjusted scores 0.518850 and 0.472451.

This is the intended mechanism: source evidence accumulates across dense and lexical ranks, then
slightly lifts additional chunks from a source that the query already supports. It is not graph
traversal and does not depend on authored relations.

## Mechanical and latency receipts

1. Candidate hash parity was 50 of 50.
2. Public baseline hash linkage was 50 of 50.
3. Shadow errors were zero.
4. Internal shadow scoring had median 1.026 ms, p95 2.029 ms, and maximum 2.364 ms.
5. End to end request latency had median 2313.438 ms, p95 3260.028 ms, and maximum 3827.654 ms.
6. Candidate selection returned five items on 19 queries, three on one, one on one, and zero on 29.

The internal span is the fitted model and selection cost after the benchmark audits already
collected candidate traces. It is not the total cost of a normal sampled production request. In the
preserved two process attempt two, shadow client latency exceeded off client latency by a median
1314.869 ms while the private audits were enabled. That comparison includes independent hosted
embedding calls and process variation, so it is a deployment warning rather than a clean overhead
estimate. Start only at a low sample rate until the main retrieval path can expose its existing
candidate traces without issuing duplicate database queries.

## Interpretation

The result crosses the registered quality and safety gates, and it does so after a corpus refresh
with a fixed model trained on the preceding Context 4 generation. This makes source conditioning
the first current retrieval treatment in this workstream with a repeated positive memory gold
direction beyond the embedding upgrade itself.

The evidence is still limited. The 50 query gold set influenced earlier design choices, only 22
queries are answerable, and the refresh changed little of the corpus. The result licenses sampled
shadow observation, not active serving. Promotion should require fresh independent gold or online
human judgments, plus a production overhead measurement without the benchmark audit path.

## Recommended next step

Review and deploy the existing mode at a low deterministic sample rate, initially 0.01 to 0.05,
with public results unchanged. Before raising the rate, refactor retrieval to retain the dense,
lexical, and trust pool traces already computed by the main request, so source conditioning does
not repeat database retrieval. Collect only aggregate counters and identifier hashes.

For the next quality experiment, build fresh blinded gold focused on queries where the baseline
finds the correct source but misses the decisive chunk. That is the failure shape repaired here and
offers more expected value than another graph traversal variant.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-shadow-0368ccba'
$env:RECALL_SOURCE_COMMIT='4121b4b8da5d0c66273b4a52c5fe76ee9002210b'
python -u scripts/run_live_source_conditioning_same_vector.py `
  --generation-id gen_808e6c2592494eebabf144eabb21f838 `
  --artifact docs/results/2026-09-13-source-conditioning-model.json `
  --output docs/results/2026-09-13-live-source-conditioning-same-vector.json
Get-FileHash docs/results/2026-09-13-live-source-conditioning-same-vector.json -Algorithm SHA256
```

The summary and latency statistics are reproduced from the raw artifact with:

```powershell
$code = @'
import json, math, statistics
from pathlib import Path
p=json.loads(Path('docs/results/2026-09-13-live-source-conditioning-same-vector.json').read_text(encoding='utf-8'))
def pct(values,q):
    xs=sorted(float(x) for x in values)
    pos=(len(xs)-1)*q
    lo=math.floor(pos); hi=math.ceil(pos)
    return xs[lo] if lo==hi else xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
print(p['decision'], p['summary'])
for key in ('shadow_internal_ms','client_observed_ms'):
    values=[row[key] for row in p['rows']]
    print(key, statistics.median(values), pct(values,0.95), max(values))
'@
$code | python -
```
