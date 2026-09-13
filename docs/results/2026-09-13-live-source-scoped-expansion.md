# Live source scoped expansion result

**Measured:** 2026-09-13T16:50:02.103151+00:00  
**Registered verdict:** `CLOSE`  
**Generation:** `gen_18d5edd2e5e847c0af1ee37e40d27893`  
**Calibration:** `cal_23d8708ac550444fa4274ac617df0870`  
**Pipeline:** `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`  
**Corpus:** `c054e26b28e62760c0fefa64815da8033433e94c2474f079ed8a2aec1b93fb1a`  
**Source commit:** `1ef5ae869e8d15739b710212af0897a77401106d`  
**Remote checkout:** `/home/sentiment/recall-repos/source-scoped-expansion-1ef5ae86`  
**Raw artifact:** `docs/results/2026-09-13-live-source-scoped-expansion.json`  
**Artifact SHA256:** `531dfa31c4344df020ebda9be46e2b09365b0dd9c918aff6835f8c5b4e1bd666`

## Outcome

Searching eight chunks inside each of the top three source support sources did not improve gold
over applying source conditioned admission to the global fused pool. The registered top three
alpha 0.08 treatment reached 17 complete queries and 19 of 25 facts, one fewer on each metric than
its global alpha 0.08 control. Safety stayed at one false evidence answer and context precision rose
from the served baseline 0.4935 to 0.6277, but those secondary gains cannot override the registered
gold rule.

The experiment therefore closes source scoped expansion in this form. It does not close source
conditioning itself. On the Context 4 substrate, global alpha 0.08 improved the served baseline from
17 to 18 complete queries and from 19 to 20 facts while keeping one false answer. Global alpha 0.15
reached 19 complete queries and 21 facts, also with one false answer. The alpha 0.08 replication is
the higher ROI implementation candidate because it was selected by the earlier frozen rule. Alpha
0.15 remains a validation candidate, not a serving choice from this reused gold set.

## Registered arms

| Arm | Complete | Facts | Source hits | Precision | False answers |
|---|---:|---:|---:|---:|---:|
| served Context 4 baseline | 17/22 | 19/25 | 19/22 | 0.4935 | 1/28 |
| global alpha 0.08 | 18/22 | 20/25 | 19/22 | 0.6122 | 1/28 |
| global alpha 0.15 | 19/22 | 21/25 | 20/22 | 0.6465 | 1/28 |
| scoped one source, alpha 0.00 | 12/22 | 14/25 | 12/22 | 0.6301 | 0/28 |
| scoped one source, alpha 0.08 | 12/22 | 14/25 | 12/22 | 0.6184 | 0/28 |
| scoped one source, alpha 0.15 | 13/22 | 15/25 | 13/22 | 0.6173 | 0/28 |
| scoped two sources, alpha 0.00 | 17/22 | 19/25 | 18/22 | 0.6237 | 1/28 |
| scoped two sources, alpha 0.08 | 17/22 | 19/25 | 18/22 | 0.6383 | 1/28 |
| scoped two sources, alpha 0.15 | 18/22 | 20/25 | 19/22 | 0.6562 | 1/28 |
| scoped three sources, alpha 0.00 | 17/22 | 19/25 | 18/22 | 0.5914 | 1/28 |
| scoped three sources, alpha 0.08 | 17/22 | 19/25 | 18/22 | 0.6277 | 1/28 |
| scoped three sources, alpha 0.15 | 18/22 | 20/25 | 19/22 | 0.6458 | 1/28 |

## Prediction accounting

1. The primary gold prediction was falsified. Top three alpha 0.08 lost one complete query and one
   fact against global alpha 0.08 instead of gaining one complete query or two facts.
2. The safety prediction held. Top three alpha 0.08 kept one false answer, equal to its control and
   below the ceiling of two.
3. The precision prediction held at 0.6277 against the registered floor of 0.48.
4. The source budget prediction held. At alpha 0.08, top three covered 19 facts versus 14 for top
   one. Alpha 0.15 showed 20 versus 15.
5. The runtime prediction held. Exactly 200 retrieval requests completed in 374753.764 ms, about
   6.25 minutes, below 30 minutes. This serial audit is not a production latency estimate.

## Case analysis after the decision

Context 4 plus global source conditioning already repaired most of the old misses. Global alpha
0.15 leaves only three incomplete answerable queries.

1. The always passing guard query has its gold source at support rank 11. A top three source cutoff
   cannot reach it.
2. The clean Git status query has its gold source at rank two. Its fact chunk is third by cosine
   inside that source, has verdict `ok`, and cosine 0.419665. It survives source retrieval but loses
   the final five item competition to chunks from the stronger source. This is evidence for a small
   source diversity selection audit, not for deeper source retrieval.
3. Gate 1774 has its gold source at rank one. Its two fact chunks are first and third inside that
   source, but their cosines 0.384160 and 0.369286 remain below the 0.4100 certified threshold after
   alpha 0.15 because source support is only 0.516513. Recovering it requires a different admission
   signal and must keep the 28 negative controls binding.

The scoped treatment also loses the merge policy fact because its gold source is support rank five.
This is the direct reason the top three arms cannot match the global controls even when their
within source retrieval succeeds.

## Decision and next action

Do not build top three source scoped expansion. Preserve the global pool and use source support as
an admission feature there.

The next implementation candidate is global alpha 0.08 in shadow mode. It now has the original
Voyage 4 safety result and a directionally stronger Context 4 replication. Before serving, its
source model needs a production fitting and versioning contract rather than leave one query out
evaluation machinery.

If another retrieval experiment is preferred first, the highest value bounded test is an offline
source diversity selector over this frozen artifact. It should reserve one eligible chunk from the
second ranked source before filling the five item budget. Its visible ceiling is one additional fact
on this set, so it ranks below implementing and validating global alpha 0.08.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-scoped-expansion-1ef5ae86'
$env:RECALL_SOURCE_COMMIT='1ef5ae869e8d15739b710212af0897a77401106d'
python -u scripts/run_live_source_scoped_expansion.py `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-scoped-expansion.json
Get-FileHash -Algorithm SHA256 docs/results/2026-09-13-live-source-scoped-expansion.json
```

The focused verification command was:

```powershell
python -m pytest tests/test_live_source_scoped_expansion.py tests/test_live_source_conditioned_admission.py tests/test_live_retrieval_leg_audit.py tests/test_live_document_expansion_audit.py tests/test_voyage_context4_production.py tests/test_doc_citations.py -q
python -m ruff check scripts/run_live_source_scoped_expansion.py tests/test_live_source_scoped_expansion.py
python -m mypy --explicit-package-bases --follow-imports=skip scripts/run_live_source_scoped_expansion.py
```

That verification passed 54 tests before measurement, with Ruff and mypy clean.
