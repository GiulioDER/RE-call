# Guarded spare slot fresh screen

Measured 2026-09-14.

## Verdict

`READY_FOR_BLIND_REVIEW`

The frozen query pool reached the preregistered triggered cohort after 270 requests. The cohort
contains 23 answerable and 20 unanswerable queries. All 270 requests carried a shadow timing
receipt, and every triggered candidate preserved the complete alpha `0.08` prefix.

This is an activation and integrity result only. It is not evidence of higher retrieval quality or
gold coverage. The base and candidate arms remain unrevealed until the human item review is
complete.

## Frozen identity

| Field | Value |
|---|---|
| Source commit | `98b6ee9d` |
| Query pool SHA256 | `af7c74d4d2b232cb79de72d5fdfd67e1ccf1d6ad814fb634ba61f98999632c75` |
| Generation | `gen_98c6f34508384ee2badffc14fcf47c4c` |
| Calibration | `cal_2cc3192509b64524a71aa949918267d8` |
| Pipeline fingerprint | `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86` |
| Corpus fingerprint | `a31ac76d107eb3705ea26460f10b59edd7710df52a6088aad3f67fa3c8371085` |
| Capture SHA256 | `30b270a53753b2fdc152197b94f655e346da9810608e851782a8b4b1af860b36` |
| Blind review SHA256 | `e24ff6cd655a9da12f2c8d2245fd46f3b7ad9f73375ca7134b13d6812f765cca` |

The private artifacts are stored outside the public repository at:

```text
C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14\capture.json
C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14\blind-review.json
```

The blind review packet contains 43 queries and exposes no base or candidate membership, no
candidate hash, and no capture mapping. All 43 review records are incomplete by construction.

## Reproduction

The successful screen command was:

```powershell
$evalDir = 'C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14'
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/guarded-spare-slot-89caaf43'
$env:RECALL_SOURCE_COMMIT='98b6ee9d'
python -u scripts/run_live_guarded_spare_slot_fresh_screen.py --query-pool docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output "$evalDir\capture.json" --review-output "$evalDir\blind-review.json" --generation-id gen_98c6f34508384ee2badffc14fcf47c4c --calibration-id cal_2cc3192509b64524a71aa949918267d8 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint a31ac76d107eb3705ea26460f10b59edd7710df52a6088aad3f67fa3c8371085
```

Verify the aggregate integrity without printing review content:

```powershell
@'
import hashlib,json
from pathlib import Path
root=Path(r'C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14')
cap=json.loads((root/'capture.json').read_text(encoding='utf-8'))
rev=json.loads((root/'blind-review.json').read_text(encoding='utf-8'))
print(cap['screen_decision'], cap['processed_queries'], cap['triggered'])
print('missing_timing',sum(r['shadow_internal_ms'] is None for r in cap['rows']))
print('bad_prefix',sum(bool(r['triggered']) and r['candidate_hashes'][:len(r['base_hashes'])] != r['base_hashes'] for r in cap['rows']))
print('review_queries',len(rev['queries']))
for name in ('capture.json','blind-review.json'):
    print(name,hashlib.sha256((root/name).read_bytes()).hexdigest())
'@ | python -
```

## Next gate

The human reviewer fills `required_facts`, `supports_question`, `covered_facts`, and
`review_complete` in the blind review packet without opening the capture. Only then may the scorer
join the neutral evidence identifiers to the hidden arm membership and evaluate the promotion
rule.
