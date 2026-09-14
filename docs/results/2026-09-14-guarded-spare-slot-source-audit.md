# Guarded spare-slot source audit

Date: 2026-09-14.

## Verdict

`INVALID_REVIEW_INSTRUMENT`.

The human review was retired before arm identities were opened. Every query in the blinded review
had an empty `required_facts` list, so the reviewer had evidence but no independent fact-level answer
key. No spreadsheet response was collected or consumed.

The registered replacement audit is descriptive only. It cannot promote or reject the guarded
rescue and does not measure answer sufficiency or false-answer rate.

## Objective source result

The audit joined the hidden base and candidate hashes to evidence sources through the frozen blind
map, then compared those sources with the labels frozen in the fresh pool before screening.

| Metric | Base | Candidate | Change |
|---|---:|---:|---:|
| Answerable queries with a gold source hit | 14 of 23 | 19 of 23 | 5 gains, 0 losses |

The treatment appended 74 items across the 43 triggered queries. Ten additions came from a frozen
gold source. Twenty-eight additions on answerable queries came from other sources, and all 36
additions on unanswerable controls were non-gold by construction. Added-item gold-source precision
was `0.1351351351`.

The source signal is real enough to preserve as development evidence: the rescue reached the
canonical source for five queries the base missed. Its present selectivity is too weak for a safe
global promotion. It found one gold-source addition for roughly six non-gold additions, and the
current artifact cannot determine whether any added chunk supplied a required fact.

## Integrity

The query pool digest was
`af7c74d4d2b232cb79de72d5fdfd67e1ccf1d6ad814fb634ba61f98999632c75`. The private capture digest
was `30b270a53753b2fdc152197b94f655e346da9810608e851782a8b4b1af860b36`. Capture and review query
sets matched, every candidate preserved the complete base prefix, and the scorer consumed no human
labels. The public JSON contains aggregate values only.

## Reproduction

Run from the repository root with access to the private frozen artifacts:

```powershell
python scripts/score_guarded_spare_slot_source_audit.py --pool docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json --capture C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14\capture.json --review C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14\blind-review.json --output docs/results/2026-09-14-guarded-spare-slot-source-audit.json
python -m pytest tests/test_guarded_spare_slot_source_audit.py -q
python -m ruff check scripts/score_guarded_spare_slot_source_audit.py tests/test_guarded_spare_slot_source_audit.py
git diff --check
```

Machine-readable result:
`docs/results/2026-09-14-guarded-spare-slot-source-audit.json`.

## Next experiment

Use the five source-hit gains and the non-gold additions only as development data for a stricter
activation rule. Confirm the revised rule on a new untouched holdout whose answerable queries freeze
an exact source ordinal and answer span before retrieval. Score admitted evidence against that span
and require an extractive answer stage to return the span exactly or `NOT_FOUND`. This removes the
need for recollection, link navigation, or post-treatment human fact authoring.
