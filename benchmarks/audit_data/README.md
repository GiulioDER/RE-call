# Vendored LOCOMO audit data

`locomo_errors.json` is a verbatim copy of a third-party audit of the LOCOMO benchmark's answer
key. It is consumed by `benchmarks/locomo_audit.py`, which reports our results both as published
and with the audited-defective questions removed.

## Source

| | |
|---|---|
| URL | <https://raw.githubusercontent.com/dial481/locomo-audit/main/errors.json> |
| Repository | <https://github.com/dial481/locomo-audit> |
| Commit | `ded8d0f08903` — "LoCoMo benchmark ground truth audit", 2026-02-18 (the only commit touching `errors.json`) |
| Fetched | 2026-07-24 |
| Bytes | 150,552 |
| Entries | 156 |

## Licence — CC BY-NC 4.0, attribution required

The audit is published under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), the
same licence as the underlying LOCOMO dataset it audits.

- **Attribution.** Any publication using these numbers must credit `dial481/locomo-audit` and link
  the repository. The module docstring and this file carry that link; so must the article.
- **NonCommercial.** This file is used for research and for a published methodology comparison. It
  is not redistributed in any built artifact: `[tool.hatch.build.targets.wheel].packages` is
  `["recall", "recall_mcp"]`, so `benchmarks/` — and therefore this directory — is never part of
  the `recall-rag` wheel or sdist on PyPI. Keep it that way.

## What the file contains, and the count that matters

156 entries, one per defective question, each with `question_id`, `question`, `golden_answer`,
`category`, `error_type`, `cited_evidence`, `correct_evidence`, `reasoning` and `correct_answer`.

`error_type` splits them into two populations that must not be conflated:

| `error_type` | n | Golden answer wrong? |
|---|---:|---|
| `WRONG_CITATION` | 57 | No — only the cited evidence turn is wrong |
| `HALLUCINATION` | 33 | Yes |
| `TEMPORAL_ERROR` | 26 | Yes |
| `ATTRIBUTION_ERROR` | 24 | Yes |
| `AMBIGUOUS` | 13 | Yes |
| `INCOMPLETE` | 3 | Yes |
| **score-corrupting total** | **99** | |

**99** is the authoritative corrupted-question count, and it is the non-`WRONG_CITATION` subset —
not the file's row count. It reconciles with both of the audit's own published claims:
`1 - 99/1540 = 0.9357`, the stated 93.57% ceiling, and the repository README's "156 total issues:
99 score-corrupting, 57 citation-only". `benchmarks/locomo_audit.py` therefore excludes 99 by
default and takes `include_citation_only=True` to widen to 156.

## Redundant copies upstream

The repository also publishes `audit/errors_conv_0.json` … `errors_conv_9.json`. Fetched
2026-07-24 and compared entry-by-entry against `errors.json`: **the same 156 `question_id`s, and
every field of every entry identical** (27, 7, 12, 22, 17, 10, 15, 13, 16, 17 per conversation).
They differ from `errors.json` only in ordering and in being split per conversation.

`errors.json` is used here: the upstream README calls it the "Consolidated error report (all
conversations)", and one file with no concatenation step is one fewer place for a reader to have
to reproduce our join.

## qa numbering

The audit's ids are `locomo_{c}_qa{n}` where `c` is the 0-based position in `locomo10.json`.
`n` is **0-based** — established, not assumed, by matching text against the dataset: all 156
entries match their question at `qa[n]` and all 156 match their golden answer, while at `qa[n-1]`
0 of 156 match. `benchmarks/locomo_audit.py` re-runs that verification on every call and raises
rather than exclude questions on an unverified mapping.
