# Preregistration: structural edge performance evaluation

Date: 2026-09-11

Status: locked before the real performance measurement

## Objective

Measure whether the source-backed structural relations proposed for ATM and LoCoMo improve
retrieval and answer quality when they are used by the real indexing and retrieval path. The
earlier LoCoMo result in `docs/results/2026-09-10-structural-edge-effect.md` is a lexical routing
probe only. It is not reused as the primary result here.

The test must separate three questions:

1. Do the relations recover labelled evidence that ordinary retrieval misses?
2. Does the recovered evidence improve the generated answer under a fixed context budget?
3. What latency, candidate growth, token, and provider failure cost does the treatment add?

## Invariant and failure mode

The invariant is that every treatment addition comes from a relation emitted by
`benchmarks.structural_edges`, with a unique source-backed endpoint and no use of the question's
gold evidence or answer. The test is intended to catch a treatment that appears to improve recall
by using labels, free text entity guesses, an unbounded context, or a different retriever.

## Population and inputs

### LoCoMo

Use the checked out `locomo10.json` without question reordering. Include categories 1 through 4
with nonempty evidence for the primary retrieval and answer comparison. Run category 5 separately
as the adversarial refusal control. Record the input SHA256, conversation count, question counts,
and the count of dropped or malformed rows.

### ATM

Use the full ATM split when all four raw files are available: questions, emails, images, and
videos. The input files are not currently assumed to be present in this checkout. If any required
file is absent, ATM is `not_run`; no synthetic or historical ATM number is substituted. Record
each input SHA256 and the count of questions and memory items.

## Fixed stack

Both arms use the same process, database, embedder, reranker, query order, and retrieval candidate
pool. Indexing is done into isolated benchmark tenants or tables, never into a production tenant.
The checkout revision and all input hashes are recorded in the artifact.

The answer provider is OpenRouter with `deepseek/deepseek-v4-flash`, temperature zero, a 512 token
output ceiling, and the configured answer envelope and citation validator. The returned model name,
provider latency, prompt tokens, completion tokens, total tokens, and provider errors are recorded.
The API key is read from the environment and never enters an artifact.

## Arms and context budget

The primary comparison is paired at a fixed ten item evidence budget.

1. `baseline`: run the normal hybrid retriever and keep its top ten items.
2. `structural_edges`: run the same retriever for five seed items, then add at most five unique
   items reachable by one authored structural relation from those seeds. Order additions
   deterministically by structural type, structural key, and source order. If fewer than five
   valid additions exist, leave the remaining budget unused. No second semantic query is issued.

The treatment therefore has the same maximum answer context size as the baseline. The five seed
items and the five item addition cap are fixed before the first score. A separate diagnostic may
report the unbounded one hop set, but it cannot be used for the primary quality claim.

The baseline and treatment use the same indexed corpus bytes. Structural metadata is ignored by
the baseline selection logic and is consumed only by the treatment expansion step. The treatment
must fail closed when an endpoint is missing, ambiguous, malformed, or outside the indexed corpus.

## Primary outcomes

Report paired question-level metrics for each dataset and for each LoCoMo category:

* any gold evidence hit at ten items;
* complete gold evidence coverage at ten items;
* mean reciprocal rank of the first gold item;
* rescue rate among baseline misses;
* evidence precision in the ten item context;
* generated answer envelope validity and citation validity;
* official benchmark answer score when the official evaluator is available.

The primary quality gate is a paired improvement in complete gold evidence coverage. The answer
score is the primary end-to-end outcome only for rows that pass the fixed envelope and citation
checks. A missing provider response is a failure row, not a removed row.

Report retrieval latency, answer latency, total latency, candidate counts, added item counts,
prompt tokens, completion tokens, total tokens, and provider failure rate. Report median and p95
latency, retaining refused, slow, and failed requests in the total denominator.

## Prediction and decision rule

The prediction is a modest positive retrieval effect, not the 16.54 percentage point upper bound
from the lexical probe. I predict a one to four percentage point absolute improvement in complete
evidence coverage, a rescue rate between two and ten percent of baseline misses, and a precision
drop no larger than five percentage points under the fixed ten item budget. I predict that answer
quality will move in the same direction only if the recovered evidence is actually cited, and that
the treatment p95 total latency will remain below twice the baseline p95.

The treatment is useful only if all of these hold:

1. complete evidence coverage improves by at least two percentage points, or the paired answer
   score improves by at least two percentage points;
2. at least five percent of baseline misses are rescued;
3. evidence precision does not fall by more than ten percentage points;
4. answer envelope and citation validation failures stay below one percent;
5. treatment p95 total latency is no more than twice baseline p95; and
6. every relation type exercised by the run is reported separately.

If the population does not contain a relation type, the result is `not_exercised`, not zero
benefit. A result with no activated structural additions is a mechanism coverage failure and does
not justify a null quality claim.

## Answer scoring and judge controls

The library's `validate_answer` check is structural only. It proves format and citation identity,
not factual correctness. For ATM, use the official evaluator and the `atm` metric, not the more
expensive `llm atm` metric, unless a separate preregistration authorises it. For LoCoMo, retain the
exact generated answers and use one fixed external judge configuration for both arms if an
end-to-end score is required. Never compare one arm judged through OpenRouter with another arm
judged through a different route.

The answer provider is not a judge. A provider success count must never be reported as an answer
quality score.

## Invalid runs and artifact rules

The run is invalid if a tenant is indexed twice, if a benchmark table contains an unexpected row
count, if input hashes change between arms, if the answer model or prompt digest differs between
arms, or if the treatment context exceeds ten items. The run must stop rather than silently repair
any of these conditions.

Write one immutable raw artifact per arm and one summary artifact under `docs/results/` with a
timestamped name. Do not edit a raw artifact after its first measurement. Append corrections as a
new record. The summary must include the preregistration path, checkout revision, input hashes,
model and returned model identities, provider configuration without secrets, all row counts, and
the paired deltas with confidence intervals.

Before the measurement, prove the test can go red with a deliberate mutation that removes one
structural addition while keeping the same query and gold labels. The red proof must be a
behavioral assertion, not an import, fixture, timeout, or collection failure. Restore the
implementation before measuring the locked arms.
