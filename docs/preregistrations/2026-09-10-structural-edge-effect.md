# Preregistration: benchmark structural edge effect

Status: protocol locked before the first structural edge effect measurement.

## Prior work searched

On 2026-09-10 I searched the local project memory and benchmark documentation for prior LoCoMo
graph activation, structural edge coverage, and retrieval effect measurements. The relevant prior
work was the graph activation diagnostic, which found that graph activation depended on seed
coverage, and the graph edge coverage result, which measured the production graph separately. No
prior result measured these new benchmark structural relations on the raw LoCoMo questions.

## Objective

Measure whether the newly added deterministic structural relations can recover gold LoCoMo turns
that a fixed top five lexical baseline misses. This is a routing and evidence reachability probe,
not an end to end answer quality claim.

## Population and data

Use the checked out `locomo10.json` without sampling or question reordering. Include categories 1
through 4 when the question has at least one string in `evidence`. Exclude category 5 because it
has no answerable gold evidence for this metric. Record the input SHA256, question count, and
conversation count in the result artifact.

## Arms

1. `baseline`: rank turns within each conversation by token overlap with the question and keep the
   first five turns. Ties are resolved by source turn order.
2. `structural_one_hop`: use the exact same five baseline seeds, then add every unique turn
   reachable by one authored structural relation in either direction. Preserve baseline order,
   then order added turns by relation structural type, structural key, and source turn order.

Tokenization is lowercase alphanumeric runs. No gold answer, evidence identifier, or entity is
used to construct a query or a relation. Structural metadata is produced only by
`locomo_structural_metadata`; free text is not scanned for entities.

## Primary outcomes

Report paired evidence hit rate over the full included population and rescue rate among baseline
misses. A rescue means that every gold evidence turn is present in the expanded treatment set and
was absent from the baseline top five.

Report the per question paired outcomes, so gains and regressions are auditable. Also report
candidate expansion size, expanded evidence precision, and results by structural type.

## Prediction and decision rule

I predict a positive but modest rescue effect. The structural edge proposal is useful on this
probe only if the treatment increases full population hit rate by at least 0.02 absolute, rescues
at least 0.05 of baseline misses, and expanded evidence precision does not fall by more than 0.10
absolute. Otherwise this probe does not justify retaining the added structural relations for
retrieval use, even if the structural coverage tests remain green.

The fixed top five baseline and the unbounded one hop expansion mean this is an upper bound on
the retrieval benefit at a matched semantic retrieval budget. Candidate growth is therefore a
required secondary result, not hidden in the primary score.

## Analysis and artifact rules

Use the committed protocol and script without editing predictions or thresholds after measurement.
Write results to a separate timestamped JSON artifact under `docs/results/`. Do not modify this
file after the first measurement. If the raw ATM inputs are unavailable, report ATM as not run
and do not substitute synthetic counts for a benchmark quality result.
