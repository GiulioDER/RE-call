# Atomic fact release confirmation construction

Status: preregistered after the candidate inventory passed and before any question writer call.

## Purpose

Construct a private 96 source prospective quality set from the frozen atomic-development-disjoint
inventory. The question writer sees only one exact fact span. It never sees source identity,
document structure, rendered atomic context, parent chunk, retrieval candidates, scores, or prior
questions.

No embedding or retrieval runs in this phase. A later committed preregistration must freeze the
completed private pool hash before any dense or atomic result is measured.

## Frozen inventory

Use the private inventory with SHA256
`a5df11069bce5eb4041e87ca5ebf3e3b1d8b35fb3b1814b61cd1725e46eba768`, candidate manifest
SHA256 `fdd00ba7d356b4b07d376c5b2d87b146c2f58c45bf35ea55b17d85e21fcf8e62`, and source snapshot
SHA256 `12885c6ea1cde2f81758d3cc247b77e3127ba10484f1c257289f9d3946b7722d`.
It contains 1,070 candidates ordered by
`SHA256("atomic-fact-release-confirmation-v1" + "\0" + source)`, then by source.

Before any model call, reconstruct every candidate from the seven source roots and require its
source SHA256, gold parent ordinal, answer span SHA256, construction, order, inventory file hash,
candidate manifest hash, and source snapshot hash to match. Any mismatch is a hard stop.

## Frozen question writer

Use OpenRouter model `deepseek/deepseek-v4-flash`, temperature zero, maximum 2,048 completion
tokens, and the repository retry policy. Use the exact system prompt and user template from the
prior blind census. The writer receives only the exact fact content inside the delimited data
block.

Call the model exactly once for each ordered candidate until 96 questions have been accepted or
160 candidates have been attempted. A validation rejection advances to the next candidate. Never
retry a rejected output, change its prompt, expose more context, or substitute a different fact
from the same source.

An output is accepted only when all of these conditions hold:

1. The response parses as exactly one JSON object with exactly one string field named `question`.
2. The collapsed question contains 6 through 28 word tokens and ends with one question mark.
3. It contains none of these case-insensitive whole words: `document`, `memory`, `memo`, `source`,
   `title`, `heading`, `field`, `record`, `recorded`, `text`, `passage`, and `according`.
4. It does not contain the normalized answer span.
5. It shares no contiguous sequence of five normalized word tokens with the answer span.
6. Its normalized form is unique among accepted questions.
7. It contains no URL, Markdown code fence, newline, NUL, or control character.

The private row records question, source identifier, source SHA256, gold parent ordinal, exact
answer span, answer span SHA256, construction, and candidate order. It remains outside the
repository. The public receipt contains only hashes, aggregate counts, validation rejections,
root and construction distributions, and provider usage.

## Frozen construction gates

Return `READY_TO_PREREGISTER_ATOMIC_RELEASE_RETRIEVAL` only if all conditions hold:

1. The inventory file, candidate manifest, and source snapshot hashes match the frozen values.
2. Exactly 96 accepted rows come from exactly 96 distinct candidate sources within at most 160
   attempts.
3. Every source, answer span, parent ordinal, construction, and order matches the frozen inventory
   and current frozen source snapshot on a second audit.
4. Every accepted question passes all frozen validation rules and is normalized-unique.
5. At least three production roots are represented and no root supplies more than 80 percent of
   accepted rows.
6. Provider metadata accounts for exactly one successful completion per attempted candidate and
   reports no unhandled generation error.
7. The private pool remains outside the repository and its Windows ACL does not grant `Everyone`,
   `BUILTIN\Users`, or `Authenticated Users`.

Otherwise return `STOP_ATOMIC_RELEASE_CONFIRMATION_CONSTRUCTION`. Do not reduce the 96 row target,
increase the 160 attempt ceiling, loosen validation, alter the prompt, change the model, or
resample under this protocol.

## Prediction

I predict the construction will pass. The prior frozen writer accepted 31 of 45 candidates, so the
160 attempt ceiling provides substantial margin for 96 accepted rows while preserving one call per
source and the exact prior validation rules.
