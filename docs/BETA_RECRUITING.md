# Beta recruiting for RE-call

This workflow is for finding likely beta users from public discussions without collecting or
processing public email addresses, phone numbers, or other direct contact fields.

It is useful for RE-call because the likely buyers and evaluators already discuss their pain in
public: stale agent memory, unsupported answers, provenance gaps, tenant isolation, and memory
systems that guess instead of abstaining.

## What to collect

Collect only discussion level signals:

| Field | Purpose |
|---|---|
| `platform` | Route the outreach playbook, for example Reddit versus Discord. |
| `community` | Keep the queue grouped by subreddit, server, forum, or site section. |
| `url` | Link back to the original public discussion. |
| `title` | Rank by visible pain point language. |
| `body` | Rank by details such as stale memory, hallucination, provenance, or compliance. |
| `author_handle` | Public handle for manual in-platform follow up. |
| `posted_at` | Prefer fresh threads where a reply is still useful. |
| `replies`, `reactions`, `upvotes`, `comments` | Prioritize threads with visible demand. |
| `tags` | Add your own labels during review. |

Do not ingest direct contact fields. The utility rejects exports that contain columns like
`email`, `contact_email`, or `phone`.

## Ranking utility

The repository includes `scripts/build_beta_queue.py`, which turns a CSV or JSONL export of public
discussions into a ranked outreach queue.

Show the accepted input fields:

```powershell
python scripts/build_beta_queue.py placeholder.csv --show-fields
```

Build a queue for RE-call:

```powershell
python scripts/build_beta_queue.py discussions.csv `
  --term "agent memory" `
  --term "stale memory" `
  --term "hallucination" `
  --term "provenance" `
  --term "tenant isolation" `
  --term "compliance" `
  --out beta_queue.csv `
  --top 75
```

The output ranks public threads and includes:

| Column | Meaning |
|---|---|
| `score` | Combined topic, engagement, and recency score. |
| `matched_terms` | Which pain terms were found in the post. |
| `action` | Suggested in-platform next step, for example `public_reply` or `community_post`. |
| `message_angle` | The angle to use in the reply or thread. |

## Recommended campaign shape

1. Research communities first: subreddits, Discord servers where posting is allowed, X lists,
   forums, and product communities where agent memory problems are discussed openly.
2. Export only the discussion fields above.
3. Rank the queue with `build_beta_queue.py`.
4. Manually review the top rows.
5. Reply in-thread or post in-community with a waitlist link and explicit opt-in.
6. Keep a separate CRM only for users who voluntarily sign up.

## Messaging angles for RE-call

Use one of these depending on the matched pain:

| Pain | Angle |
|---|---|
| Stale or contradictory memory | Validity aware retrieval and supersession. |
| Hallucinated answers from memory | Explicit abstention when memory does not support an answer. |
| Compliance or enterprise concerns | Tenant isolation, provenance, and policy controls. |
| General retrieval quality pain | Trustworthy memory over plain nearest neighbor search. |

Example public reply:

> We built RE-call for this exact memory failure mode: retrieval with provenance, validity, and an
> explicit abstain path when the memory does not support the answer. If you are testing agent
> memory stacks, I have a beta waitlist here: `<your link>`.

Adjust the wording to the community rules. Do not mass post the same copy.

## What this workflow is not for

This is not a bulk cold email pipeline. It is a public discussion research and opt-in conversion
workflow. If someone wants updates, send them to a waitlist or booking page where they knowingly
submit their details.
