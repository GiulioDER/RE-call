"""Authored corpus and queries for the successor directed expansion measurement.

Pre-registered in `docs/preregistrations/2026-08-19-successor-directed-expansion.md`. Apparatus
check 4 of that record forbids the defect that withdrew the successor column from the generated
scale corpus (`results/FINDINGS.md:376`): hundreds of documents that were one sentence with a
different opaque token, so the column measured token discrimination rather than the trust layer.

Everything here is therefore hand written, and each pair is built to two rules:

1. **v2 must change the ANSWER, not just a number.** A reader who believed v1 is wrong about what
   to do after reading v2, not merely out of date on a constant.
2. **v2 must be worded UNLIKE v1.** This is the adversarial half and it is deliberate. A successor
   that repeats its predecessor's vocabulary is retrieved by the same query that found the
   predecessor, which puts it in the pool for free and makes the measurement trivially easy. Real
   corrections are written by someone reframing the problem, and those are exactly the successors
   that fall outside the pool.

The distractor mass is NOT authored here. The probe indexes this corpus alongside the repository's
own `docs/`, which is real prose of the same register and large enough that the fused pool cannot
hold it. Authoring distractors would have reintroduced defect 4 by the back door.
"""

from __future__ import annotations

from dataclasses import dataclass

#: One supersession pair. `query` is worded from v1's vocabulary on purpose: it is the adversarial
#: case the trust layer exists for, where nearest-match retrieval prefers the memory that is wrong.
@dataclass(frozen=True)
class Pair:
    slug: str
    v1: str
    v2: str
    query: str


PAIRS: tuple[Pair, ...] = (
    Pair(
        slug="abstention_floor",
        v1=(
            "# Abstention floor\n\n"
            "The service refuses to answer when the top cosine falls below 0.62. The floor was "
            "picked by eye from a histogram of the first week of traffic, and it is applied "
            "globally, the same number for every tenant and every embedder. Status: adopted.\n"
        ),
        v2=(
            "# Refusal is now fitted per corpus, not chosen\n\n"
            "A hand-picked global number cannot separate answerable from unanswerable once two "
            "tenants disagree about what a good match looks like. Refusal is now derived from a "
            "labelled set for each tenant, refitted whenever the encoder or the corpus changes, "
            "and a tenant with no fitted artifact is served degraded rather than guessed at.\n"
        ),
        query="what cosine does the service refuse below, and who picked that floor",
    ),
    Pair(
        slug="reranker_choice",
        v1=(
            "# Cross encoder selection\n\n"
            "We rerank the top candidates with ms-marco-MiniLM-L-6-v2. It was the cheapest model "
            "that beat no reranking at all on the first evaluation, and it runs on CPU inside the "
            "request. Status: adopted.\n"
        ),
        v2=(
            "# Reranking is now a pool size decision\n\n"
            "Swapping the model was the wrong axis. The measured effect is dominated by how many "
            "candidates the stage is shown: a wide pool loses accuracy no matter which encoder "
            "scores it, because the encoder is given more rope rather than more to choose from. "
            "The stage is capped first and the model chosen second.\n"
        ),
        query="which cross encoder model do we use to rerank the top candidates",
    ),
    Pair(
        slug="chunk_size",
        v1=(
            "# Indexing chunk size\n\n"
            "Documents are split into 512 token chunks with 50 tokens of overlap, the framework "
            "default. The value was never revisited after the first import. Status: adopted.\n"
        ),
        v2=(
            "# Splitting follows structure, not arithmetic\n\n"
            "Counting tokens cuts through the middle of tables and separates a heading from the "
            "section it names, so the vector describes a fragment nobody would have written. "
            "Boundaries now follow the document: headings, list and table units stay whole, and "
            "size is a range the packer respects rather than a target it hits exactly.\n"
        ),
        query="how many tokens per chunk and how much overlap do we index with",
    ),
    Pair(
        slug="embedding_model",
        v1=(
            "# Encoder selection\n\n"
            "Retrieval runs on a hosted 1536 dimension encoder. It scored best on a public "
            "leaderboard, and the hosted API removed the need to ship model weights. "
            "Status: adopted.\n"
        ),
        v2=(
            "# The encoder is now a deployment constraint, not a leaderboard rank\n\n"
            "Leaderboard order did not survive contact with this corpus, and sending every "
            "passage to a third party turned an indexing job into a data transfer question. "
            "Selection is now made locally, on the corpus that will be served, and any hosted "
            "option is opt in with the transfer stated.\n"
        ),
        query="which hosted encoder do we embed with and how many dimensions",
    ),
    Pair(
        slug="index_refresh",
        v1=(
            "# Index refresh cadence\n\n"
            "The whole corpus is reindexed nightly at 02:00. A full rebuild is simpler to reason "
            "about than incremental updates, and the corpus is small enough that the rebuild "
            "finishes before the working day. Status: adopted.\n"
        ),
        v2=(
            "# Rebuilds are content addressed and incremental\n\n"
            "A nightly full rebuild spends most of its time re-embedding text that did not "
            "change, and it hides deletions: a document removed from disk stayed searchable "
            "until the next run. Files are now compared by content hash and only changed ones "
            "are re-embedded, with a guard that refuses to prune when most of the corpus has "
            "apparently vanished.\n"
        ),
        query="when does the nightly full reindex of the whole corpus run",
    ),
    Pair(
        slug="tenant_isolation",
        v1=(
            "# Tenant separation\n\n"
            "Each customer gets a separate database. Separation at the database boundary is easy "
            "to explain to an auditor and impossible to get wrong in a query. Status: adopted.\n"
        ),
        v2=(
            "# One table, a tenant column, and a policy the database enforces\n\n"
            "A database per customer multiplied migrations, connection pools and backups by the "
            "customer count, and the operational cost arrived long before the second auditor "
            "did. Rows now carry a tenant identifier, the isolation is a row level policy the "
            "database enforces rather than a convention queries follow, and it is verified as an "
            "unprivileged role because a superuser bypasses it.\n"
        ),
        query="does each customer get their own separate database for separation",
    ),
    Pair(
        slug="retry_budget",
        v1=(
            "# Store retry budget\n\n"
            "A failed query to the store is retried up to five times with a fixed 200ms pause "
            "between attempts. Status: adopted.\n"
        ),
        v2=(
            "# Retries now distinguish the fault from the load\n\n"
            "A fixed pause multiplied by five turned a momentarily busy database into a request "
            "that hung for a second and then failed anyway, and it retried faults that could "
            "never succeed. Only transient connection faults are retried now, with backoff, and "
            "a statement that timed out is reported as a timeout rather than attempted again.\n"
        ),
        query="how many times do we retry a failed store query and how long between attempts",
    ),
    Pair(
        slug="latency_budget",
        v1=(
            "# Latency budget\n\n"
            "The p95 budget for a retrieval call is 400ms end to end, measured from the client. "
            "Status: adopted.\n"
        ),
        v2=(
            "# The budget is stated per stage, because one number hid the cause\n\n"
            "An end to end figure told us a call was slow and nothing about which part. Each "
            "stage now reports its own milliseconds, so a widened candidate scan and a slow "
            "reranker are distinguishable, and the aggregate is derived from those rather than "
            "measured separately.\n"
        ),
        query="what is the p95 latency budget end to end for a retrieval call",
    ),
    Pair(
        slug="eval_query_set",
        v1=(
            "# Evaluation query set\n\n"
            "Quality is tracked with twelve questions written by the team during the first "
            "sprint. They are answerable by design, so the score is easy to read at a glance. "
            "Status: adopted.\n"
        ),
        v2=(
            "# A set with no unanswerable questions cannot score a refusal\n\n"
            "Twelve answerable questions measure ranking and nothing else: a system that never "
            "refuses scores perfectly on them while being wrong in exactly the way that matters. "
            "The set now carries both classes, and any fitting procedure refuses a one sided "
            "file rather than fitting a threshold to it.\n"
        ),
        query="how many questions are in the evaluation set the team wrote in the first sprint",
    ),
    Pair(
        slug="backup_schedule",
        v1=(
            "# Backup schedule\n\n"
            "A full dump is taken weekly on Sunday and kept for four weeks. Restores have not "
            "been exercised. Status: adopted.\n"
        ),
        v2=(
            "# A backup nobody has restored is not a backup\n\n"
            "The weekly dump was real and the recovery path was theoretical, which is the "
            "combination that reads as safety and is not. Restores are now exercised on a "
            "schedule against a scratch instance, and the retention window is derived from how "
            "long a restore actually takes rather than from a round number of weeks.\n"
        ),
        query="how often is the full dump taken and how long is it kept",
    ),
)

#: Deliberately absent from BOTH the authored pairs and the repository docs the probe indexes
#: alongside them. `abstention_accuracy` is measured over these, and the record predicts it must
#: not fall: expansion fires on a supersession edge, and a query with no superseded hit has none.
UNANSWERABLE: tuple[str, ...] = (
    "what is the office wifi password",
    "who won the interdepartmental five a side final",
    "what colour was the original logo before the rebrand",
    "how many parking spaces does the building have",
    "which caterer supplies the friday lunch",
    "what is the fire drill assembly point",
)


def documents() -> dict[str, str]:
    """`{filename: markdown}` for the authored corpus, successors carrying their edge."""
    out: dict[str, str] = {}
    for pair in PAIRS:
        out[f"{pair.slug}_v1.md"] = pair.v1
        out[f"{pair.slug}_v2.md"] = f"---\nsupersedes: {pair.slug}_v1.md\n---\n{pair.v2}"
    return out
