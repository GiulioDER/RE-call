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

#: Twenty more pairs, authored for the fourth record
#: (`docs/preregistrations/2026-08-20-successor-ordering-regression.md`) to take the set from 10 to
#: 30 and the absent stratum from 6 queries to something whose interval can be read. Kept in a
#: separate tuple rather than merged into the one above so provenance stays visible: the first ten
#: were written before three measurements, these twenty after, and a reader can tell which is which.
_ADDED_2026_08_20: tuple[Pair, ...] = (
    Pair(
        slug="log_retention",
        v1=(
            "# Log retention\n\n"
            "All application logs are kept for 90 days in one bucket, uncompressed. A single "
            "retention rule is easy to explain and easy to audit. Status: adopted.\n"
        ),
        v2=(
            "# Retention now follows what a log is for\n\n"
            "One rule for every class meant paying archival prices to store debug chatter while "
            "the security trail expired on the same schedule as it. Errors and audit events are "
            "kept for a year, ordinary request logs for two weeks.\n"
        ),
        query="how long are application logs kept and in what bucket",
    ),
    Pair(
        slug="on_call_rotation",
        v1=(
            "# On call rotation\n\n"
            "On call is a weekly rotation, handed over on Monday morning in the team channel. "
            "Status: adopted.\n"
        ),
        v2=(
            "# Handover is an artifact, not a message\n\n"
            "A chat message is not a handover. The incoming engineer inherited open incidents "
            "nobody had written down, and found out during the next page. A shift now ends by "
            "recording what is open, what was already tried, and what is expected overnight.\n"
        ),
        query="when does the weekly on call rotation hand over",
    ),
    Pair(
        slug="feature_flags",
        v1=(
            "# Feature flag lifecycle\n\n"
            "Flags are deleted once the feature has launched. Cleanup is tracked as an item on "
            "the launch checklist. Status: adopted.\n"
        ),
        v2=(
            "# Flags now expire on their own\n\n"
            "Cleanup tracked on a checklist is cleanup that happens when somebody remembers, and "
            "the oldest flag in the codebase had outlived the team that added it. Every flag "
            "carries an owner and an expiry date where it is defined, and the build fails when "
            "one passes its date.\n"
        ),
        query="when are feature flags deleted after a launch",
    ),
    Pair(
        slug="api_versioning",
        v1=(
            "# API versioning\n\n"
            "The public API is versioned in the URL path, so a breaking change ships as `/v2/` "
            "alongside `/v1/`. Status: adopted.\n"
        ),
        v2=(
            "# Additive change only, and no second version\n\n"
            "Running two versions doubled the surface under test and the older one never retired, "
            "because nothing forces a client to move. Changes are additive: fields may be added, "
            "never removed or repurposed, and a genuinely incompatible change gets a new resource "
            "rather than a new version of an old one.\n"
        ),
        query="how is the public API versioned in the URL path",
    ),
    Pair(
        slug="secret_storage",
        v1=(
            "# Deployment secrets\n\n"
            "Secrets are stored as environment variables in the CI project settings, where only "
            "maintainers can read them. Status: adopted.\n"
        ),
        v2=(
            "# Credentials are short lived and issued on demand\n\n"
            "A durable secret in a settings page has no expiry and no record of who used it. Jobs "
            "now request a token scoped to the task at the moment they start, valid for minutes, "
            "and nothing lasting is kept in the build configuration.\n"
        ),
        query="where are deployment secrets stored for CI",
    ),
    Pair(
        slug="deploy_strategy",
        v1=(
            "# Release strategy\n\n"
            "Releases go out blue green. The new version is brought up alongside the old, traffic "
            "switches in one step, and the previous version stays warm for rollback. "
            "Status: adopted.\n"
        ),
        v2=(
            "# Rollout is progressive and driven by signal\n\n"
            "An instant switch means every user meets a regression at the same moment, and warm "
            "standby paid for capacity that sat idle between releases. Traffic now moves in "
            "cohorts with health gates between them, and a failing gate halts the rollout without "
            "waiting for somebody to decide.\n"
        ),
        query="how do releases go out and how long does the old version stay warm",
    ),
    Pair(
        slug="error_budget",
        v1=(
            "# Availability target\n\n"
            "The service targets 99.9% uptime per month. Missing it is reviewed at the monthly "
            "operations meeting. Status: adopted.\n"
        ),
        v2=(
            "# The budget governs releases, not meetings\n\n"
            "A target reviewed after the fact changes nothing while it is being spent. Remaining "
            "budget is computed continuously, and once it is exhausted feature releases stop "
            "until it recovers, which turns the target from a report into a control.\n"
        ),
        query="what monthly uptime does the service target and where is it reviewed",
    ),
    Pair(
        slug="test_data",
        v1=(
            "# Test data\n\n"
            "Integration tests run against a weekly snapshot of the production database, restored "
            "into a dedicated test instance. Status: adopted.\n"
        ),
        v2=(
            "# Tests no longer run on copies of real records\n\n"
            "A restored snapshot is production data with a different hostname, and it had already "
            "spread to laptops and CI caches by the time anyone counted. Fixtures are generated "
            "to the same shapes and distributions, with no record traceable to a person.\n"
        ),
        query="where does the integration test database snapshot come from",
    ),
    Pair(
        slug="dependency_updates",
        v1=(
            "# Dependency updates\n\n"
            "Dependencies are reviewed and bumped manually once a month, by whoever holds "
            "maintenance duty that week. Status: adopted.\n"
        ),
        v2=(
            "# Updates arrive continuously and the suite decides\n\n"
            "A monthly pass batches a month of change into one risky merge, and a security fix "
            "waits for the calendar. Updates now open one at a time and merge on a green suite, "
            "with major versions held back for a human.\n"
        ),
        query="how often are dependencies reviewed and bumped manually",
    ),
    Pair(
        slug="code_review",
        v1=(
            "# Review policy\n\n"
            "Every change needs two approvals before it can merge, regardless of size. "
            "Status: adopted.\n"
        ),
        v2=(
            "# Review depth follows risk\n\n"
            "A flat two approval rule taught reviewers to skim, because most changes do not "
            "warrant two careful reads and everything queued behind the few that did. Changes are "
            "classified by merge path and blast radius, and only the high risk classes need a "
            "second reviewer.\n"
        ),
        query="how many approvals does every change need before merge",
    ),
    Pair(
        slug="incident_severity",
        v1=(
            "# Incident severity\n\n"
            "Incidents are graded one to three by the number of affected users, decided by the "
            "responder on the page. Status: adopted.\n"
        ),
        v2=(
            "# Severity follows the symptom, not a headcount\n\n"
            "The affected user count is unknown at the moment severity has to be declared, so "
            "responders guessed low and escalated late. Severity is now set from what a customer "
            "can observe, which is knowable in the first minute.\n"
        ),
        query="how are incidents graded one to three by affected users",
    ),
    Pair(
        slug="metrics_cardinality",
        v1=(
            "# Metric dimensions\n\n"
            "Metrics carry as many tags as a team finds useful. More dimensions make debugging "
            "easier. Status: adopted.\n"
        ),
        v2=(
            "# Dimensions have a budget, enforced at ingest\n\n"
            "Unbounded tags turned a request identifier into a metric series, and the store spent "
            "most of its memory on labels nobody ever queried. Each metric declares its "
            "dimensions and ingest rejects series outside that declaration.\n"
        ),
        query="how many tags can metrics carry and who decides",
    ),
    Pair(
        slug="cache_invalidation",
        v1=(
            "# Cache invalidation\n\n"
            "Cached entries expire on a TTL. Nothing else invalidates them, which keeps the write "
            "path simple. Status: adopted.\n"
        ),
        v2=(
            "# Writes invalidate directly, with TTL as a backstop\n\n"
            "A TTL guarantees only that the data is wrong for no longer than the TTL, which is "
            "not a guarantee anyone wanted to pay for. The write path publishes an invalidation, "
            "and the expiry remains only to catch what the event path misses.\n"
        ),
        query="what expires cached entries besides the TTL",
    ),
    Pair(
        slug="queue_backpressure",
        v1=(
            "# Work queue\n\n"
            "The work queue is unbounded, so a burst is absorbed rather than rejected. "
            "Status: adopted.\n"
        ),
        v2=(
            "# The queue is bounded and sheds load deliberately\n\n"
            "An unbounded queue does not absorb a burst, it conceals one: latency grew without "
            "limit while every dashboard showed a healthy consumer. Depth is capped, and work "
            "past the cap is refused at the edge with a retriable response.\n"
        ),
        query="is the work queue unbounded and what happens during a burst",
    ),
    Pair(
        slug="schema_migrations",
        v1=(
            "# Schema migrations\n\n"
            "Migrations run inside a scheduled downtime window on Sunday night. "
            "Status: adopted.\n"
        ),
        v2=(
            "# Migrations no longer need a window\n\n"
            "Expand then contract: add the new shape, write to both, backfill, move reads, drop "
            "the old shape. Every step is safe with both versions of the code running, so no "
            "window is needed and the change can be abandoned at any point.\n"
        ),
        query="when do schema migrations run and how long is the downtime window",
    ),
    Pair(
        slug="rate_limit",
        v1=(
            "# Rate limiting\n\n"
            "Requests are rate limited per source IP address. Status: adopted.\n"
        ),
        v2=(
            "# Limits attach to the principal\n\n"
            "Limiting by address punished everyone behind one office egress while letting a "
            "single credential spread itself across a cloud range. The limit attaches to the "
            "authenticated principal, which is the thing whose usage the limit is about.\n"
        ),
        query="what are requests rate limited by",
    ),
    Pair(
        slug="audit_log",
        v1=(
            "# Audit events\n\n"
            "Audit events are written to the standard application log and picked out later by "
            "filtering on a prefix. Status: adopted.\n"
        ),
        v2=(
            "# Audit events go to their own append only store\n\n"
            "A trail inside the application log is a trail anybody with log access can rotate "
            "away, and needing a filter to find it is an admission that it is not a trail. Events "
            "go to a separate store with no delete path.\n"
        ),
        query="where are audit events written and how are they found later",
    ),
    Pair(
        slug="pii_redaction",
        v1=(
            "# Personal data in responses\n\n"
            "Personal data is stripped from responses by a set of regular expressions applied on "
            "the way out. Status: adopted.\n"
        ),
        v2=(
            "# Classified at ingest, never stored raw\n\n"
            "Redacting on the way out means the data was stored, so every miss was already "
            "durable and every new field started unprotected. Fields are classified when they "
            "arrive, sensitive ones are stored tokenised, and the response path has nothing left "
            "to strip.\n"
        ),
        query="how is personal data stripped from responses",
    ),
    Pair(
        slug="embedding_cache",
        v1=(
            "# Embedding cost\n\n"
            "Every indexing run embeds every chunk. There is no cache, which keeps the pipeline "
            "easy to reason about. Status: adopted.\n"
        ),
        v2=(
            "# Vectors are content addressed and reused\n\n"
            "Re embedding unchanged text was most of the cost of every run, and it made "
            "re indexing expensive enough that people avoided doing it. Vectors are keyed by the "
            "hash of the text together with the model identity, so a model change invalidates "
            "correctly while unchanged text costs nothing.\n"
        ),
        query="does every indexing run embed every chunk",
    ),
    Pair(
        slug="doc_ownership",
        v1=(
            "# Documentation ownership\n\n"
            "Documentation is owned by the team collectively. Anyone may edit any page. "
            "Status: adopted.\n"
        ),
        v2=(
            "# Each page names an owner and a review date\n\n"
            "Collective ownership meant no page was anybody's problem, and the oldest pages were "
            "the most confidently wrong. Every page names one owner and a date by which it must "
            "be read again, and an overdue page is marked where readers can see it.\n"
        ),
        query="who owns the documentation and who can edit pages",
    ),
)

PAIRS = PAIRS + _ADDED_2026_08_20


#: A regression case: a query whose gold answer is a THIRD document, worded so that a superseded
#: document is also retrieved. An ordering that promotes successors unconditionally displaces
#: `gold` here, which is the cost the first three records could not see because their fixture
#: contained only queries the successor was supposed to win.
@dataclass(frozen=True)
class Regression:
    slug: str
    doc: str
    query: str
    #: The superseded document expected to be dragged in by shared vocabulary. Recorded so a
    #: failure to retrieve it is reported as an excluded case rather than a passing one.
    expects_stale: str


REGRESSIONS: tuple[Regression, ...] = (
    Regression(
        slug="store_timeout",
        doc=(
            "# Store statement timeout\n\n"
            "A single store query is capped at five seconds by a statement timeout. Past that the "
            "query is cancelled and the caller is told it timed out.\n"
        ),
        query="how many seconds before a single store query is cancelled",
        expects_stale="retry_budget_v1.md",
    ),
    Regression(
        slug="log_shipping",
        doc=(
            "# Log shipping\n\n"
            "Logs reach the aggregator through a sidecar that batches for two seconds and retries "
            "with backoff if the aggregator is unavailable.\n"
        ),
        query="how do application logs reach the aggregator",
        expects_stale="log_retention_v1.md",
    ),
    Regression(
        slug="flag_naming",
        doc=(
            "# Feature flag naming\n\n"
            "Flag names are prefixed with the owning team and written in lower snake case, so the "
            "owner is readable from the name alone.\n"
        ),
        query="what naming convention do feature flags follow",
        expects_stale="feature_flags_v1.md",
    ),
    Regression(
        slug="api_pagination",
        doc=(
            "# Pagination\n\n"
            "List endpoints paginate with an opaque cursor. Page numbers are not supported, "
            "because a page number is wrong as soon as the underlying set changes.\n"
        ),
        query="how do list endpoints paginate",
        expects_stale="api_versioning_v1.md",
    ),
    Regression(
        slug="backup_encryption",
        doc=(
            "# Backup encryption\n\n"
            "Backups are encrypted at rest with a key held in a separate account, so possession "
            "of the backup is not possession of the data.\n"
        ),
        query="how are backups encrypted at rest and where is the key",
        expects_stale="backup_schedule_v1.md",
    ),
    Regression(
        slug="deploy_notification",
        doc=(
            "# Release announcements\n\n"
            "A release posts a summary to the release channel naming the change, the operator who "
            "started it, and the cohort it is currently serving.\n"
        ),
        query="who is named in the summary a release posts",
        expects_stale="deploy_strategy_v1.md",
    ),
    Regression(
        slug="slo_dashboard",
        doc=(
            "# Uptime dashboard\n\n"
            "The uptime dashboard refreshes every minute and is readable by anyone in the "
            "company without a request.\n"
        ),
        query="how often does the uptime dashboard refresh",
        expects_stale="error_budget_v1.md",
    ),
    Regression(
        slug="test_parallelism",
        doc=(
            "# Integration suite parallelism\n\n"
            "The integration suite runs eight workers in parallel, each against its own database, "
            "so no two workers can see each other's rows.\n"
        ),
        query="how many workers does the integration suite run in parallel",
        expects_stale="test_data_v1.md",
    ),
    Regression(
        slug="review_sla",
        doc=(
            "# Review response time\n\n"
            "A review request is expected to receive a first response within one working day, "
            "even if that response is only to say when a full read will happen.\n"
        ),
        query="how quickly should a review request get a first response",
        expects_stale="code_review_v1.md",
    ),
    Regression(
        slug="queue_metrics",
        doc=(
            "# Queue observability\n\n"
            "Queue depth and the age of the oldest item are reported every ten seconds, because "
            "depth alone cannot distinguish a fast queue from a stalled one.\n"
        ),
        query="what queue metrics are reported and how often",
        expects_stale="queue_backpressure_v1.md",
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


#: The labelled set the in-run calibration is fitted from. Added for the second measurement,
#: registered in `docs/preregistrations/2026-08-20-successor-expansion-recalibrated.md`.
#:
#: The first run fitted its threshold from the ten `Pair.query` strings and the six controls below.
#: That was wrong twice: 10 and 6 against a stated minimum of 20 per class
#: (`recall/calibration.py:44`), and, worse, it fitted on the very queries being scored. Those are
#: worded from v1 and match v1 strongly, so the answerable distribution was the top of the study's
#: own score range and the threshold inherited it.
#:
#: Both sets below are therefore DISJOINT from everything the probe measures: no `Pair.query`, and
#: none of the six `UNANSWERABLE` controls.
CALIBRATION_ANSWERABLE: tuple[str, ...] = (
    "how does RE-call authenticate its MCP HTTP transports",
    "what is a generation bound calibration",
    "how are database migrations versioned and verified",
    "what does the installation wizard ask for",
    "how is row level security tested against a superuser",
    "what does the MTRAG benchmark measure",
    "how does the reasoning graph projection work",
    "what is an immutable index generation",
    "what licences do the bundled models carry",
    "how does the extraction cache persist parsed documents",
    "what is the security model for tenant isolation",
    "how does document expansion build an evidence bundle",
    "what does the environment reference say about configuration variables",
    "what is the dependency policy for this project",
    "how does as of retrieval use a reference time",
    "what does the case study say about where RE-call came from",
    "how do I serve a corpus that has no calibration yet",
    "what does the repository map list",
    "how is truth extraction reviewed before a rewrite",
    "what is the research protocol for this project",
    "what operating modes does RE-call support",
    "how does the inference proposal protocol work",
    "what does the production posture say is not shipped",
    "how do I use RE-call with Claude over MCP",
)

#: Genuinely off topic, and off topic in the same register as a real question. NOT an answerable
#: query with a nonsense token appended, which is the defect `results/FINDINGS.md:370` records as
#: leaving the two classes not separable at all.
CALIBRATION_UNANSWERABLE: tuple[str, ...] = (
    "how do I claim mileage for a client visit",
    "what time does the canteen stop serving breakfast",
    "who is the first aider on the second floor",
    "what is the dress code for the summer party",
    "which airline do we have a corporate account with",
    "how many days of carry over holiday are allowed",
    "what is the procedure for booking a meeting room",
    "who maintains the coffee machine",
    "what is the guest wifi network called",
    "how do I report a broken chair",
    "which bank do we use for expenses",
    "what is the notice period for a desk move",
    "who organises the christmas party",
    "what is the postcode of the registered office",
    "how do I get a replacement door badge",
    "what is the recycling collection day",
    "which gym does the corporate membership cover",
    "what is the phone number for the reception desk",
    "what is the process for ordering business cards",
    "how do I book the electric car charging bay",
    "which shredding company collects confidential waste",
    "what is the visitor sign in procedure",
)


def documents() -> dict[str, str]:
    """`{filename: markdown}` for the authored corpus, successors carrying their edge.

    Includes the regression documents, which carry NO supersession edge of their own. They are the
    gold answers that an unconditional promote-to-first would displace, and the reason the fourth
    record can report a cost the first three could not see.
    """
    out: dict[str, str] = {}
    for pair in PAIRS:
        out[f"{pair.slug}_v1.md"] = pair.v1
        out[f"{pair.slug}_v2.md"] = f"---\nsupersedes: {pair.slug}_v1.md\n---\n{pair.v2}"
    for regression in REGRESSIONS:
        out[f"{regression.slug}.md"] = regression.doc
    return out
