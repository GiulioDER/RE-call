# DRAFT, not published. For review before anything goes out.

# I ran the neutral memory benchmark on my memory system. Here is everything, signed.

Bench'd (benchd.ai) calls itself the neutral benchmark for AI memory systems. It runs
LongMemEval (500 questions) and LoCoMo (1,540 questions) through one open source harness, with
a locked answerer and judge (gpt-4o-mini, temperature 0), and publishes cryptographically
signed result manifests. Its verified leaderboard currently tops out with Graphiti at 65.0,
LlamaIndex and LangChain at 59.0, and a no-memory GPT-4o-mini baseline at 57.6.

On 2026-08-23 I ran RE-call, my open source memory layer, through that harness end to end:
the full 500 and the full 1,540, official judge, official prompts, no modifications to the
scoring path. The results, under the leaderboard's own statistic:

| Benchmark | RE-call | Best verified score on the board |
|---|---|---|
| LongMemEval 500 | **69.0** | 65.0 (Graphiti) |
| LoCoMo 1,540 | **71.6** | 54.8 (LlamaIndex, in the repo's manifests; the public board shows no LoCoMo column at all) |

Both manifests are Ed25519 signed, carry all 2,040 full traces, and verify with the harness's
own `benchd verify`. They are published here, with run records pinning every commit hash,
dataset hash and configuration knob:
https://github.com/GiulioDER/RE-call/releases/tag/benchd-official-2026-08-23

The whole campaign, tuning included, cost $6.64 in API spend, measured by the provider's
meter, not estimated.

## How the score happened

RE-call's retrieval was already at leaderboard level before any tuning: the baseline
configuration matched the leader's retrieval hit rate (48% against 45%) on my tuning slice.
The benchmark is not retrieval limited, it is conversion limited: the locked answerer is
strict, and handing it a page of raw conversation transcript makes it answer "insufficient
information", which the locked judge scores as wrong.

So the winning configuration makes the memory layer answer like a memory, not like a search
engine: hybrid dense plus lexical retrieval (voyage-4 embeddings), Voyage rerank-2.5, and a
synthesis step in which DeepSeek v4 pro distills the retrieved chunks into a two sentence
evidence digest, which is what the answerer sees. Mean recall output: 33 to 65 tokens per
question, which also makes it one of the most token efficient systems on the board.

Two tuning findings worth stealing. First, extended thinking hurt: the reasoning traces made
the digests hedge, and hedged digests get scored as abstention, a measured 8.3 point penalty.
Second, abstention is a forfeit here: every question is answerable by construction and the
judge scores "insufficient information" as incorrect, so I set the abstention threshold to
zero and say so openly. At RE-call's default threshold the run loses 26.7 points to honest
abstention. Benchmarks reward different behaviour than production does, and pretending
otherwise would just be a quieter way of misreporting.

Everything above was pre-registered before measurement, with predictions frozen in git and
the gaps recorded after, including the one that stung: my first full LongMemEval attempt
crashed on a connection limit my small tuning runs could never have surfaced. That run is
voided in the record, the manifest kept, the bug fixed and proven at double the failing scale
before the rerun. The pre-registration files, with every wrong prediction left standing, are
in the repository under `docs/preregistrations/`.

## What I found on the leaderboard while doing this

I checked the verified board against the harness's own aggregation rule and against the
signed manifests shipped in its repository. Three of the top four rows fail that check, and I
filed the details as an issue rather than a thread of accusations:
https://github.com/benchdai/harness/issues/5

In short: one row shows an overall of 80.0 whose own dimension scores average 26.7, one shows
60.0 from dimensions that are all zero, and the row at 100.0 combines perfect recall with a
reliability score of 4.0, the signature of an adapter that echoes expected answers, and none
of the three has a published manifest, although all three wear the Community-Verified badge
and the site states that every verified score is publicly verifiable. The rows I compare
against above, Graphiti and LlamaIndex, are the ones whose manifests exist and whose
arithmetic checks out.

## Why this is not on their leaderboard yet

Two reasons, both documented in the open. First, their submission endpoint caps uploads at
about 4.5 MB, and no full trace manifest fits through it, including their own 16 MB ones, so
I published the manifests myself and asked how they want to receive large files:
https://github.com/benchdai/harness/issues/4

Second, the path onto the trusted tier as a vendor is paid: verification starts at $299 per
month on their pricing page. I declined. A trust badge I paid for would be worth exactly what
a competitor would say it is worth. Instead, everything needed to verify my numbers is public
and free: the signed manifests, the adapter (one file), the run script that pins every hash,
and a one command reproduction. I would welcome a community verified rerun by the Bench'd
team, at their convenience, on their infrastructure, and the standing offer is in the issue.

## Verify it yourself

```
pip install benchd-harness
benchd verify longmemeval-v1.manifest.signed.json
benchd verify locomo-v1.manifest.signed.json
```

Signing key fingerprint: 92dae5232b5c8af6. Adapter, patches, tests and run script:
`benchmarks/benchd/` in the RE-call repository. The only harness change I made is an optional
`--workers` flag for concurrency, published as a patch, with the per item pipeline untouched
and score stability across worker counts measured and recorded.

RE-call is open source, Apache 2.0, on PyPI as `recall-rag`.
