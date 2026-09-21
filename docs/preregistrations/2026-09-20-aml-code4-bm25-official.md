# AML CAMBench Coding official candidate preregistration

Date frozen: 2026-09-20

Status: preregistered before hosted deployment, Smoke, or Full measurement

## Decision and scope

The only candidate licensed for the next CAMBench Coding official run is
`C5_code4_bm25`. It is a direct replacement candidate derived from the successful Voyage Code4
screen. Context4 direct replacement and the two protected Context4 fusion variants are closed for
coding. They must not be added to this candidate.

The intended leaderboard division is Commercial Products. This avoids treating the unresolved
organizer question about external embedding and reranking providers in the Open source Methods
division as answered. A change of division requires a new preregistration before an official run.

## Frozen Add path

1. Normalize PostgreSQL NUL characters with the existing hosted contract.
2. Render every session message in request order as optional timestamp, role, and content.
3. Join the rendered messages into one session text.
4. Split raw whitespace separated words into windows of 160 words with stride 120.
5. Embed every window with registered profile `voyage-code-4-v1`, model `voyage-code-4`, width
   1024, using document input semantics.
6. Persist raw windows only.

The candidate does not run a compiler, facet generator, reranker, SPLADE encoder, code neighbor
expansion, graph sidecar, evidence packer, or multimodal embedding path.

## Frozen Search path

1. Embed the original query once with `voyage-code-4-v1` using query input semantics.
2. Retrieve the top 100 dense windows.
3. Rank all windows for the same user with `canonical-bm25-k1-1.5-b0.75-v1`.
4. Keep the top 100 positive BM25 windows.
5. Fuse dense and BM25 ranks with unweighted reciprocal rank fusion, constant 60.
6. Break fused score ties by chunk identifier and return at most the requested top K, with the AML
   maximum fixed at 100.

The BM25 tokenizer is lowercase regex `[a-z][a-z0-9_]*`, requires tokens longer than one
character, removes the short stopword set frozen in `recall_aml/code4.py`, retains repeated query
terms, and uses Okapi parameters K1 1.5 and B 0.75.

## Official execution settings

Set Max add concurrency to 3 and Search concurrency to 3 on the AML Evaluation page. Use the
platform fixed Search Top K of 100. Do not reinterpret three workers as three service processes.
The hosted application remains one process and handles request concurrency asynchronously.

## Evidence that licensed this candidate

The paired Code4 Task Solve replay admitted 100 pairs and recorded 67 successful Code4 attempts
against 59 successful Code3 attempts, a net gain of 8. Its source retrieval check recovered all
34 queries at recall at 10 and increased MRR from 0.5063 to 0.8464. All frozen promotion gates
passed. These measurements license the candidate shape; they are not an official AML score.

The source experiment used the same 160 word windows, stride 120, dense top 100, BM25 top 100,
unweighted reciprocal rank fusion at 60, and result width 100. This hosted candidate intentionally
does not import the unrelated Context4, graph, compiler, or reranker treatments from the currently
running endpoint.

## Mechanical release gates

Do not start Full unless all items below are true for the exact committed release:

1. Focused Code4 contracts pass, including profile identity, windowing, BM25, sparse bypass, and
   release manifest identity.
2. The hosted contract suite passes with three test workers.
3. The built wheel and clean commit are bound by a release manifest for `C5_code4_bm25`.
4. The target database generation is empty before the platform run.
5. `/version` reports the bound commit, variant `C5_code4_bm25`, profile
   `voyage-code-4-v1`, BM25 profile, window size 160, and stride 120.
6. Health, isolation, idempotency, deletion, restart persistence, and external HTTPS checks pass.
7. AML Smoke passes against the bound public version.

If Smoke fails, stop. Do not patch the frozen deployment in place and continue. Diagnose the
failure, create a new preregistration and release identity, then repeat the gates.

## Full run and reporting rule

Use one Full run for CAMBench Coding only after all release gates pass. Record the AML job identity,
bound version identity, start time, completion state, and platform result without editing this
preregistration. The second Full allowance remains unspent unless a separately preregistered
candidate earns it.
