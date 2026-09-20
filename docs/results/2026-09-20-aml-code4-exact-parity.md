# AML Code4 exact parity validation

Date measured: 2026-09-20

Status: local and public prelaunch gates passed. No official AML Smoke or Full job was started.

## Candidate identity

1. Candidate commit: `ad7bdcf7090504f90cb7baa0b8d33620cd084d5e`.
2. Variant: `C6_code4_exact_bm25`.
3. Generation: `aml-code4-exact-bm25-v1`.
4. Public endpoint: `https://memory.pred-markets.com`.
5. Release wheel SHA256: `72076f6df6d4f7b5ae7db664a93c6cf9deb26b94b32cfd18fbf72feeadf34154`.
6. Release manifest SHA256: `927963f43ea65ec9701083638a8e7edd34a7f910b1194603ba50b58b07c863d8`.

The previous C5 process remains active on port 18006 as the rollback target. The public tunnel now
routes the memory endpoint to C6 on port 18007. The tunnel configuration backup is
`/home/sentiment/.cloudflared/config.yml.pre-c6-exact-ad7bdcf7`.

## Retrieval parity

The committed verifier is in Agent Memory Benchmark commit
`a2577df29083be87967ec32b3bfaef5e848b9c10`. The immutable result and report were committed in
`04ebb20819f598007c255baee29ee12a5faebad5`.

1. Corpus manifest SHA256 was `58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814`.
2. The corpus contained 196 sessions and 1,220 windows.
3. All 34 hosted Top 100 rankings exactly matched the independent NumPy plus canonical BM25
   reference, including order.
4. Mean and minimum Top 100 overlap were 100. Mean Jaccard was 1.0.
5. Hosted recall at 10 and recall at 100 were both 1.0.
6. Hosted and reference MRR were both 0.8611111111. The earlier screen MRR was 0.8464052288.
7. Thirty three historical first relevant ranks were unchanged. `ts-schema-additive` improved from
   rank 2 to rank 1.
8. Add p50 was 330.15 ms and p95 was 555.00 ms with three workers.
9. Hosted Search p50 was 431.11 ms and p95 was 1073.16 ms.
10. Independent NumPy exact dense p50 was 7.81 ms and p95 was 72.10 ms.

Two windows contained decoded NUL characters. PostgreSQL cannot store NUL in text, so the frozen
hosted contract replaced those characters with the visible NUL symbol. A preflight compared the
candidate chunk builder against the replay corpus and matched all 1,220 hosted chunk identities.
The first parity attempt identified this boundary and deleted its tenant. The corrected replay was
committed before measurement and passed.

## Database and service gates

1. A database fixture inserted 101 identical 1,024 dimension vectors in reverse order. Exact top
   100 returned stable source session ordinals 0 through 99.
2. The local hosted contract passed health, version, Add, immediate Search, identical replay,
   conflicting replay, cross chunk retrieval, peer Add, tenant isolation, Top K, and deletion.
3. Restart persistence passed. A marker added before an explicit service restart was retrievable
   after restart and then deleted.
4. The service remained active with `NRestarts=0`.
5. A local 16 Add client plus 16 Search client cycle completed with zero errors. Add p95 was
   1766.65 ms and Search p95 was 1425.48 ms.
6. The public HTTPS contract passed every check after the C6 route cutover.
7. A public 16 Add client plus 16 Search client cycle completed with zero errors. Add p95 was
   1580.47 ms and Search p95 was 1420.23 ms.
8. The final C6 table check reported zero rows and zero tenants.

## Remaining official gate

The live Coding Smoke worker was available when last checked, while the Full worker was not. The
official Smoke remains intentionally unstarted. It requires the user's explicit approval before
any evaluation creation request is sent.
