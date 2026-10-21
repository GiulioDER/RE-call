# AML Multimodal Full 2: version and method disclosure

The version registered for the second Multimodal Full run of the Agent Memory Leaderboard, Cycle 2,
Open-source Methods, and the changes since the first run. The same text is entered on the
platform when the version is registered.

## Version

| | |
|---|---|
| Name | RE-call C9 625b7138 |
| Repository | https://github.com/GiulioDER/RE-call |
| Fixed commit | `625b71381c2f5c7de6d34c8dc5b9f45e08370b58` |
| Tag | `aml-c2-full2-625b7138` (signed) |

`GET /version` on the endpoint reports the running commit as `git_commit`.

## Models

- Embedders: voyage-multimodal-3.5 for images; voyage-context-4 and voyage-code-4 for text. These
  are the same models as in the first Multimodal Full.
- Reranker: Voyage rerank-3 reorders the text candidates of each Search; image hits are fused after
  it.
- Generative model: none is called. The Add compiler (gpt-4o-mini) is off in this version, so no
  generative model runs during Add, Search or service start.

## Configuration

Variant `C9_routed_specialists_grounded_graph_atomic`, with `RECALL_AML_ROUTE_GATES=data`,
`RECALL_AML_T1_GATE=content`, `RECALL_AML_T1_RENDER=v2`, `RECALL_AML_COMPILER=0`,
`RECALL_AML_RERANKER=1`, `RECALL_AML_RERANK_MODEL=rerank-3` and `RECALL_AML_FEEDBACK_GUARD=1`. Every
other setting is the default at that commit.

The first run served variant `C8_routed_specialists_grounded_graph` at commit `5d82b516` (tag
`aml-c2-full1-multimodal`), with the compiler on and no reranker.

## Changes compared with the first run

1. Text memories are returned beside images on the image route. The first run dropped them there.
2. Every returned item carries its date; image items carry the timestamp of their message. The first
   run returned them undated.
3. A relative time phrase keeps its original words, and the date it refers to is added beside it in
   brackets.
4. Image search runs when the user's memory holds images, not when the question contains a visual
   word. An image found only by image search is returned as the image itself.
5. Text candidates are reordered by Voyage rerank-3; a query that is only an image keeps its fused
   order.
6. Atomic views are built inside each Add, so atomic rescue acts on every Search. In the first run it
   found no views and did not act.
7. A stored record of an earlier answer that reports a failure, about a question the same user was
   already asked, is not kept; a record that also carries the reference answer is kept. Every other
   message is stored unchanged.
8. Each Search has a 90 second time limit. If the reranker or the image search runs out of time, it
   is skipped, and a question with text is answered from text alone. If the query cannot be embedded
   in time, the service answers HTTP 503 with Retry-After instead of running past the connection
   limit.
9. An image that cannot be read is skipped during Add; the Add is refused only when nothing in it can
   be read.
10. Stability: each user's writes are applied one at a time, and Voyage requests stay within the
    provider's limits.

## Deployment

The same endpoint as registered on the platform for Coding Full 1 (`POST /v1/add`,
`POST /v1/search`, `POST /v1/delete`; `GET /health`, `GET /version`), self-hosted behind a
Cloudflare Tunnel. Authentication: `Authorization: Bearer <key>`. Capacity: one service process with
a 12 GB memory limit; declared concurrency Add 16 and Search 16. The full API contract is in
[RECALL_HOSTED_INTEGRATION.md](RECALL_HOSTED_INTEGRATION.md).

## Attribution

RE-call is my own work. The rule in change 7 follows the record shapes of the public
AgentMemoryBench protocol ([solomoon313/AgentMemoryBench](https://github.com/solomoon313/AgentMemoryBench),
MIT) and the findings of StreamBench (arXiv:2406.08747) and Evo-Memory (arXiv:2511.20857) that
showing a model its failed attempts lowers accuracy; no code from them is used. Components,
references and licences are listed in the method paper submitted with the materials.

## Licence

The commit carries the PolyForm Noncommercial License 1.0.0 in its tree;
[LICENSE_GRANT_AGPL.md](../LICENSE_GRANT_AGPL.md) licenses it under AGPL-3.0-only by its full hash.
