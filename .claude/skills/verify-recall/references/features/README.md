# RE-call verification map

What a user can do with RE-call, how to reach it, how to drive it with `control_recall.py`, and what
end state proves it. Each file is named after a feature id in `docs/feature_map.toml`; that map says
where the code lives, this one says how to prove it works.

## Baseline preconditions

- A run from `launch` on this checkout, and `doctor --run <run>` healthy before the first drive.
- The seed is `recall/eval/corpus`: 22 memos, among them two supersession pairs
  (`cache_ttl_v2.md` replaces `cache_ttl_v1.md`, `rollout_flag_v2.md` replaces `rollout_flag_v1.md`).
- Embedder `hashing` unless the feature says otherwise, so strict search refuses. Use
  `--development` for ranking and supersession; never to prove trust.
- Start each recipe from a fresh run when it depends on the seed being untouched.

## Proof and skip reporting

- Record the run id and the evidence files with every claim.
- Show the action and a read-only second view of the result.
- A path that cannot be reached here (a certified answer, PostgreSQL, a hosted embedder) is reported
  as unreachable with its prerequisite, never replaced by a weaker drive that passes.

## Features

- [indexing](indexing.md): `recall index` a folder into a lite store.
- [search](search.md): `recall search` and the strict trust gate's refusal codes.
- [supersession](supersession.md): a superseded memo is marked and points at its replacement.
- [lite](lite.md): `recall lite refresh` keeps the store current with its memo folder.
- [forget](forget.md): `recall forget` previews, deletes, and what a refresh does afterwards.
- [mcp-server](mcp-server.md): the MCP server over stdio, its tools, and `recall_search`.
- [doctor](doctor.md): `recall doctor --json` on a lite store.
