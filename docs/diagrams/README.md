# Interactive diagrams

Two self-contained HTML diagrams of how RE-call works, generated with
[Archify](https://github.com/tt-a1i/archify). Each box carries links to the exact source lines it
describes, so a reader can check every claim against the code.

| Diagram | Shows |
|---|---|
| [recall-overview.html](recall-overview.html) | The system overview: entry points, the generation build, trusted retrieval and the trust gate. |
| [recall-search.html](recall-search.html) | One `recall_search` call from start to finish, with its three outcomes: refuse, abstain or answer. |

GitHub shows an HTML file as source rather than as a page. To explore a diagram, download it and
open it in a browser. It needs no install and works offline, apart from the source links. Each
page has light and dark themes, path tracing, and export to PNG and SVG.

## What they describe

Both diagrams are pinned to commit `0d5d407b731e62e3929589bdc30b1a60fc46cf4b`. Their source links
point at that commit, so they keep resolving after master moves, and they describe the code as it
was then. When the code moves on, regenerate them rather than editing the HTML.

The overview follows the flowchart in the [README](../../README.md#how-it-works), with every box
and arrow checked against the code. It leaves out the opt in reasoning, fact application and
ingest paths, and a card on the page summarises them instead.

The sequence diagram folds PostgreSQL into the generation store column, whose source links cite
the SQL, and draws the optional reranker as a note rather than as a step.

## Regenerating

The JSON beside each page is the editable source. Archify validates it, renders the page, and
checks every cited path and line range against the git objects at the pinned commit. With
Archify 3.0.1 installed (`npx skills add tt-a1i/archify -g`), run from the repository root,
replacing `<archify>` with the installed package's directory:

```bash
node <archify>/bin/archify.mjs finalize architecture docs/diagrams/recall-overview.architecture.json docs/diagrams/recall-overview.html --repo-root . --quality showcase --json
```

```bash
node <archify>/bin/archify.mjs finalize sequence docs/diagrams/recall-search.sequence.json docs/diagrams/recall-search.html --repo-root . --quality showcase --json
```

To move to a newer commit, update `meta.repository.revision` in the JSON, re-check every cited
line range against that commit, and run the commands again. Archify writes receipts beside each
page; this folder's `.gitignore` keeps them out of the tree, because they record local paths.
