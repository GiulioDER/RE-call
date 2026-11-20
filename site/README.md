# Legacy site source

This directory contains the retired GitHub Pages site. Its canonical URLs and sitemap point at
the new domain, `https://re-call.dev/`, at the same paths. On 2026-11-20 only `/` and
`/start.html` answered there, and that `start.html` is the new site's own page, so publish a page
at its path before relying on its canonical. The maintained
project documentation is in [docs/README.md](../docs/README.md).

## Page identity

Each guide and blog page has an absolute canonical URL. The homepage uses the
directory URL rather than `index.html`; the blog index follows the same convention.
Keep a page's title and description specific to its content. The homepage's structured
data connects the project site to its GitHub repository and PyPI package.

The homepage repeats its title in `og:title`, and its description in `og:description` and in
each JSON-LD node; change every copy together. `tests/test_site_metadata.py` fails when they
disagree.

`installation.html` and `requirements.html` are moved-page aliases. Keep their
redirects, `noindex` directives, and canonical links to `start.html` together.
The generated diagrams remain maintained through `docs/diagrams/README.md`.

## Sitemap

`sitemap.xml` lists the canonical guides and blog pages. Add or remove an entry when
publishing or retiring one of those pages. Leave redirect aliases out. Do not invent
`lastmod` dates or update them merely because the site was deployed.

## Pages for agents

`llms.txt` is a plain Markdown index for language models and agents, following the
llms.txt convention: a one-paragraph summary, when to choose RE-call, the install facts, and
links to the canonical guides. It repeats facts kept elsewhere, so update it when the install
path, the licence, the package name or a linked page changes. It is linked from the homepage
`<head>` and footer and is not listed in the sitemap, which holds HTML pages only.

`compare.html` and `benchmarks.html` have maintained Markdown copies in `docs/COMPARE.md` and
`docs/BENCHMARKS.md`, which `context7.json` keeps indexed while it excludes this directory; change
both copies together. `compare.html` describes other memory projects from their public sources, with the check date in
its text. Recheck those descriptions and update the date together; keep score claims to the
paired measurements on the benchmark scorecard.

## Benchmark scorecard

`benchmarks.html` is the compact register of published benchmark families. It is static
HTML, so the scores, sources and disclosures remain available without JavaScript.
`benchmarks.css` extends the shared site tokens. The homepage and sitemap link to it.

To add a result, including a future AML result:

1. Establish that it is publishable. Retain a public compact score artifact or an
   organizer result, with the metric, split, run date, model stack and reproduction path.
   A partial screen, failed run or private competition result is not a leaderboard score.
2. Copy one `benchmark-row`, give it a unique id and name the exact metric and scope.
   Link the artifact and method in its native `details` element. Preserve negative
   results and relevant comparator or configuration limits.
3. Label who operated the run. An official scorer, signature or accepted submission
   does not justify `Independently verified`. That label requires a public independent
   rerun. Use `Organizer-listed` separately when applicable.
4. Check the live organizer board. State the rank's metric, track, eligible row set,
   ties and check date. Do not insert a local result into an official table to invent
   its place. An owner-run board must say so. Keep a local recomparison labelled local.
5. Recheck every displayed value against its linked source, including rounding and
   percentage conversion. Never substitute Hit@k for Recall@k or retrieval for QA.
   Update the evidence review date after reviewing the rows, not just deploying.
6. Preview at narrow and desktop widths, expand the disclosures with the keyboard,
   check horizontal overflow, and run the repository's public-tree guard. Inspect the
   canonical URL and sitemap entry if the page identity changes.

The scorecard intentionally carries no placeholder score or rank for an unpublished
competition. Its historical sections preserve tested families whose artifacts or
former public boards are unavailable. Detailed configuration sweeps remain linked to
their existing records rather than being presented as additional leaderboard entries.
