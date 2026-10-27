# Publishing and search discovery

GitHub Pages publishes this directory through `.github/workflows/pages.yml`.
The public base URL is `https://giulioder.github.io/RE-call/`.

## Page identity

Each guide and blog page has an absolute canonical URL. The homepage uses the
directory URL rather than `index.html`; the blog index follows the same convention.
Keep a page's title and description specific to its content. The homepage's structured
data connects the project site to its GitHub repository and PyPI package.

`installation.html` and `requirements.html` are moved-page aliases. Keep their
redirects, `noindex` directives, and canonical links to `start.html` together.
The generated diagrams remain maintained through `docs/diagrams/README.md`.

## Sitemap

`sitemap.xml` lists the canonical guides and blog pages. Add or remove an entry when
publishing or retiring one of those pages. Leave redirect aliases out. Do not invent
`lastmod` dates or update them merely because the site was deployed.

The site lives under a GitHub Pages project path. A `robots.txt` inside this directory
would be served at `/RE-call/robots.txt`, but crawlers read `/robots.txt` at the host
root. A missing host-root robots file does not block crawling. Submit this project's
sitemap directly in Search Console instead of relying on a project-path robots file.

## Google Search Console

1. Add the URL-prefix property `https://giulioder.github.io/RE-call/`.
2. Choose HTML tag verification and add the exact Google-provided meta tag to the
   homepage's `<head>`. Preserve it across later edits.
3. Publish the site, confirm the tag appears in the live HTML, then verify ownership.
4. Submit `https://giulioder.github.io/RE-call/sitemap.xml` under Sitemaps.
5. Inspect the homepage URL, run the live test, and request indexing if eligible.
   Use Page indexing and URL inspection to diagnose exclusions and Google's selected
   canonical. Public search queries alone do not establish the indexing state.

A sitemap and metadata help discovery and identification; neither guarantees indexing
or a particular search position.

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
