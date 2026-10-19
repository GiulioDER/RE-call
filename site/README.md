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
