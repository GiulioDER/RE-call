# Repository Map

This repository contains the installable library, product documentation, examples, and the evidence
record behind the public claims. Use this map when deciding what is product surface and what is
retained for reproducibility.

| Path | Role | Stability |
|---|---|---|
| `recall/` | Core Python library. | Supported API is listed in [API.md](API.md). |
| `recall_mcp/` | MCP server and tool implementation. | Supported MCP tools are listed in [API.md](API.md). |
| `recall_hooks/` | Claude Code session hooks (deliberately import-light; shipped in the wheel). | Supported through the plugin. |
| `recall_interop/` | Interop adapters used by benchmark harnesses. | Repository support, not packaged library API. |
| `recall_consistency/` | Repository consistency checks. | Repository support, not packaged library API. |
| `plugin/` | The Claude Code plugin: MCP wiring, hooks, and the search skill. | Supported; see [../plugin/README.md](../plugin/README.md). |
| `site/` | Source of the setup guide published at giulioder.github.io/RE-call. | Published on every push to master. |
| `launch/` | Launch assets: the terminal demo video/GIF and its renderer. | Asset source, regenerate rather than edit outputs. |
| `hooks/` | Deployable copies of repository guard hooks. | Repository support. |
| `packaging/` | Windows installer packaging. | Release tooling. |
| `docker/` | Container build files. | Release tooling. |
| `examples/` | Runnable product examples. | Stable enough for users to copy and adapt. |
| `corpus/` | Small example corpus for local retrieval checks. | Example data only. |
| `docs/` | Product, operating, architecture, and evidence guides. | Product docs are the public entry path. |
| `benchmarks/` | Benchmark harnesses, protocols, and reproduction helpers. | Evidence support, not library API. |
| `benchmarks/archive/` | Older preregistrations and benchmark protocol records. | Audit archive. |
| `results/` | Published result summaries, compact artifacts, and claim baselines. | Evidence record. |
| `docs/archive/` | Historical changelog and program status records. | Audit archive. |
| `scripts/` | Reproduction and maintenance helpers. | Task specific. |
| `tests/` | Unit, integration, and regression tests. | Maintained gate. |

## Reader Paths

| Goal | Start here |
|---|---|
| Try the product | [README.md](../README.md#quickstart), then [examples/README.md](../examples/README.md). |
| Connect an agent via MCP, or set up CLAUDE.md/memory files | [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md). |
| Integrate the library | [API.md](API.md), [PRODUCTION.md](PRODUCTION.md), and [OPERATING_MODES.md](OPERATING_MODES.md). |
| Configure deployment | [ENVIRONMENT.md](ENVIRONMENT.md), [MIGRATIONS.md](MIGRATIONS.md), and [AUTH.md](AUTH.md). |
| Audit public claims | [EVIDENCE.md](EVIDENCE.md), [../results/README.md](../results/README.md), and [../benchmarks/README.md](../benchmarks/README.md). |
| Reproduce benchmarks | [../benchmarks/README.md](../benchmarks/README.md), then the referenced result artifact. |

## What Is Not Product Surface

Benchmark modules, result builders, preregistrations, archived program ledgers, and generated
figures are retained so claims can be audited. They are intentionally visible, but they are not the
supported application API.
