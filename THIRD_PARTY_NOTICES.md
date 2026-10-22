# Third party notices

RE-call is the original work of Giulio D'Erme and is licensed under AGPL-3.0-only (see `LICENSE`
and `NOTICE`). It redistributes no third party source code in the packages it ships. It does
depend on the libraries, models and hosted services below, which it uses unmodified and which
remain under their own licences and terms.

Every licence on this page was read from its source on 2026-10-20: Python packages from the PyPI
JSON API for the exact version locked in `uv.lock`, models from their Hugging Face page,
repositories from GitHub, papers from arXiv or Crossref. None was filled in from memory. When a
lock entry changes, the row for it is stale until it is read again.

Not legal advice.

## Copyleft and file level copyleft libraries

These are the dependencies whose licence carries conditions beyond attribution. All are used as
unmodified, separately installed libraries.

| Package | Licence | What it is here |
|---|---|---|
| `psycopg` 3.3.4 | LGPL-3.0-only | The PostgreSQL adapter. A core dependency (`psycopg[binary]`). |
| `psycopg-binary` 3.3.4 | LGPL-3.0-only | The compiled distribution of psycopg pulled in by `psycopg[binary]`. Its wheels bundle the PostgreSQL client library libpq (PostgreSQL Licence) and, where the platform needs it, OpenSSL (OpenSSL 3, Apache-2.0); the Windows wheel for 3.3.4 carries `libpq`, `libssl-3` and `libcrypto-3`. |
| `psycopg-pool` 3.3.3 | LGPL-3.0-only | Connection pooling; the `pool`, `mcp` and `hosted` extras. |
| `certifi` 2026.6.17 | MPL-2.0 | Mozilla's CA bundle, pulled in by `requests` and `httpx`. |
| `tqdm` 4.69.0 | MPL-2.0 AND MIT | Progress bars, pulled in by `fastembed` and `huggingface-hub`. |
| `orjson` 3.11.9 | MPL-2.0 AND (Apache-2.0 OR MIT) | JSON, pulled in by `langsmith`, which `voyageai` reaches through `langchain-text-splitters` and `langchain-core`. |

Sources: PyPI JSON API for each version above; https://www.psycopg.org/psycopg3/docs/basic/install.html
("Precompiled C extensions, packaged with client libraries"); https://www.postgresql.org/about/licence/;
https://openssl-library.org/source/license/index.html.

## Python runtime dependencies

The full dependency closure, read from `uv.lock`, of the base install and of the extras the
shipped services use: `mcp` (the MCP server), `pool`, `voyage`, `hosted` and `openai` (the
hosted service and `recall_aml`), `fastembed` (the default local embedder), `documents` (file
extraction) and `atomic`. The last column names every one of those that pulls the package in;
`core` means the base install.

Research and benchmarking extras (`sparse`, `rerank`, `finetune`, `eval`, `bench`, `agent-ab`,
`agent`, `aws`, `s3`, `langchain`, `dev`) are not listed. Their pinned closure is in `uv.lock`, and
the model weights some of them download are covered in [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md).

The licence is quoted as PyPI declares it for that version, from the field named in the fourth
column: the SPDX `License-Expression` where the package sets one, otherwise its `License` field,
otherwise its trove classifier.

| Package | Locked version | Licence as declared | Declared in | Installed by |
|---|---|---|---|---|
| `aiohappyeyeballs` | 2.7.1 | PSF-2.0 | License | voyage, hosted |
| `aiohttp` | 3.14.5 | Apache-2.0 AND MIT | License-Expression | voyage, hosted |
| `aiolimiter` | 1.2.1 | MIT | License | voyage, hosted |
| `aiosignal` | 1.4.0 | Apache 2.0 | License | voyage, hosted |
| `annotated-types` | 0.7.0 | MIT License | classifier | core |
| `anyio` | 4.15.1 | MIT | License-Expression | mcp, voyage, hosted, openai, fastembed |
| `attrs` | 26.1.0 | MIT | License-Expression | mcp, voyage, hosted |
| `beautifulsoup4` | 4.15.0 | MIT License | License | documents |
| `certifi` | 2026.6.17 | MPL-2.0 | License | voyage, hosted, fastembed |
| `cffi` | 2.1.0 | MIT-0 | License-Expression | mcp, documents |
| `charset-normalizer` | 3.4.9 | MIT | License | voyage, hosted, fastembed, documents |
| `click` | 8.4.2 | BSD-3-Clause | License-Expression | mcp, voyage, hosted, fastembed, documents |
| `colorama` | 0.4.6 | BSD License | classifier | mcp, voyage, hosted, fastembed, documents (Windows only) |
| `cryptography` | 50.0.0 | Apache-2.0 OR BSD-3-Clause | License-Expression | mcp, documents |
| `distro` | 1.9.0 | Apache License, Version 2.0 | License | voyage, hosted |
| `et-xmlfile` | 2.0.0 | MIT | License | documents |
| `fastembed` | 0.8.1 | Apache-2.0 | License | fastembed |
| `filelock` | 3.31.1 | MIT | License-Expression | voyage, hosted, fastembed |
| `flatbuffers` | 25.12.19 | Apache 2.0 | License | fastembed |
| `frozenlist` | 1.8.0 | Apache-2.0 | License | voyage, hosted |
| `fsspec` | 2026.9.0 | BSD-3-Clause | License-Expression | voyage, hosted, fastembed |
| `h11` | 0.16.0 | MIT | License | mcp, voyage, hosted, openai, fastembed |
| `hf-xet` | 1.5.2 | Apache-2.0 | License-Expression | voyage, hosted, fastembed |
| `httpcore` | 1.0.9 | BSD-3-Clause | License-Expression | voyage, hosted, fastembed |
| `httpcore2` | 2.13.1 | BSD-3-Clause | License-Expression | mcp, hosted, openai |
| `httpx` | 0.28.1 | BSD-3-Clause | License | voyage, hosted, fastembed |
| `httpx2` | 2.13.1 | BSD-3-Clause | License-Expression | mcp, hosted, openai |
| `httpx2-jsfetch` | 1.0 | BSD-3-Clause | License-Expression | mcp, hosted, openai (Emscripten only) |
| `huggingface-hub` | 1.24.0 | Apache-2.0 | License | voyage, hosted, fastembed |
| `idna` | 3.18 | BSD-3-Clause | License-Expression | mcp, voyage, hosted, openai, fastembed |
| `jiter` | 0.16.0 | MIT | License-Expression | hosted, openai |
| `jsonpatch` | 1.33 | Modified BSD License | License | voyage, hosted |
| `jsonpointer` | 3.1.1 | Modified BSD License | License | voyage, hosted |
| `jsonschema` | 4.26.0 | MIT | License-Expression | mcp |
| `jsonschema-specifications` | 2025.9.1 | MIT | License-Expression | mcp |
| `langchain-core` | 1.6.7 | MIT | License | voyage, hosted |
| `langchain-protocol` | 0.0.18 | MIT | License | voyage, hosted |
| `langchain-text-splitters` | 1.1.2 | MIT | License | voyage, hosted |
| `langsmith` | 0.10.7 | MIT | License | voyage, hosted |
| `loguru` | 0.7.3 | MIT License | classifier | fastembed |
| `lxml` | 6.1.1 | BSD-3-Clause | License | documents |
| `mcp` | 2.2.0 | MIT | License | mcp |
| `mcp-types` | 2.2.0 | MIT | License | mcp |
| `mmh3` | 5.2.1 | MIT License | classifier | fastembed |
| `multidict` | 6.9.1 | Apache License 2.0 | License | voyage, hosted |
| `numpy` | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | License-Expression | core |
| `olefile` | 0.47 | BSD | License | documents |
| `onnxruntime` | 1.31.0 | MIT License | License | fastembed |
| `openai` | 3.26.0 | Apache-2.0 | License-Expression | hosted, openai |
| `openpyxl` | 3.1.5 | MIT | License | documents |
| `opentelemetry-api` | 1.44.0 | Apache-2.0 | License-Expression | mcp |
| `orjson` | 3.11.9 | MPL-2.0 AND (Apache-2.0 OR MIT) | License-Expression | voyage, hosted |
| `packaging` | 26.2 | Apache-2.0 OR BSD-2-Clause | License-Expression | voyage, hosted, fastembed |
| `pdfminer-six` | 20260107 | MIT | License-Expression | documents |
| `pdfplumber` | 0.11.10 | MIT License | classifier | documents |
| `pgvector` | 0.5.0 | MIT | License-Expression | core |
| `pillow` | 12.3.0 | MIT-CMU | License-Expression | voyage, hosted, fastembed, documents |
| `propcache` | 0.5.2 | Apache-2.0 | License | voyage, hosted |
| `protobuf` | 6.33.6 | 3-Clause BSD License | License | fastembed |
| `psycopg` | 3.3.4 | LGPL-3.0-only | License-Expression | core |
| `psycopg-binary` | 3.3.4 | LGPL-3.0-only | License-Expression | core |
| `psycopg-pool` | 3.3.3 | LGPL-3.0-only | License-Expression | mcp, pool, hosted |
| `py-rust-stemmers` | 0.1.8 | none declared on PyPI; MIT in its source repository, qdrant/py-rust-stemmers |  | fastembed |
| `pycparser` | 3.0 | BSD-3-Clause | License-Expression | mcp, documents |
| `pydantic` | 2.13.5 | MIT | License-Expression | core |
| `pydantic-core` | 2.46.5 | MIT | License-Expression | core |
| `pyjwt` | 2.15.1 | MIT | License-Expression | mcp |
| `pypdfium2` | 5.13.0 | BSD-3-Clause, Apache-2.0, dependency licenses | License | documents |
| `python-docx` | 1.2.0 | MIT | License | documents |
| `python-multipart` | 0.0.32 | Apache-2.0 | License-Expression | mcp |
| `python-oxmsg` | 0.0.2 | MIT | License | documents |
| `python-pptx` | 1.0.2 | MIT | License | documents |
| `pywin32` | 312 | PSF | License | mcp (Windows only) |
| `pyyaml` | 6.0.3 | MIT | License | voyage, hosted, fastembed |
| `referencing` | 0.37.0 | MIT | License-Expression | mcp |
| `requests` | 2.34.2 | Apache-2.0 | License | voyage, hosted, fastembed |
| `requests-toolbelt` | 1.0.0 | Apache 2.0 | License | voyage, hosted |
| `rpds-py` | 2026.6.3 | MIT | License-Expression | mcp |
| `sniffio` | 1.3.1 | MIT OR Apache-2.0 | License | voyage, hosted, openai |
| `soupsieve` | 2.9.2 | MIT | License-Expression | documents |
| `sse-starlette` | 3.4.5 | BSD-3-Clause | License-Expression | mcp |
| `starlette` | 1.7.0 | BSD-3-Clause | License-Expression | mcp, hosted |
| `tenacity` | 9.1.4 | Apache 2.0 | License | voyage, hosted |
| `threadpoolctl` | 3.7.0 | BSD-3-Clause | License-Expression | atomic |
| `tokenizers` | 0.23.3 | Apache Software License | classifier | voyage, hosted, fastembed |
| `tqdm` | 4.69.0 | MPL-2.0 AND MIT | License | voyage, hosted, fastembed |
| `truststore` | 0.10.4 | MIT | License-Expression | mcp, hosted, openai |
| `typing-extensions` | 4.16.0 | PSF-2.0 | License-Expression | core |
| `typing-inspection` | 0.4.2 | MIT | License-Expression | core |
| `tzdata` | 2026.3 | Apache-2.0 | License | core (Windows only) |
| `urllib3` | 2.8.0 | MIT | License-Expression | voyage, hosted, fastembed |
| `uuid-utils` | 0.17.0 | BSD-3-Clause | License-Expression | voyage, hosted |
| `uvicorn` | 0.54.0 | BSD-3-Clause | License-Expression | mcp, hosted |
| `voyageai` | 0.5.0 | MIT | License | voyage, hosted |
| `websockets` | 16.1.1 | BSD-3-Clause | License-Expression | voyage, hosted |
| `win32-setctime` | 1.2.0 | MIT license | License | fastembed (Windows only) |
| `xlrd` | 2.0.2 | BSD | License | documents |
| `xlsxwriter` | 3.2.9 | BSD-2-Clause | License | documents |
| `xxhash` | 3.8.1 | BSD-2-Clause | License | voyage, hosted |
| `yarl` | 1.25.1 | Apache-2.0 | License | voyage, hosted |
| `zstandard` | 0.25.0 | BSD-3-Clause | License-Expression | voyage, hosted |

## Hosted services used by `recall_aml`

`recall_aml` calls these models over their providers' APIs. No weights are downloaded or
redistributed; each is used under its provider's terms of service.

| Service | Models | Where | Terms |
|---|---|---|---|
| Voyage AI (MongoDB) | `voyage-code-4`, `voyage-context-4` and `voyage-multimodal-3.5` (embeddings); `rerank-3` (reranking) | `recall_aml.code4`, `recall_aml.config` with `recall.embedding_providers.voyage`, `recall_aml.multimodal`, `recall_aml.rerank_switch` | https://www.voyageai.com/tos |
| OpenAI, through OpenRouter | `gpt-4o-mini` (`openai/gpt-4o-mini`) | `recall_aml.config` (`GENERATION_MODEL`, `OPENROUTER_BASE_URL`) and the compiler that uses it | https://openrouter.ai/terms, and the model provider's terms that OpenRouter applies |

Two Voyage tokenizers are downloaded from Hugging Face at run time, only to count tokens before a
request; they are listed with their licences in [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md).

## Text adapted from other projects

| Where | Adapted from | Licence |
|---|---|---|
| `recall/evidence.py`, `DATED_READER_CONTRACT` | the reader contract of [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) | MIT, Copyright (c) 2025 Vectorize AI, Inc. |

Evaluation prompts copied into the research harnesses under `benchmarks/`, which are not part of
any shipped package, are recorded beside them with their sources and licences:
[benchmarks/beam/NOTICE.md](benchmarks/beam/NOTICE.md) for mem0ai/memory-benchmarks (Apache-2.0),
BEAM (MIT) and AML's public pipeline (no licence declared),
[benchmarks/benchd/NOTICE.md](benchmarks/benchd/NOTICE.md) for the Bench'd harness (MIT), and
[benchmarks/mtrag/NOTICE.md](benchmarks/mtrag/NOTICE.md) for the MTRAG paper (CC BY-SA 4.0).

## Published methods

The retrieval and statistics code implements these published methods. Each is also cited where
it is implemented. References checked on Crossref, arXiv or zbMATH on 2026-10-20.

| Method | Reference | Implemented in |
|---|---|---|
| Okapi BM25 | S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", Foundations and Trends in Information Retrieval, 2009, doi:10.1561/1500000019 | `recall/eval/bm25.py`, `recall_aml/code4.py` |
| Reciprocal rank fusion | G. V. Cormack, C. L. A. Clarke and S. Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods", SIGIR 2009, pp. 758 to 759, doi:10.1145/1571941.1572114 | `recall/retriever.py`, `recall_aml/retrieval.py`, `recall_aml/multimodal.py` |
| SPLADE | T. Formal, B. Piwowarski and S. Clinchant, "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking", SIGIR 2021, doi:10.1145/3404835.3463098; T. Formal, C. Lassance, B. Piwowarski and S. Clinchant, "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval", arXiv:2109.10086, 2021 | `recall/sparse.py` |
| Wilson score interval | E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference", Journal of the American Statistical Association 22(158), 1927, pp. 209 to 212, doi:10.1080/01621459.1927.10502953 | `recall/eval/metrics.py` |
| Percentile bootstrap | B. Efron, "Bootstrap Methods: Another Look at the Jackknife", The Annals of Statistics 7(1), 1979, pp. 1 to 26, doi:10.1214/aos/1176344552 | `recall/eval/metrics.py`, `recall/promotion.py` |
| Holm correction | S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure", Scandinavian Journal of Statistics 6, 1979, pp. 65 to 70 | `recall/promotion.py` |

The feedback guard in `recall_aml/feedback_guard.py` recognises the record shapes of
[AgentMemoryBench](https://github.com/solomoon313/AgentMemoryBench) (MIT, read at commit
`9dfda8de`) and rests on findings of StreamBench (C.-K. Wu et al., arXiv:2406.08747, 2024) and
Evo-Memory (T. Wei et al., arXiv:2511.20857, 2025). No code is copied from any of them; its
module docstring gives the full references.
