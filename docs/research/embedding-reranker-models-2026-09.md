# Embedding and Reranker Model Landscape for Retrieval Quality

## Executive conclusion

The strongest next experiment is not another flat embedding swap. The project already has evidence that the dominant failure is often candidate recall: the gold memory is absent from the candidate pool, so a reranker cannot recover it. The first priority should therefore be a contextual retrieval arm, with conversation or document context available while each retrievable unit is encoded.

The highest value test sequence is:

1. Voyage Context 4, grouped by conversation or source document, against the current Voyage 4 Large baseline.
2. Jina Embeddings v5 Text Small as the first serious local alternative, using its retrieval adapter and full 1024 dimensions for the screen.
3. Jina Reranker v3.5 and Cohere Rerank v4 Pro as the strongest new reranker alternatives.
4. Qwen3 Embedding 0.6B and Qwen3 Reranker 0.6B as the practical Apache 2.0 local pair.
5. ZeroEntropy zembed 1 and zerank 2 as a high upside research lane, self hosted only because the ZeroEntropy service was scheduled for sunset on September 4, 2026.

The project should score gold coverage at the candidate pool boundary before scoring final top k. A model that improves answer ranking while leaving gold absent from the pool is not solving the primary problem.

## What the project evidence says

The local project evidence gives a useful baseline. Voyage 4 Large has won 16 of 17 measured corpora, with a median hit at 5 improvement of 0.059. A prior reranker run improved hit at 5 from 0.671 to 0.777, but added approximately 1.05 seconds per query. Another pool level experiment lifted hit at 5 from 0.640 to 0.870 with Voyage reranking over a top 50 pool, while the remaining gap was attributed to pool recall rather than ranking. These are internal measurements, not vendor claims, and they are the correct comparator for new models.

The project has also observed a conversational failure mode in which 71 percent of category 1 failures had no gold item in context. Isolated turns such as a short pronoun or a date fragment are semantically weak. A model with better general benchmark scores may still lose if it is asked to embed an ambiguous turn without the surrounding conversation.

The current operating constraints matter. Embedding runs belong on VPS2, only one embedding process may run at a time, and the indexing path is bounded because VPS2 also runs live services. The current Voyage reranker project limit observed by the benchmark was 2,000,000 rerank tokens per minute, which caused a candidate pool of 200 to be rejected in one run. Any cloud comparison must therefore report token volume and the exact candidate pool, not only quality.

Internal references: [VOYAGE_REFERENCE.md](../benchmarks/VOYAGE_REFERENCE.md), [EVIDENCE.md](../docs/EVIDENCE.md), and the prior [rerank pool results](../results/peps_rerank_pool/README.md).

## Model shortlist

| Priority | Model | Role | Deployment | Key capability | Main reason to test | Main risk |
|---:|---|---|---|---|---|---|
| 1 | `voyage-context-4` | Embedding | Cloud | Contextualized chunk embeddings, 1024 dimensions, 32K chunk context, 120K document budget | Directly targets ambiguous conversation turns and document context loss | Requires grouping chunks by source document; it is not a drop in embedder |
| 2 | `jina-embeddings-v5-text-small` | Embedding | Local or Jina API | 677M parameters, 32K context, 1024 dimensions, retrieval adapter, Matryoshka and binary quantization | Best new practical local candidate under 1B parameters by the vendor reported MTEB results | CC BY NC 4.0 licensing requires a commercial use review |
| 3 | `jina-reranker-v3.5` | Reranker | Local or Jina API | 0.6B listwise reranker, 131K context, hybrid attention, self distillation | New listwise architecture can use relationships among the candidate documents instead of scoring every pair in isolation | CC BY NC 4.0 licensing and a different listwise integration shape |
| 4 | `rerank-v4.0-pro` | Reranker | Cloud or private deployment | Multilingual, structured data support, 32K context | The cleanest cloud comparison against Voyage Rerank 2.5 | Hosted cost, latency, and no public project specific evidence |
| 5 | `Qwen3-Embedding-0.6B` | Embedding | Local | 0.6B parameters, 32K context, 1024 dimensions, instruction aware, Apache 2.0 | Strong local control that should fit the bounded VPS2 lane | Prompt formatting is mandatory and results may differ sharply with or without instructions |
| 6 | `Qwen3-Reranker-0.6B` | Reranker | Local | 0.6B parameters, 32K context, instruction aware, Apache 2.0 | Low resource reranker baseline with the same model family as the embedder | Smaller than the likely quality ceiling |
| 7 | `Qwen3-Reranker-4B` | Reranker | Local or dedicated GPU | 4B parameters, 32K context, Apache 2.0 | Stronger open reranker with a model card score of 69.76 on MTEB retrieval subsets | Likely outside the current 8 GB bounded inference envelope without quantization or another host |
| 8 | `zeroentropy/zerank-2-reranker` | Reranker | Local or self managed provider | 4B parameters, 32K context, zELO calibrated relevance training, Apache 2.0 on the current model card | High upside model with vendor reported gains on conversational, finance, legal, code, and STEM data | The hosted API was scheduled for sunset; public comparisons are vendor supplied |
| 9 | `zeroentropy/zembed-1-embedding` | Embedding | Local or self managed provider | 4B parameters, 32K context, flexible 40 to 2560 dimensions, Apache 2.0 on the current model card | The current model card reports a large gain over Voyage 4 Nano on domain averages | No hosted provider is listed, it is larger than the current local lane, and the evidence is not an independent benchmark |
| 10 | `embed-v4.0` | Embedding | Cloud or private deployment | Multimodal and multilingual, 128K context, 256 to 1536 dimensions | Strong cloud control, especially if structured documents, images, or PDFs enter the corpus | Pure text gain over Voyage 4 Large is uncertain |
| 11 | `gemini-embedding-2` | Embedding | Cloud | Unified text, image, audio, video, and PDF space, 3072 dimensions, custom task instructions | Useful if the corpus becomes multimodal or needs task specific embedding prompts | The embedding space is incompatible with Gemini Embedding 001 and requires a full reindex |
| 12 | `google/embeddinggemma-300m` | Embedding | Local | 300M parameters, 768 dimensions, 2K context, MRL, quantization aware training | Excellent low resource control and easy edge deployment | The short 2K input limit makes it a weaker fit for long conversation or document context |

## Embedding models

### Voyage Context 4

Voyage Context 4 is the best fit for the project’s known failure mode. Voyage’s API accepts a list of chunk lists, with each inner list representing one document. The model embeds each chunk with awareness of the other chunks in that document. It supports 32K tokens per chunk and a 120K token document budget, with the same 1024 default dimension family as Voyage 4 Large. The official documentation also provides batch inference support.

This should be tested with a conversation as the document and individual turns as the retrievable chunks. The query remains a standalone question. The primary result should be gold in candidate pool, not only top k precision. A positive result would justify the implementation effort more strongly than a small improvement from a larger isolated turn encoder.

### Jina Embeddings v5 Text Small

Jina’s February 2026 model is a 677M parameter multilingual text encoder built on Qwen3 0.6B Base. It supports 32K input, 1024 output dimensions, task specific adapters, and truncation down to 32 dimensions. The model card reports 64.88 retrieval on MTEB Multilingual, 66.84 on RTEB, 56.67 on BEIR, and 66.39 on LongEmbed. These are vendor reported benchmark figures and should be treated as screening evidence rather than a prediction for this corpus.

The important implementation detail is the retrieval adapter. Queries and passages should use the asymmetric retrieval configuration, with the documented query and document prefixes. A symmetric embedding call would make the comparison invalid. The model also supports binary quantization, which could be tested only after the full precision screen.

### Qwen3 Embedding

Qwen3 Embedding is the strongest permissively licensed local family in the shortlist. The 0.6B model outputs up to 1024 dimensions. The 4B model outputs up to 2560 dimensions, and the 8B model outputs up to 4096 dimensions. All support 32K context, user instructions, and flexible dimensions.

The Qwen model card reports 64.64 retrieval on its multilingual MTEB table for the 0.6B model, 69.60 for 4B, and 70.88 for 8B. It also states that instructions commonly improve retrieval by 1 percent to 5 percent in the authors’ tests. Every experiment must therefore record the exact query instruction, passage formatting, model revision, and normalization path.

For the current host, start with 0.6B. Reserve 4B and 8B for a separate GPU lane, quantized trial, or cloud inference provider. A model that cannot run inside the production safety envelope is useful as a ceiling experiment, not as an immediate deployment candidate.

### ZeroEntropy zembed 1

The current Hugging Face model card describes zembed 1 as a 4B multilingual embedding model distilled from zerank 2 using the zELO training method. It supports 32K context and output dimensions from 40 through 2560. Its domain table reports a mean NDCG at 10 of 0.5561, compared with 0.5050 for Voyage 4 Nano, 0.5013 for Qwen3 4B, 0.4957 for Cohere Embed 4, and 0.4837 for Gemini Embedding 001.

This is worth testing because the reported conversational score is 0.5385, well above the listed Voyage 4 Nano score of 0.4045. It is not evidence that zembed 1 will beat Voyage 4 Large on the project corpus. The table is averaged across public and private domain benchmarks and comes from the model publisher. The model has no hosted inference provider listed on its current card, so it belongs in a self hosted research lane.

### Cohere Embed v4 and Gemini Embedding 2

Cohere Embed v4 is a cloud or private deployment control with 128K context, Matryoshka dimensions of 256, 512, 1024, and 1536, and unified text and image embeddings. It is most interesting if the corpus contains semi structured data, PDFs, images, or tables. For a pure text conversational memory corpus, its incremental value over Voyage 4 Large is uncertain and must be measured.

Gemini Embedding 2 is Google’s newer unified multimodal embedding model. Google documents text, image, video, audio, and PDF input in one semantic space, 3072 default dimensions, adjustable output dimensionality, and custom task instructions embedded in the prompt. Gemini Embedding 2 cannot share an index with Gemini Embedding 001, so a comparison requires a full reindex. It is a secondary screen for this project unless multimodal memory becomes a near term requirement.

### EmbeddingGemma

EmbeddingGemma is a 300M parameter open model intended for on device use. Google documents 768, 512, 256, and 128 dimension outputs, a 2K input limit, and quantization aware checkpoints. The model card reports 61.15 on multilingual MTEB at 768 dimensions and 69.67 on English MTEB.

It is not the likely quality winner here, but it is a valuable operational control. It can establish the quality floor for a very small local model and measure whether a local model can replace the current bge path without adding host pressure. Use the required retrieval query and document prompts from the model card.

### Additional embedding candidates

Jina Embeddings v5 Text Nano is a 239M parameter model with 8K context and 768 dimensions. Jina reports 63.3 retrieval on MMTEB and positions it for edge deployment, with GGUF and llama.cpp support. It is a useful lower resource comparison against EmbeddingGemma and Qwen3 Embedding 0.6B. It is less suitable than v5 Text Small for long conversation context, and it carries the same CC BY NC 4.0 licensing consideration.

Nomic Embed v2 MoE is a fully open Apache 2.0 model with 475M total parameters and 305M active parameters, trained for roughly 100 languages. Its unusual feature is sparse mixture of experts routing combined with Matryoshka dimensions. The model card reports a 512 token maximum, so it is a compact short passage control rather than a contextual conversation encoder. It is worth including if the benchmark has short, atomic memory chunks and local inference efficiency matters.

KaLM Embedding Gemma3 12B is a quality ceiling candidate. Its model card reports 72.32 mean task score on its MMTEB table and 75.66 on retrieval, above the reported Qwen3 Embedding 8B score in that table. The model is too large for the current VPS2 bounded lane, but it can answer an important infrastructure question on a dedicated GPU: whether the project still has meaningful quality headroom above Voyage 4 Large. Treat the figures as publisher reported and verify the license before any production use.

NVIDIA Llama Embed Nemotron 8B is another ceiling candidate with a released training dataset and recipe. The model card identifies multilingual retrieval as a target and marks the license as a customized NVIDIA and Llama license. The model artifact is approximately 15 GB in the current repository listing and is marked for noncommercial or research use, so it is not an immediate production option. It is useful only if a separate GPU comparison is already planned.

## Reranker models

### Jina Reranker v3.5

Jina Reranker v3.5 is the most interesting architectural alternative. It is a 0.6B listwise reranker that reads a query and a candidate list in one context, rather than treating every query document pair as independent. The model uses three sliding window layers followed by two global layers, then reads a representation from the final token of each document. Jina reports 63.20 NDCG at 10 on BEIR, 74.11 on MIRACL, and 70.95 on RTEB, using a top 100 protocol.

This directly targets the project’s candidate interaction problem. It can learn that one candidate is a better answer than a near duplicate, that two candidates are redundant, or that one candidate is a distractor in the context of the entire list. The test must preserve the list order and candidate pool while comparing listwise and pairwise rerankers. It should not be reduced to independent pair scores unless the implementation requires a compatibility mode.

### Cohere Rerank v4

Cohere Rerank v4 has Pro and Fast variants. Cohere documents both as multilingual, 32K context rerankers that also accept semi structured JSON. Pro is the quality arm, and Fast is the latency arm. This is the clearest managed alternative to Voyage Rerank 2.5.

The project should test Pro at candidate pools of 25, 50, and 100, then test Fast only if Pro’s quality is promising. The relevant comparison is marginal gold recovery per rerank token and per second. A larger context limit does not itself imply better ranking, especially when each memory chunk is short.

### Qwen3 Reranker

Qwen3 Reranker provides 0.6B, 4B, and 8B models under Apache 2.0. The model card reports multilingual retrieval subset scores of 65.80 for 0.6B, 69.76 for 4B, and 69.02 for 8B, using top 100 candidates retrieved by Qwen3 Embedding 0.6B. It supports custom instructions and 32K context.

The 0.6B model is the correct first local experiment. The 4B model is the quality ceiling to test later. Its raw output is a yes logit rather than a calibrated probability, so the comparison should use rank order for retrieval and fit any threshold only on a separate calibration set.

### ZeroEntropy zerank 2

zerank 2 is a 4B Qwen3 based reranker trained with adjusted Elo style relevance targets. The current Hugging Face card reports average NDCG at 10 of 0.6714 over seven domains, compared with 0.5847 for Cohere Rerank 3.5 and 0.5999 for a Gemini 2.5 Flash listwise reference. The same table reports 0.6140 on conversational data.

The model is now an Apache 2.0 artifact on Hugging Face, but ZeroEntropy’s managed products were announced for sunset on September 4, 2026. Use it only as a self hosted or independently hosted experiment. Its strongest value is as a new ranking training recipe and a high quality local ceiling, not as a managed service dependency.

## New techniques that may matter more than model choice

### Contextual retrieval

Anthropic’s contextual retrieval method prepends chunk specific explanatory context before building both the dense embedding and the BM25 representation. Anthropic reported that contextual embeddings plus contextual BM25 plus reranking reduced its top 20 retrieval failure rate from 5.7 percent to 1.9 percent across its tests. The project’s ambiguous turn failure suggests a related treatment is likely more valuable than an isolated model swap.

There are two versions worth separating in the experiment:

1. Provider native contextualization, using Voyage Context 4.
2. Locally generated context headers, using a fixed summarizer or deterministic conversation metadata, then embedding and indexing the enriched text.

The second version must keep the enriched text out of the answer context if it is only an indexing aid. Otherwise the test changes both retrieval and answer evidence.

### Late chunking

Late chunking encodes the long document before pooling individual chunks, so each chunk vector carries information from the wider document. Jina documents late chunking support on Jina Embeddings v3, and the method is described in the original paper. This is a promising local proxy for Voyage Context 4, but it requires a new indexing path and careful chunk offsets.

### Listwise reranking

Jina Reranker v3 and v3.5 represent a meaningful change from conventional pairwise cross encoders. The reranker sees the entire candidate list and can use candidate relationships. This is especially relevant for the project’s invalid extra document metric and exact document set coverage, where choosing several individually plausible but redundant memories is a common failure mode.

## Recommended experiment matrix

### Phase 0, frozen protocol

Use the same source snapshot, question order, chunk boundaries, query identities, hybrid fusion, output k, and gold annotations. Runtime must not read gold fields. Pre register the prediction before measuring. Store the model identifier, revision, prompt format, dimensions, dtype, host fingerprint, token volume, candidate pool, and latency.

### Phase 1, candidate recall screen

Compare these first stage arms:

1. Current Voyage 4 Large.
2. Voyage Context 4, grouped by conversation or source document.
3. Jina Embeddings v5 Text Small, retrieval adapter.
4. Qwen3 Embedding 0.6B, instruction aware.
5. Cohere Embed v4.
6. Gemini Embedding 2 only if a cloud multimodal control is desired.

At this phase, measure gold in pool at 20, 50, and 100, plus hit at 5, hit at 10, MRR, exact set coverage, invalid extras, and embedding latency. Do not rerank yet. This isolates whether a model actually raises the ceiling available to every second stage.

### Phase 2, reranker screen

Fix the strongest two first stage arms from Phase 1 and compare:

1. Voyage Rerank 2.5.
2. Cohere Rerank v4 Pro.
3. Jina Reranker v3.5.
4. Qwen3 Reranker 0.6B.
5. zerank 2, self hosted if the host can safely run it.

Use candidate pools of 25, 50, and 100. Keep final k fixed. The primary measure is gold recovery at final k conditional on gold being present in the pool. The secondary measure is total gold in pool, which diagnoses whether a reranker is being asked to solve an upstream failure.

### Phase 3, local quality ceiling

Run Qwen3 Embedding 4B, Qwen3 Reranker 4B, zembed 1, and zerank 2 only on a dedicated GPU or a separately bounded host. Do not relax the current VPS2 production guard merely to fit a benchmark. The purpose of this phase is to determine whether the quality frontier justifies future infrastructure.

## Decision rules

Promote a model only when it improves the frozen gold metric on the held out confirmation set, does not create a material invalid extra regression, and stays within the declared latency and token budget. Treat benchmark leaderboard differences as priors, not proof.

For this project, the most valuable result would be a model or method that raises gold in pool and preserves exact document set coverage. A reranker that only improves the ordering of already retrieved gold items is useful, but it should not be mistaken for a solution to the dominant candidate recall gap.

## Sources

1. [Voyage text embeddings documentation](https://docs.voyageai.com/docs/embeddings)
2. [Voyage contextualized chunk embeddings](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
3. [Voyage pricing and free tier](https://docs.voyageai.com/docs/pricing)
4. [Cohere Embed v4 release](https://docs.cohere.com/changelog/embed-multimodal-v4)
5. [Cohere Rerank v4 release](https://docs.cohere.com/changelog/rerank-v4.0)
6. [Cohere Rerank documentation](https://docs.cohere.com/docs/rerank)
7. [Google Gemini Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
8. [Qwen3 Embedding 8B model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
9. [Qwen3 Reranker 4B model card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
10. [Jina Embeddings v5 Text Small](https://jina.ai/models/jina-embeddings-v5-text-small/)
11. [Jina Reranker v3.5](https://jina.ai/models/jina-reranker-v3.5/)
12. [Jina late chunking model reference](https://jina.ai/models/jina-embeddings-v3/)
13. [Nomic Embed v2 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
14. [Google EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)
15. [ZeroEntropy zembed 1 model card](https://huggingface.co/zeroentropy/zembed-1-embedding)
16. [ZeroEntropy zerank 2 model card](https://huggingface.co/zeroentropy/zerank-2-reranker)
17. [ZeroEntropy migration guide](https://www.zeroentropy.dev/articles/zeroentropy-migration-guide/)
18. [MTEB model leaderboard](https://leaderboard.mteb.org/models)
19. [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
20. [Late Chunking paper](https://arxiv.org/abs/2409.04701)
