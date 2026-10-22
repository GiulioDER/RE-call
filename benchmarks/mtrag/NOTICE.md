# Third party text in `benchmarks/mtrag/`

This directory is a research harness for the MTRAG benchmark. It is not part of any package RE-call
ships. One prompt in it is quoted from the MTRAG paper. Sources and licences were read on
2026-10-20.

| | |
|---|---|
| Text | the `official` prompt in `generation.py` (`PROMPTS["official"]`), the evaluation instruction of the MTRAG baselines |
| Source | Y. Katsis, S. Rosenthal, K. Fadnis, C. Gunasekara, Y.-S. Lee, L. Popa, V. Shah, H. Zhu, D. Contractor and M. Danilevsky, "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems", arXiv:2501.03468, 2025, Appendix D.2 "Model invocation" |
| Licence of the paper | CC BY-SA 4.0, as declared on its arXiv page (https://arxiv.org/abs/2501.03468) |
| Changes | the paper's curly quotation marks are straight ones here; the words are unchanged. How the passages and turns are laid out around the instruction is described in the `generation.py` comments |

The benchmark data the harness reads is IBM's release, https://github.com/IBM/mt-rag-benchmark,
which declares Apache-2.0 for the repository. It is read from a local checkout passed as
`--mtrag-root` and is not copied into this repository.
