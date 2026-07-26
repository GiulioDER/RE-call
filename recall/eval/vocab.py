"""Corpus vocabulary statistics — candidate predictors of the cloud-embedder gap.

FINDINGS §8 established a *conditional* rule — a cloud embedder is worth +0.28 on a corpus whose
vocabulary is idiosyncratic and +0.02 on ordinary technical English — but left "idiosyncratic"
unoperationalised, which makes the rule unusable by a reader and uncritiqueable by a reviewer.

This module computes the candidate operationalisations. It deliberately does NOT decide which one
is right: that is an empirical question answered in `recall.eval.gap_study`, against a null model
(the local embedder's own score), because the gap and the local score are mechanically related and
any predictor that does not beat that baseline has explained nothing.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable

#: Word pattern for vocabulary statistics. Underscores are kept inside words because identifiers
#: (`signal_filter`, `close_all`) are exactly the kind of token this study is about.
_WORD = re.compile(r"\w+", re.UNICODE)


def word_types(texts: Iterable[str]) -> set[str]:
    """The distinct lowercased words in `texts`.

    Case-folded because the embedders under study are uncased, so `RiskBook` and `riskbook` are
    one vocabulary item to the model and must be one item here too.
    """
    types: set[str] = set()
    for text in texts:
        types.update(m.group(0).lower() for m in _WORD.finditer(text))
    return types


def oov_rate(
    texts: Iterable[str],
    tokenize: Callable[[str], list[str]],
    *,
    min_pieces: int = 2,
) -> float:
    """Share of word *types* that `tokenize` shatters into `min_pieces` or more subwords.

    Type-weighted, not token-weighted, and that is a claim rather than a detail: a corpus is
    idiosyncratic because of how many unusual words it uses, not how often it repeats one of them.
    Token-weighting would let a single frequent codename dominate the statistic and would rank a
    corpus that says `RMK_KILL` ten thousand times above one that uses a thousand distinct
    codenames once each — the opposite of what the gap is thought to track.

    `tokenize` is injected: the real segmenter is a model artifact (the embedder's own tokenizer),
    and which model's tokenizer you ask is part of the experiment, not a constant.
    """
    types = word_types(texts)
    if not types:
        return float("nan")
    shattered = sum(1 for word in types if len(tokenize(word)) >= min_pieces)
    return shattered / len(types)
