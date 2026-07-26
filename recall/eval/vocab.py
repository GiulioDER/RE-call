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

import random
import re
from collections.abc import Callable, Iterable

#: Word pattern for vocabulary statistics. Underscores are kept inside words because identifiers
#: (`signal_filter`, `close_all`) are exactly the kind of token this study is about.
_WORD = re.compile(r"\w+", re.UNICODE)


def word_tokens(texts: Iterable[str]) -> list[str]:
    """Every lowercased word occurrence in `texts`, in order.

    Case-folded because the embedders under study are uncased, so `RiskBook` and `riskbook` are
    one vocabulary item to the model and must be one item here too.
    """
    return [m.group(0).lower() for text in texts for m in _WORD.finditer(text)]


def word_types(texts: Iterable[str]) -> set[str]:
    """The distinct lowercased words in `texts`."""
    return set(word_tokens(texts))


#: The local embedder under study. Kept in step with `FastEmbedEmbedder`'s default, because the
#: point of the metric is to ask *this* model what it does and does not have a word for.
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


def bge_encoder(model_name: str = DEFAULT_LOCAL_MODEL) -> Callable[[str], list[str]]:
    """A raw subword encoder for `model_name`, for feeding to `subword_pieces`.

    Uses `tokenizers` directly rather than `transformers`: it is already a fastembed dependency,
    so asking the local embedder how it segments text costs no new install. The returned callable
    emits the tokenizer's raw output, special tokens included — stripping them is
    `subword_pieces`' job, and keeping the two separate is what let the stripping be tested
    without a network round trip.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model_name)
    return lambda word: tokenizer.encode(word).tokens


#: Special tokens a sentence-embedding tokenizer wraps around its input. These are not vocabulary
#: evidence — they are present for every word, including words the model knows perfectly well.
SPECIAL_TOKENS = frozenset({"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "<s>", "</s>", "<pad>"})


def subword_pieces(
    word: str,
    encode: Callable[[str], list[str]],
    *,
    special: frozenset[str] = SPECIAL_TOKENS,
) -> list[str]:
    """The subword pieces `encode` splits `word` into, with special tokens removed.

    bge-small is BERT-family, so a raw `encode("crash")` returns `["[CLS]", "crash", "[SEP]"]` —
    three pieces for a word the model knows. Left in, every type in every corpus clears any
    `min_pieces` threshold and `oov_rate` returns 1.0 everywhere: a number that looks fine, varies
    plausibly with nothing, and would make the whole study meaningless without ever erroring.
    """
    return [piece for piece in encode(word) if piece not in special]


def oov_rate(
    texts: Iterable[str],
    tokenize: Callable[[str], list[str]],
    *,
    min_pieces: int = 2,
    token_budget: int | None = None,
    seed: int = 0,
) -> float:
    """Share of word *types* that `tokenize` shatters into `min_pieces` or more subwords.

    Type-weighted, not token-weighted, and that is a claim rather than a detail: a corpus is
    idiosyncratic because of how many unusual words it uses, not how often it repeats one of them.
    Token-weighting would let a single frequent codename dominate the statistic and would rank a
    corpus that says `RMK_KILL` ten thousand times above one that uses a thousand distinct
    codenames once each — the opposite of what the gap is thought to track.

    `tokenize` is injected: the real segmenter is a model artifact (the embedder's own tokenizer),
    and which model's tokenizer you ask is part of the experiment, not a constant.

    `token_budget` is the size control, and it is not optional for comparing corpora. Type counts
    grow with corpus length (Heaps' law): common vocabulary saturates while rare words keep
    arriving, so a larger corpus scores as more idiosyncratic even when its per-document
    composition is identical. Measured on a synthetic corpus of constant composition, the
    uncontrolled statistic moves 0.33 -> 0.91 across a 20x size range — comfortably larger than
    the effect this study is trying to detect. Holding TOKENS fixed is what makes two corpora
    comparable, because tokens are what types are a function of.
    """
    if token_budget is None:
        types = word_types(texts)
    else:
        tokens = word_tokens(texts)
        if len(tokens) > token_budget:
            tokens = random.Random(seed).sample(tokens, token_budget)
        types = set(tokens)
    if not types:
        return float("nan")
    shattered = sum(1 for word in types if len(tokenize(word)) >= min_pieces)
    return shattered / len(types)
