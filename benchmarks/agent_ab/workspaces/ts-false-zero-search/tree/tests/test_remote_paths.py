from recall.remote_config import DEFAULT_CORPUS_ROOT


def test_default_corpus_root():
    assert DEFAULT_CORPUS_ROOT == "/home/sentiment/recall-repos"
