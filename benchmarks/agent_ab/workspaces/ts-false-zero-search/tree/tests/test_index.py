from recall.index import index


def test_index_dedupes():
    assert index(["b", "a", "b"]) == ["a", "b"]
