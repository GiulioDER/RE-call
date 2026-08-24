from recall.normalise import normalise


def test_collapses_a_newline():
    assert normalise("a\nb") == "a b"


def test_collapses_repeated_spaces():
    assert normalise("a    b") == "a b"


def test_strips_the_ends():
    assert normalise("  a b  ") == "a b"
