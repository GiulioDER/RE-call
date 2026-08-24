from recall.uploads import store_memo


def test_store_memo_writes_the_text():
    path = store_memo("job-1", "hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_store_memo_is_keyed_by_job():
    a = store_memo("job-a", "a")
    b = store_memo("job-b", "b")
    assert a != b
