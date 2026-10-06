import numpy as np

from er.pipeline import Cache, query_chunks


def test_query_chunks_never_split_a_query():
    q = np.array([5, 5, 5, 2, 2, 9, 9, 9, 9, 1])
    chunks = list(query_chunks(q, 4))
    assert chunks[0][0] == 0 and chunks[-1][1] == len(q)
    for (a, b), (c, _) in zip(chunks, chunks[1:]):
        assert b == c and q[b - 1] != q[b]
    assert list(query_chunks(np.array([], dtype=int), 4)) == []


def test_cache_keys_follow_settings():
    assert Cache.key("pairs", "k", 16) == Cache.key("pairs", "k", 16)
    assert Cache.key("pairs", "k", 16) != Cache.key("pairs", "k", 12)
