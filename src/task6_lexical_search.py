"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not CORPUS:
        return []

    query_tokens = query.lower().split()
    if not query_tokens:
        return []

    bm25 = build_bm25_index(CORPUS)
    scores = bm25.get_scores(query_tokens)
    query_vocabulary = set(query_tokens)
    ranked = sorted(
        (
            (max(float(score), 0.0), index, item)
            for index, (score, item) in enumerate(zip(scores, CORPUS))
            if query_vocabulary.intersection(item["content"].lower().split())
        ),
        key=lambda match: (-match[0], match[1]),
    )[:top_k]

    return [
        {
            "id": item["id"],
            "content": item["content"],
            "score": score,
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        }
        for score, _, item in ranked
    ]


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
