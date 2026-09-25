"""Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output theo SearchResult, sắp xếp giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    if top_k <= 0:
        return []

    query_vectors = embed_texts([query.strip()])
    if not query_vectors:
        raise ValueError("embed_texts returned no query embedding")

    response = get_collection().query(
        query_embeddings=[query_vectors[0]],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids") or [[]]
    documents = response.get("documents") or [[]]
    metadatas = response.get("metadatas") or [[]]
    distances = response.get("distances") or [[]]

    # Giữ kết quả tốt nhất nếu backend trả về ID trùng lặp.
    results_by_id: dict[str, dict] = {}
    for item_id, content, metadata, distance in zip(
        ids[0], documents[0], metadatas[0], distances[0]
    ):
        if not isinstance(item_id, str) or not item_id.strip():
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if not isinstance(metadata, dict):
            continue

        # Với cosine space của ChromaDB: similarity = 1 - distance.
        score = max(0.0, 1.0 - float(distance))
        result = {
            "id": item_id,
            "content": content,
            "score": score,
            "metadata": metadata,
            "retrieval_method": "dense",
        }
        previous = results_by_id.get(item_id)
        if previous is None or score > previous["score"]:
            results_by_id[item_id] = result

    return sorted(
        results_by_id.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
