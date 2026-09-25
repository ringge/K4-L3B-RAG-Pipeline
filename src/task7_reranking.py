"""Task 7 — Reciprocal Rank Fusion và Jina reranking.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Jina là bước rerank nâng cao, cần cả query và danh sách ứng viên. Vì contract
``rerank_rrf(ranked_lists, top_k, k)`` không nhận query, hai bước được tách
thành hai hàm để không phá vỡ giao diện chung của project.
"""

import os

import requests
from dotenv import load_dotenv


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists bằng RRF và trả hybrid SearchResult."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                continue
            if item_id in seen_in_list:
                continue

            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items.setdefault(item_id, item)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        result = items[item_id].copy()
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


def rerank_with_jina(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Rerank các SearchResult theo query bằng Jina Reranker API.

    ``candidates`` thường là tập ứng viên đã được RRF hợp nhất. Hàm giữ lại
    toàn bộ content/metadata gốc, chỉ thay score bằng relevance score của Jina.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    if top_k <= 0 or not candidates:
        return []

    load_dotenv()
    api_key = os.getenv("JINA_API_KEY", "").strip()
    rerank_url = os.getenv("JINA_RERANK_URL", "").strip()
    rerank_model = os.getenv("JINA_RERANK_MODEL", "").strip()
    if not api_key or not rerank_url or not rerank_model:
        raise RuntimeError(
            "JINA_API_KEY, JINA_RERANK_URL, and JINA_RERANK_MODEL must be configured"
        )

    unique_candidates: list[dict] = []
    seen_ids: set[str] = set()
    for item in candidates:
        item_id = item.get("id")
        content = item.get("content")
        if (
            not isinstance(item_id, str)
            or not item_id.strip()
            or item_id in seen_ids
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        seen_ids.add(item_id)
        unique_candidates.append(item)

    if not unique_candidates:
        return []

    payload = {
        "model": rerank_model,
        "query": query.strip(),
        "top_n": min(top_k, len(unique_candidates)),
        "documents": [item["content"] for item in unique_candidates],
    }
    response = requests.post(
        rerank_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    jina_results = data.get("results")
    if not isinstance(jina_results, list):
        raise RuntimeError("Jina response does not contain a results list")

    results: list[dict] = []
    for jina_item in jina_results:
        index = jina_item.get("index")
        score = jina_item.get("relevance_score")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(unique_candidates)
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
        ):
            continue

        result = unique_candidates[index].copy()
        result["score"] = float(score)
        result["retrieval_method"] = "hybrid"
        results.append(result)

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    print("Use rerank_rrf() to fuse rankings, then rerank_with_jina() with a query.")
