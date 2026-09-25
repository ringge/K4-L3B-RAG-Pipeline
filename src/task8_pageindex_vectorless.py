"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"
PDF_CACHE_DIR = Path(__file__).parent.parent / "pageindex_pdfs"
REQUEST_TIMEOUT = 60


def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _run_with_timeout(function: Any, *args: Any, **kwargs: Any) -> Any:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(function, *args, **kwargs)
        return future.result(timeout=REQUEST_TIMEOUT)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _markdown_to_pdf(markdown_path: Path) -> Path:
    """Create a stable PDF input for PageIndex's PDF-only upload endpoint."""
    from fpdf import FPDF

    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output = PDF_CACHE_DIR / f"{markdown_path.stem}.pdf"
    if output.exists():
        return output

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.exists():
        pdf.add_font("ArialUnicode", fname=str(font_path))
        pdf.set_font("ArialUnicode", size=10)
    else:
        pdf.set_font("Helvetica", size=10)
    text = markdown_path.read_text(encoding="utf-8")
    pdf.multi_cell(0, 5, text.encode("latin-1", errors="replace").decode("latin-1"))
    pdf.output(str(output))
    return output


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        return

    try:
        from pageindex import PageIndexClient

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        cache = _load_cache()
        for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
            source = path.relative_to(STANDARDIZED_DIR).as_posix()
            if source in cache:
                continue
            response = _run_with_timeout(
                client.submit_document,
                str(_markdown_to_pdf(path)),
            )
            document_id = response.get("doc_id") if isinstance(response, dict) else None
            if not isinstance(document_id, str) or not document_id:
                continue
            cache[source] = document_id
            CACHE_PATH.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except (Exception, TimeoutError):
        return


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip() or not PAGEINDEX_API_KEY:
        return []

    try:
        from pageindex import PageIndexClient

        cache = _load_cache()
        if not cache:
            upload_documents()
            cache = _load_cache()
        if not cache:
            return []

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        response = _run_with_timeout(
            client.chat_completions,
            [{"role": "user", "content": query}],
            False,
            list(cache.values()),
            enable_citations=True,
        )
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        citations = message.get("citations", []) if isinstance(message, dict) else []
        if not isinstance(content, str) or not content.strip():
            return []

        source_by_id = {document_id: source for source, document_id in cache.items()}
        if not isinstance(citations, list) or not citations:
            citations = [{}]

        results = []
        for rank, citation in enumerate(citations[:top_k], 1):
            citation = citation if isinstance(citation, dict) else {}
            text = citation.get("quote") or citation.get("content") or content
            document_id = citation.get("doc_id") or citation.get("document_id")
            source = source_by_id.get(document_id, next(iter(cache)))
            results.append({
                "id": f"pageindex::{source}::{rank}",
                "content": str(text),
                "score": 1.0 / rank,
                "metadata": {
                    "source": source,
                    "title": Path(source).stem,
                    "doc_type": "legal" if source.startswith("legal/") else "news",
                    "url": None,
                    "chunk_index": rank - 1,
                },
                "retrieval_method": "pageindex",
            })
        return results
    except (Exception, TimeoutError):
        return []


if __name__ == "__main__":
    upload_documents()
