# Individual contribution report

## Thông tin

- Họ và tên: Trần Gia Thành
- Mã học viên: 2A202602626
- Nhóm: Transformer
- Repository/branch: https://github.com/ringge/K4-L3B-RAG-Pipeline

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 3 — Convert Markdown | Dùng MarkItDown cho legal; chuyển JSON news sang Markdown có metadata; tránh file rỗng/trùng. | `src/task3_convert_markdown.py`; `2a8f923` | Done |
| Task 5 — Semantic search | Embed query, query ChromaDB, đổi cosine distance sang similarity và trả dense `SearchResult` đã sắp xếp. | `src/task5_semantic_search.py`; `954a35c` | Done |
| Task 7 — Reranking | Hợp nhất dense/BM25 bằng RRF; thêm Jina rerank tùy chọn qua `.env`. | `src/task7_reranking.py`; `9b9ed13`, `8e2b177` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Chuẩn hóa riêng `legal/` và `news/`; chỉ ghi Markdown khi hợp lệ và thay đổi.  
   **Lý do/evidence:** Giữ metadata news, không tạo file rỗng/trùng; đầu ra hiện có 3 legal và 6 news Markdown.  
   **Trade-off:** Cần tránh các file legal trùng stem ở bước thu thập.

2. **Quyết định:** Tách RRF và Jina rerank.  
   **Lý do/evidence:** Cosine score và BM25 score khác thang đo; RRF hợp nhất theo rank, Jina dùng query để xếp lại ứng viên.  
   **Trade-off:** Jina phụ thuộc API/mạng và tăng latency retrieval.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest tests/test_contracts.py -q -k semantic_search`; `pytest tests/test_contracts.py -q -k rrf`; query Jina “Saigon Water Bus khởi hành từ bến nào?”; A/B 18 case trong `reports/RESULT.md`.
- Kết quả trước/sau nếu có: Semantic search và RRF đều `1 passed, 14 deselected`. Jina xếp `news/article_06.md::water-bus` đứng đầu (`0.64153469`), đúng bến Bạch Đằng. Hybrid + RRF/Jina tăng average 4 metric `0.7962 → 0.8019` và relevance `0.9500 → 0.9944`; recall giảm `0.9486 → 0.9133`.
- Lỗi đã phát hiện và cách xử lý: Task 3 xử lý file rỗng hoặc JSON thiếu metadata bằng cách bỏ qua và báo lỗi theo từng nguồn. Với retrieval, giữ dense similarity gốc cho fallback; RRF score chỉ dùng để xếp hạng.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Hybrid chưa nên là mặc định chỉ với mức tăng `0.0057`; Config B mất gold passage G06 và tăng 45,6 ms/câu.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Nạp BM25 corpus từ cùng Chroma collection, kiểm tra citation theo source và chạy lại A/B với mở rộng chunk liền kề cho G18.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Trần Gia Thành
