# Individual contribution report

## Thông tin

- Họ và tên: Trần Kim Phương
- Mã học viên: 2A202602565
- Nhóm: Transformer
- Repository/branch: https://github.com/ringge/K4-L3B-RAG-Pipeline

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Thu thập tài liệu du lịch và OCR PDF | Chọn, tải 3 PDF công khai; xây dựng OCR từng trang bằng vision model, lưu Markdown theo trang/ghép trang và tiếp tục sau gián đoạn. | `e0cdb33` · `src/task1_collect_legal_docs.py`, `src/aiOCR.py`, `tests/test_ai_ocr.py`, `data/landing/legal/` | Done |
| Bổ sung nguồn tin du lịch | Thêm URL bài viết cho danh sách đầu vào của crawler; không nhận ownership toàn bộ crawler/chuyển đổi Markdown. | `554f2da` · `src/task2_crawl_news.py` | Done |
| Chunking và chỉ mục vector | Triển khai chia chunk, embedding Jina v3, upsert ChromaDB; chỉ embedding lại chunk thay đổi; tạo dữ liệu chuẩn hóa và chỉ mục trong commit. | `62e8d63` · `src/task4_chunking_indexing.py`, `tests/test_jina_embeddings.py`, `tests/test_task4_incremental.py` | Done |
| Tập câu hỏi đối chiếu | Soạn 18 câu hỏi kèm đáp án, ngữ cảnh mong đợi và nguồn cho keyword, semantic, cross-source; đây là dữ liệu tham chiếu, không phải kết quả đo. | `917d13d` · `group_project/evaluation/golden_dataset.json` | Done |
| Truy xuất, sinh câu trả lời và giao diện | Ghép dense/lexical bằng RRF, fallback PageIndex theo điểm dense; tích hợp LLM OpenAI/Gemini/Anthropic, citation và từ chối an toàn; hiển thị nguồn trong Streamlit. | `8134d53` · `src/task9_retrieval_pipeline.py`, `src/task10_generation.py`, `app.py`, `tests/test_generation.py`, `tests/test_app.py` | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Hỗ trợ Jina embeddings v3 (`text-matching`) với collection Chroma riêng và ID chunk ổn định.  
   **Lý do/evidence:** `62e8d63`; `src/task4_chunking_indexing.py` dùng chung `embed_texts()` khi lập chỉ mục/tìm kiếm và bỏ qua chunk không đổi khi chạy lại.  
   **Trade-off:** Cần API key, kết nối mạng và chi phí dịch vụ; phải duy trì collection tách biệt với model cục bộ.

2. **Quyết định:** Dùng điểm cosine của dense retrieval để kích hoạt PageIndex fallback, không dùng điểm RRF.  
   **Lý do/evidence:** `8134d53`; `src/task9_retrieval_pipeline.py` lấy `dense[0]["score"]` trước khi so threshold; RRF có thang điểm khác.  
   **Trade-off:** Ngưỡng cố định `0.3` phụ thuộc phân phối điểm embedding; fallback có thể tăng độ trễ.

## Kiểm thử và kết quả

- Test đã dùng: `uv run pytest -q tests/test_ai_ocr.py tests/test_jina_embeddings.py tests/test_task4_incremental.py tests/test_generation.py tests/test_app.py` — 15 passed (25/9/2026). Các test kiểm tra OCR/khôi phục tiến độ, thứ tự embedding và tái lập chỉ mục, provider LLM/citation/từ chối an toàn, hiển thị nguồn.
- Kết quả trước/sau: commit `62e8d63` thêm kiểm tra chunk không đổi để tránh embedding lại khi chạy pipeline; chưa có số liệu đo hiệu năng hoặc điểm chất lượng end-to-end được xác nhận từ các commit của tôi.
- Lỗi đã phát hiện và cách xử lý: OCR gián đoạn được lưu theo trang để chạy tiếp (`e0cdb33`); lỗi provider hoặc truy xuất trong generation dẫn tới câu trả lời từ chối an toàn thay vì nội dung không có nguồn (`8134d53`).

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: prompt yêu cầu citation nhưng `generate_with_citation()` chưa kiểm tra ID citation trong văn bản trả về có khớp với nguồn đã truy xuất.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: kiểm tra citation ở đầu ra và từ chối câu trả lời khi dẫn nguồn không tồn tại trong tập chunks truy xuất.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/9/2026
- Tên thành viên: Trần Kim Phương
