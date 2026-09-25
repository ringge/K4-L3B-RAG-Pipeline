# Individual contribution report

Mỗi thành viên copy template này thành:

```text
reports/<student-id>-<short-name>.md
```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---

## Thông tin

- Họ và tên: Nguyễn Minh Thái
- Mã học viên: 2A202602726
- Nhóm: Transformer
- Repository/branch: https://github.com/ringge/K4-L3B-RAG-Pipeline

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 2 - Crawl news | Tích hợp Firecrawl async, đọc `FIRECRAWL_API_KEY` từ `.env`, lưu JSON đủ metadata và kiểm tra kết quả crawl. | `src/task2_crawl_news.py`; commit `8bf6a0e`, `554f2da` | Done |
| Task 6 - Lexical search | Xây dựng BM25 index, trả `SearchResult` đúng contract, xử lý query/corpus rỗng và corpus nhỏ. | `src/task6_lexical_search.py`; commit `ac2575f` | Done |
| Task 8 - PageIndex fallback | Upload tài liệu qua PageIndex, chuyển Markdown sang PDF, cache document ID, timeout và parse citation thành kết quả PageIndex. | `src/task8_pageindex_vectorless.py`; commit `d5569d2` | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Dùng Firecrawl async cho Task 2 và nạp API key từ `.env`.  
   **Lý do/evidence:** Firecrawl trả Markdown trực tiếp, phù hợp với module chuẩn hóa tiếp theo; key không hard-code trong mã nguồn.  
   **Trade-off:** Phụ thuộc dịch vụ ngoài và cần API key hợp lệ.

2. **Quyết định:** Dùng cache mapping `source -> document_id` và chuyển Markdown sang PDF trước khi upload PageIndex.  
   **Lý do/evidence:** PageIndex 0.2.8 nhận PDF và cache giúp chạy lại không upload trùng.  
   **Trade-off:** Tạo thêm thư mục PDF tạm và chất lượng trích xuất phụ thuộc quá trình chuyển đổi.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `python -m py_compile`; `pytest -q tests/test_contracts.py -k lexical_search`; kiểm thử mock PageIndex upload/cache/citation; chạy `python src/task2_crawl_news.py`.
- Kết quả trước/sau nếu có: Test lexical đạt `1 passed`; mock PageIndex đạt; Firecrawl tạo thành công `data/landing/news/article_01.json` với đủ bốn trường bắt buộc.
- Lỗi đã phát hiện và cách xử lý: `.env` chưa được load khiến Firecrawl báo thiếu API key; đã thêm `load_dotenv`. SDK PageIndex cần PDF và không có timeout ở method; đã thêm chuyển đổi PDF và wrapper timeout.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Task 2 và Task 8 phụ thuộc API bên ngoài; PageIndex trả kết quả an toàn rỗng khi provider lỗi, và việc parse citation phụ thuộc response của SDK.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: bổ sung test mock chính thức cho Task 2/8 và test timeout, đồng thời thêm nhiều URL news hợp lệ để tạo đủ corpus reproducible.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Nguyễn Minh Thái
