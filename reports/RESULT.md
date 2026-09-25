# Kết quả đánh giá A/B RAG

## Run information

| Trường | Giá trị |
| --- | --- |
| Ngày chạy retrieval và generation | 2026-09-25 (UTC) |
| Framework | `group_project/evaluation/run_ab.py`; ChromaDB 1.5.9, rank-bm25 0.2.2, OpenAI SDK 2.54.0 |
| Evaluator model | `OPENAI_MODEL=gemini-3.8-flash-high` qua `OPENAI_BASE_URL`/`OPENAI_API_KEY`, temperature 0, một lần chấm cho mỗi answer |
| Generator model | Cùng `OPENAI_MODEL=gemini-3.8-flash-high` và endpoint cho cả A/B |
| Embedding model | Jina `jina-embeddings-v3`, `task=text-matching`, 1024 chiều; collection `rag_documents_jina_v3` |
| Corpus commit | `8134d53bcc99e2d2de00b1a580d688e95f86fdb2` |
| SHA-256 corpus Markdown chuẩn hóa | `7341100daa1bbbf71cc04f5229b9a5cc951818be2230159f5cc2806c826aba45` |
| Golden dataset | 18 case: 6 keyword, 6 semantic, 6 cross-source; SHA-256 `997b8a9d15691d1df1afa807a886766048819da4f501331ed944dd9ab356add5` |
| Index | 1.727 chunks trong Chroma; cả 18 `expected_context` đều tìm thấy nguyên văn trong file `source` tương ứng |
| `top_k` | 5; mỗi retriever lấy 10 ứng viên trước khi cắt top 5 |
| Fallback threshold | `0.3` trong ứng dụng; PageIndex fallback **tắt ở cả hai arm** để chỉ thay retrieval strategy. Threshold không tham gia A/B này; chưa có calibration out-of-domain. |
| Generator/prompt | Cùng `SYSTEM_PROMPT`, `format_context`, `reorder_for_llm`, temperature `0.3`, top-p `0.9` của `src/task10_generation.py` |

## Configurations

- **Config A — dense-only:** một query embedding Jina; Chroma cosine search; lấy 5 chunk đầu theo cosine similarity.
- **Config B — hybrid + RRF:** dùng **cùng kết quả dense** và BM25 trên chính 1.727 chunk của collection Chroma; lấy 10 ứng viên mỗi nhánh, fuse một lần với `k=60`, rồi lấy 5 chunk. BM25 corpus được nạp tường minh trong script đánh giá; `src/task6_lexical_search.py` hiện khởi tạo `CORPUS=[]` và app chưa nạp corpus này, nên đây là cấu hình hybrid đã khởi tạo cho thí nghiệm, chưa phải hành vi hiện tại của UI.

Hai arm dùng cùng 18 câu hỏi, corpus, embedding, `top_k`, fallback setting, generator model, evaluator model và prompt. Script dùng bản sao tạm của Chroma để không sửa database được track.

## Metric definitions

- **Context recall (đã đo):** số 5-gram chuẩn hóa từ `expected_context` xuất hiện trong hợp các chunk được lấy, chia cho tổng số 5-gram của gold passage. Đây là lexical evidence coverage, không phải Ragas context recall; đoạn cùng nghĩa nhưng khác chữ có thể bị chấm thấp.
- **Context precision (đã đo):** tỷ lệ trong 5 chunk có ít nhất ba 5-gram chung với gold passage. Đây là passage focus proxy, không phải Ragas context precision; kết quả phụ thuộc độ dài và phạm vi gold passage.
- **Faithfulness:** cùng `OPENAI_MODEL` làm judge, chấm từ 0 đến 1 theo tỷ lệ claim của answer được chính retrieved context hỗ trợ. Judge nhận question, context và answer.
- **Answer relevance:** cùng judge và cùng rubric cho A/B, chấm từ 0 đến 1 theo mức trả lời đúng và đủ câu hỏi. Judge không kiểm tra citation có đúng tài liệu được nêu trong question; chúng tôi kiểm tra nguồn riêng bằng tay.

Hai metric context là proxy xác định theo 5-gram; hai metric answer là điểm LLM judge, **không phải bốn metric Ragas**. Judge dùng cùng model với generator nên có nguy cơ đồng thuận thiên lệch. Mỗi arm chạy một lần, chưa có lặp lại hoặc khoảng tin cậy.

## Overall scores

Trung bình macro trên 18 case; delta = B−A. Kết quả từng case, answer, lý do của judge, retrieved IDs và latency có trong `group_project/evaluation/ab_results.json`.

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.9972 | 1.0000 | +0.0028 |
| Answer relevance | 0.9500 | 0.9944 | +0.0444 |
| Context recall (5-gram proxy) | 0.9486 | 0.9133 | −0.0354 |
| Context precision (5-gram proxy) | 0.2889 | 0.3000 | +0.0111 |
| **Average of four metrics** | **0.7962** | **0.8019** | **+0.0057** |

## A/B comparison

Config B nhỉnh hơn về answer relevance (+0.0444) vì G18 có bằng chứng “bến Bạch Đằng”: B trả lời đúng, A từ chối do chỉ lấy chunk giới thiệu. B cũng tăng context precision nhẹ (+0.0111), nhưng A giữ gold passage tốt hơn theo recall (+0.0354 so với B). Faithfulness gần trần ở cả hai arm; số +0.0028 không đủ để khẳng định ưu thế. Average của B chỉ hơn **0.0057**, trong khi B mất hẳn gold passage của G06. Không chọn B làm mặc định chỉ theo average này.

Đọc answer và citation cho thấy một lỗi ngoài bốn điểm: cả A lẫn B trả lời đúng tháng ở G06 nhưng đều cite `legal/camnang-dulich-saigon.md::chunk-236`, không phải bài Green SM được hỏi; A cũng mắc lỗi nguồn tương tự ở G16. Trên 18 case, 16 answer của A và 17 answer của B cite ít nhất một chunk từ file gold; đây chỉ là kiểm tra đúng file, không xác nhận citation hỗ trợ đúng claim. G14 có gold question nhắc số thực tế 2019 nhưng `expected_context` chỉ chứa mục tiêu 2025; cả hai answer nêu 18 triệu và giải thích thiếu dữ liệu 2019, là phản ứng thận trọng với context.

Latency retrieval trung bình A **842.8 ms**, B **888.4 ms**, tăng **45.6 ms/câu** (mỗi câu có một request Jina, B dùng lại embedding/dense result). Generation latency quan sát trung bình A **6,175 ms**, B **4,549 ms**; do chạy A rồi B tuần tự qua endpoint mạng và chỉ một lần/case, không quy chênh lệch này cho retrieval strategy. Mỗi arm dùng 18 request generation và 18 request judge; B thêm BM25/RRF local nhưng không thêm embedding API request. Token usage và tiền API không được log, nên chưa định lượng chênh lệch cost. Context có thể khác độ dài dù cùng `top_k`.

## Worst performers

Ba arm-case có trung bình bốn điểm thấp nhất; phải đọc answer và gold evidence để xác định lỗi.

| # | Case / question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | G06 — Mùa khô Sài Gòn theo Green SM? | B | 1.0000 | 1.0000 | 0.0000 | 0.0000 | Retrieval + generation citation | BM25/RRF đẩy `news/article_06.md::chunk-2` (gold Green SM, A xếp #2) khỏi top 5; B lấy các đoạn mùa khô chung và Đà Nẵng. B vẫn có đúng khoảng tháng từ `legal/camnang-dulich-saigon.md::chunk-236`, nên answer đúng nhưng cite **sai nguồn được hỏi**. A có gold passage nhưng generator cũng chọn citation từ tài liệu legal. |
| 2 | G18 — Saigon Water Bus khởi hành từ bến nào? | A | 1.0000 | 0.3000 | 0.5074 | 0.4000 | Chunking/retrieval | A lấy `news/article_06.md::chunk-59` giới thiệu Water Bus nhưng bỏ `chunk-60` chứa “bến Bạch Đằng”; answer A nói không đủ thông tin. B có chunk-60, trả lời đúng, relevance 1.0 và recall 0.9706. Gold evidence nằm qua ranh giới chunk. |
| 3 | G07 — Quỹ cộng đồng hỗ trợ gì? | B | 1.0000 | 1.0000 | 0.6903 | 0.2000 | Retrieval / metric scope | B bỏ `legal/sotay-dulich-congdong-vietnam.md::chunk-25` chứa phần mở đầu gold passage và thay bằng chunk cùng chủ đề khác đoạn. `chunk-26` vẫn chứa cầu, đường, điện, y tế, giáo dục; answer B đúng. Recall proxy giảm vì gold passage gồm phần mở đầu không thiết yếu cho câu trả lời. |

G14 cũng cần sửa dữ liệu đánh giá: question đòi đối chiếu năm 2019 nhưng gold passage chỉ có mục tiêu 2025. A relevance 0.8, B 0.9 phản ánh thiếu bằng chứng trong context, không đủ để kết luận generator kém.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Nạp BM25 corpus trong app từ cùng collection Chroma; khi question chỉ rõ nguồn, ưu tiên/giới hạn retrieval theo source và kiểm tra citation cuối cùng có thuộc nguồn đó. | G06: B mất `article_06::chunk-2`; cả A/B cite tài liệu legal thay Green SM; G16 A cũng cite sai nguồn. App hiện để `CORPUS=[]`. | Hybrid chạy thật trong UI và citation đúng nguồn; có thể tăng CPU/latency hoặc bỏ lỡ nguồn khi nhận diện sai. | Giữ 18 case/commit, xác nhận G06 lấy và cite Green SM, G16 cite Green SM, G18 vẫn lấy chunk-60; so bốn metric, source match và latency với baseline này. |
| 2 | Thử mở rộng 1 chunk liền kề quanh hit được chọn, kèm giới hạn token và khử trùng lặp. | G18: A lấy chunk-59 nhưng câu trả lời ở chunk-60. | Tăng coverage cho thông tin bị cắt qua ranh giới; có thể giảm precision và tăng token/cost. | Ablation A/B trên cùng 18 case với neighbor expansion là biến duy nhất; kiểm tra G18 có “bến Bạch Đằng”, bốn metric, context token và latency/cost. |
| 3 | Rút `expected_context` thành bằng chứng tối thiểu; thêm đoạn 2019 vào gold của G14 hoặc sửa câu hỏi, và chấm đúng nguồn/citation bên cạnh bốn metric. | G07 thiếu mở đầu gold nhưng answer đúng; G14 hỏi 2019 mà gold chỉ có 2025; G06 answer đúng số nhưng sai nguồn. | Metric ít phạt thông tin không thiết yếu, phân biệt answer đúng số với đúng nguồn, tránh sửa retrieval theo tín hiệu sai. | Version dataset mới sau khi rà soát 18 case; chạy lại cả A và B trên cùng version, đối chiếu G07, G14, G06 với answer và citation được đọc tay. |

## Reproduction

```bash
.venv/bin/python -m group_project.evaluation.run_ab --stage retrieval
.venv/bin/python -m group_project.evaluation.run_ab --stage complete
pytest tests/test_acceptance.py -q
```

`--stage retrieval` ghi từng case, rank IDs, context scores và latency vào `group_project/evaluation/ab_results.json`. `--stage complete` dùng `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` từ `.env` cho cùng generator và evaluator; nó lưu sau mỗi answer để có thể tiếp tục nếu endpoint gián đoạn. Endpoint đã có lúc trả 503, sau đó hoạt động; toàn bộ 36 answer và 36 lần chấm đã hoàn tất.

## Bonus experiments

Chưa chạy bonus; không có delta hoặc latency/cost bonus để báo cáo.
