# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lưu Thị Ngọc Quỳnh  
**Nhóm:** C401_D3  
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector gần cùng hướng, nên hai câu hoặc hai đoạn văn thường có nội dung hoặc ngữ cảnh khá giống nhau. Trong NLP, điều này thường được hiểu là hai văn bản gần nhau về mặt ngữ nghĩa hơn là chỉ giống từ bề mặt.

**Ví dụ HIGH similarity:**
- Sentence A: `Tài xế cần bằng lái A1 hoặc A2 để đăng ký.`
- Sentence B: `Đăng ký tài khoản Xanh SM Bike cần bằng lái xe máy A1/A2.`
- Tại sao tương đồng: Cả hai câu đều nói về cùng một yêu cầu giấy tờ khi đăng ký tài khoản tài xế.

**Ví dụ LOW similarity:**
- Sentence A: `Quy trình giao hàng gồm nhận đơn, lấy hàng, giao hàng và hoàn thành đơn.`
- Sentence B: `Xe điện VinFast có thể sạc tại trạm công cộng.`
- Tại sao khác: Một câu nói về quy trình vận hành giao hàng, câu còn lại nói về việc sạc xe điện.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity được ưu tiên vì nó đo hướng của vector, tức gần hơn với ý nghĩa ngữ nghĩa của văn bản. Euclidean distance dễ bị ảnh hưởng bởi độ lớn vector, trong khi hai câu có ý giống nhau nhưng độ dài khác nhau vẫn có thể có cosine similarity cao.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`  
> `= ceil((10000 - 50) / (500 - 50))`  
> `= ceil(9950 / 450)`  
> `= ceil(22.11)`  
> `= 23`
>
> *Đáp án:* `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`. Khi overlap tăng lên, số chunk cũng tăng vì cửa sổ trượt đi ít hơn ở mỗi bước. Overlap nhiều hơn giúp giữ ngữ cảnh tốt hơn giữa hai chunk liền kề và giảm khả năng ý quan trọng bị cắt dở.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** XanhSM

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain **XanhSM** vì đây là bộ tài liệu gần với nhu cầu thực tế của một hệ thống knowledge assistant: có FAQ cho khách hàng, hướng dẫn cho tài xế, quy trình giao hàng, chính sách dữ liệu cá nhân, điều khoản chung và tài liệu cho nhà hàng đối tác. Bộ data này cũng phù hợp để thử retrieval vì có nhiều loại câu hỏi khác nhau như quy trình, giấy tờ đăng ký, quyền lợi người dùng, chính sách bảo mật và tài liệu dài dạng điều khoản/pháp lý.

### Data Inventory

> Bộ dữ liệu thực tế được thêm vào thư mục `data/data/data/`. Thư mục hiện có 12 file `.txt`/`.md`; trong báo cáo này tôi tập trung vào 6 tài liệu XanhSM chính để benchmark retrieval, còn các file còn lại được giữ làm tài liệu tham chiếu/phụ trợ.

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `khach_hang.txt` | Website XanhSM / FAQ khách hàng | 39,577 | `category=customer_support`, `audience=customer`, `source=khach_hang.txt` |
| 2 | `tai_xe.txt` | Website XanhSM / FAQ tài xế | 8,830 | `category=driver_policy`, `audience=driver`, `source=tai_xe.txt` |
| 3 | `donhang.txt` | Website XanhSM / quy trình giao hàng | 12,112 | `category=delivery_process`, `audience=driver`, `source=donhang.txt` |
| 4 | `nhahang.txt` | Website XanhSM / hướng dẫn đối tác nhà hàng | 29,583 | `category=merchant_policy`, `audience=merchant`, `source=nhahang.txt` |
| 5 | `Chính sách bảo vệ dữ liệu cá nhân.txt` | Website XanhSM / chính sách dữ liệu cá nhân | 27,417 | `category=privacy_policy`, `audience=all`, `source=Chính sách bảo vệ dữ liệu cá nhân.txt` |
| 6 | `ĐIỀU KHOẢN CHUNG.txt` | Website XanhSM / điều khoản sử dụng | 157,688 | `category=general_terms`, `audience=all`, `source=ĐIỀU KHOẢN CHUNG.txt` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | `customer_support`, `driver_policy`, `privacy_policy` | Giúp lọc đúng loại tài liệu trước khi search, ví dụ câu hỏi về tài xế không nên tìm trong FAQ khách hàng |
| audience | string | `customer`, `driver`, `merchant`, `all` | Hữu ích với bộ dữ liệu có nhiều đối tượng sử dụng khác nhau, giúp giảm nhiễu khi retrieval |
| source | string | `khach_hang.txt`, `tai_xe.txt` | Giúp truy vết chunk gốc, hỗ trợ kiểm tra lại bằng chứng và trích dẫn trong câu trả lời |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu đại diện với `chunk_size=200`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `khach_hang.txt` | FixedSizeChunker (`fixed_size`) | 220 | 199.80 | Trung bình, dễ cắt giữa FAQ và câu trả lời |
| `khach_hang.txt` | SentenceChunker (`by_sentences`) | 174 | 225.40 | Khá, giữ được câu nhưng chưa bám tốt vào heading |
| `khach_hang.txt` | RecursiveChunker (`recursive`) | 303 | 128.78 | Tốt cho FAQ dài, dễ tách theo đoạn và tiêu đề |
| `tai_xe.txt` | FixedSizeChunker (`fixed_size`) | 49 | 199.80 | Khá, ổn định về kích thước nhưng cắt dở ý |
| `tai_xe.txt` | SentenceChunker (`by_sentences`) | 58 | 151.14 | Khá, dễ đọc nhưng hơi vụn |
| `tai_xe.txt` | RecursiveChunker (`recursive`) | 65 | 134.63 | Tốt, giữ được section hỏi - đáp rõ hơn |
| `donhang.txt` | FixedSizeChunker (`fixed_size`) | 68 | 197.82 | Khá, nhưng bảng và bước xử lý dễ bị cắt |
| `donhang.txt` | SentenceChunker (`by_sentences`) | 75 | 159.95 | Khá, phù hợp với mô tả theo bước |
| `donhang.txt` | RecursiveChunker (`recursive`) | 86 | 139.20 | Tốt, tách được theo section `Xanh Express`, `Bước 1`, `Bước 2` linh hoạt hơn |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**
> Tôi dùng `RecursiveChunker(chunk_size=400)` vì dữ liệu XanhSM có rất nhiều cấu trúc dạng heading, bullet, bảng và FAQ đánh số. `RecursiveChunker` sẽ ưu tiên tách theo `\n\n`, sau đó là `\n`, rồi tới câu hoặc từ, nên phù hợp với tài liệu có nhiều tầng cấu trúc. So với fixed-size chunking, cách này ít cắt ngang một mục FAQ hoặc một block quy trình hơn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Với domain XanhSM, nhiều file không phải văn bản tự do mà là tài liệu vận hành và chính sách có cấu trúc rõ ràng. Vì vậy, chunk theo separator lớn trước sẽ hợp lý hơn chunk theo số ký tự hoặc chỉ theo câu. `RecursiveChunker` cũng giúp giữ được heading và nội dung ngay sau heading trong cùng vùng ngữ cảnh tốt hơn.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `khach_hang.txt` | FixedSizeChunker — baseline | 220 | 199.80 | Trung bình, top-1 đôi khi trúng đúng file nhưng lệch mục FAQ |
| `khach_hang.txt` | **RecursiveChunker — của tôi** | **303** | **128.78** | **Khá, top-3 thường gom đúng mục hỗ trợ, dù top-1 có lúc vẫn lệch sang đoạn cùng chủ đề** |
| `donhang.txt` | SentenceChunker — baseline | 75 | 159.95 | Khá, dễ đọc nhưng chưa bám hết cấu trúc section + bảng |
| `donhang.txt` | **RecursiveChunker — của tôi** | **86** | **139.20** | **Tốt hơn với tài liệu quy trình có nhiều heading và bullet** |

### So Sánh Với Thành Viên Khác

> Trong repo hiện tại không có artifact benchmark riêng của từng thành viên, nên tôi đối chiếu ba built-in strategy như ba phương án A/B/C trên cùng bộ query.

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 8 | Hợp với tài liệu dạng FAQ, policy, heading dài; top-3 ổn định hơn | Sinh nhiều chunk hơn, nên cần filter tốt để tránh nhiễu |
| Baseline A | FixedSizeChunker | 7 | Dễ cài, số chunk dễ dự đoán | Cắt dở câu và mục FAQ, làm top-1 dễ trượt |
| Baseline B | SentenceChunker | 6 | Dễ đọc, hợp với văn bản thuần câu | Với tài liệu XanhSM nhiều heading/bảng, câu chưa phải ranh giới tốt nhất |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với bộ data XanhSM, `RecursiveChunker` là lựa chọn hợp lý nhất trong ba built-in strategy vì tài liệu có cấu trúc nhiều lớp: tiêu đề, mục đánh số, bảng và hướng dẫn theo bước. Nó không hoàn hảo, nhưng giữ được coherence tốt hơn so với `FixedSizeChunker`, đồng thời bám được structure tốt hơn `SentenceChunker`.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của tôi khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi kiểm tra text rỗng trước, sau đó dùng regex `(?<=[.!?])\s+` để tách theo ranh giới câu và `strip()` để bỏ khoảng trắng thừa. Sau khi có danh sách câu, tôi gom theo `max_sentences_per_chunk` để tạo chunk gọn, dễ đọc và vẫn giữ ngữ nghĩa.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Tôi dùng thứ tự separator `["\n\n", "\n", ". ", " ", ""]` và xử lý đệ quy từ separator lớn đến nhỏ. Base case là khi đoạn hiện tại ngắn hơn `chunk_size`, còn nếu vẫn quá dài thì tiếp tục split nhỏ hơn cho tới khi fallback về cắt theo số ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Tôi chuẩn hóa mỗi document thành record gồm `id`, `content`, `metadata`, `embedding`, sau đó lưu trong in-memory store; nếu có ChromaDB thì đồng bộ thêm sang collection. Khi search, tôi embed query rồi tính dot product giữa query embedding và các embedding đã lưu, sau đó sort giảm dần theo score.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter()` lọc metadata trước rồi mới search, vì filter sau khi rank sẽ giữ lại nhiều kết quả nhiễu không đúng tập dữ liệu cần dùng. `delete_document()` xóa tất cả record có `metadata["doc_id"] == doc_id`, đồng thời thử xóa cùng IDs đó khỏi Chroma nếu backend này đang bật.

### KnowledgeBaseAgent

**`answer`** — approach:
> Tôi retrieve `top_k` chunks từ store, ghép chúng thành context có kèm `source`, sau đó dựng prompt theo format `Question -> Context -> Answer`. Nếu context yếu hoặc rỗng thì prompt vẫn ghi rõ điều đó để LLM trả lời cẩn trọng hơn thay vì bịa.

### Test Results

```text
py -m pytest tests/ -q
..........................................
42 passed, 1 warning in 0.14s
```

**Số tests pass:** `42 / 42`

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tài xế cần bằng lái A1 hoặc A2 để đăng ký. | Đăng ký tài khoản Xanh SM Bike cần bằng lái xe máy A1/A2. | high | 0.2034 | Có |
| 2 | Khách hàng chỉ có thể hủy chuyến khi chuyến xe chưa bắt đầu. | Bạn có thể hủy chuyến xe trên ứng dụng nếu chuyến xe chưa bắt đầu. | high | -0.1390 | Không |
| 3 | Nhà hàng cần giấy phép kinh doanh và giấy chứng nhận vệ sinh ATTP. | Đối tác nhà hàng phải chuẩn bị giấy phép kinh doanh và hồ sơ vệ sinh ATTP. | high | -0.1311 | Không |
| 4 | Xanh SM có thể chia sẻ dữ liệu cho công ty liên kết và cơ quan nhà nước có thẩm quyền. | Xanh SM sẽ không bán dữ liệu cá nhân cho bất cứ bên nào. | low | -0.1709 | Có |
| 5 | Quy trình giao hàng gồm nhận đơn, lấy hàng, giao hàng và hoàn thành đơn. | Xe điện VinFast có thể sạc tại trạm công cộng. | low | -0.1825 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp bất ngờ nhất là pair 2 và pair 3 vì về mặt ngữ nghĩa chúng rất giống nhau nhưng score lại âm. Điều này cho thấy trong bài lab hiện tại tôi đang dùng `_mock_embed`, nên similarity score không thật sự phản ánh semantic meaning như một embedding model thật. Công thức cosine similarity vẫn đúng, nhưng chất lượng embedding đầu vào quyết định việc score có đáng tin hay không.

---

## 6. Results — Cá nhân (10 điểm)

Trong phần benchmark này, tôi dùng `RecursiveChunker(chunk_size=400)` vì nó phù hợp hơn với cấu trúc FAQ/policy của bộ data XanhSM. Tôi cũng gắn metadata `category` và `audience` để thử `search_with_filter()` trên từng nhóm tài liệu cụ thể. Ở `main.py`, tôi cập nhật manual demo để tự chunk file theo `RecursiveChunker` trước khi đưa vào `EmbeddingStore`; riêng `ĐIỀU KHOẢN CHUNG.txt` được loại khỏi demo mặc định vì quá dài và làm phần preview/search bằng mock embedding kém rõ ràng hơn, nhưng file vẫn được giữ nguyên trong thư mục `data/data/data/` để ingest riêng khi cần.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Tôi muốn hủy chuyến xe trên ứng dụng thì làm thế nào? | Khách hàng chỉ có thể hủy chuyến xe trên ứng dụng nếu chuyến xe chưa bắt đầu; nếu không thao tác được do tình huống khẩn cấp thì cần thông báo tài xế hoặc liên hệ Xanh SM để được hỗ trợ. |
| 2 | Để đăng ký tài khoản Xanh SM Bike cần những giấy tờ gì? | Cần CCCD/CMND hoặc hộ chiếu, bằng lái A1/A2, lý lịch tư pháp, tài khoản ngân hàng chính chủ và sim chính chủ. |
| 3 | Xanh SM có bán dữ liệu cá nhân cho bên khác không, và dữ liệu có thể được chia sẻ cho ai? | Xanh SM không bán dữ liệu cá nhân; dữ liệu có thể được chia sẻ cho công ty mẹ/con/liên kết, cá nhân/tổ chức tham gia xử lý dữ liệu, hoặc cơ quan nhà nước có thẩm quyền theo quy định pháp luật. |
| 4 | Quy trình giao hàng Xanh Express tiêu chuẩn gồm những bước nào? | Gồm 4 bước chính: nhận đơn, đến lấy hàng, giao hàng cho khách nhận và hoàn thành đơn. |
| 5 | Nhà hàng cần chuẩn bị gì để đăng ký hợp tác với Xanh SM Ngon? | Cần giấy tờ tùy thân, giấy phép kinh doanh nếu áp dụng, giấy chứng nhận vệ sinh ATTP, bộ menu và hình ảnh, cùng tài khoản ngân hàng. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi muốn hủy chuyến xe trên ứng dụng thì làm thế nào? | Top-1 chưa rơi đúng mục `2.10`, nhưng top-3 đã có chunk trong `khach_hang.txt` nói rõ chỉ được hủy khi chuyến chưa bắt đầu | 43.28 | Có (top-3) | Agent trả lời gần đúng, nhưng để grounding tốt hơn nên dùng custom chunking theo heading FAQ |
| 2 | Để đăng ký tài khoản Xanh SM Bike cần những giấy tờ gì? | Chunk trong `tai_xe.txt` chứa đúng heading `1.2` và phần liệt kê giấy tờ đăng ký tài khoản tài xế | 79.65 | Có | Agent trả lời khá đầy đủ về CCCD/CMND, bằng lái, LLTP, tài khoản ngân hàng và sim chính chủ |
| 3 | Xanh SM có bán dữ liệu cá nhân cho bên khác không, và dữ liệu có thể được chia sẻ cho ai? | Top-1 trong `Chính sách bảo vệ dữ liệu cá nhân.txt` nêu các nhóm bên có thể được chia sẻ dữ liệu như công ty liên kết và cơ quan nhà nước có thẩm quyền | 23.78 | Có | Agent trả lời đúng ý “không bán” và “có thể chia sẻ theo phạm vi được nêu trong chính sách”, nhưng cần top-2/top-3 để đủ ý hơn |
| 4 | Quy trình giao hàng Xanh Express tiêu chuẩn gồm những bước nào? | Top-1 trả về chunk chứa Bước 3 và Bước 4 của quy trình; top-3 mới ghép đủ toàn bộ các bước | 22.59 | Có (top-3) | Agent trả lời được khung quy trình, nhưng retrieval cho thấy query dạng quy trình nhiều bước vẫn cần chunk tốt hơn |
| 5 | Nhà hàng cần chuẩn bị gì để đăng ký hợp tác với Xanh SM Ngon? | Chunk trong `nhahang.txt` chứa mục `1.2` và danh sách hồ sơ như giấy tờ tùy thân, GPKD, vệ sinh ATTP | 20.37 | Có | Agent trả lời đúng các loại hồ sơ chính cần chuẩn bị khi đăng ký đối tác nhà hàng |

**Bao nhiêu queries trả về chunk relevant trong top-3?** `5 / 5`

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Dù chưa có artifact benchmark riêng của từng thành viên trong repo, khi đối chiếu ba built-in strategy tôi thấy rất rõ tác động của data structure lên retrieval. Với bộ XanhSM, tài liệu FAQ/policy không hợp lắm với chunk theo độ dài cố định; cần bám vào heading và nhóm câu hỏi.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Điều quan trọng không chỉ là “retrieve đúng file” mà còn là “retrieve đúng mục nhỏ trong file”. Với các file dài như `khach_hang.txt` hay `ĐIỀU KHOẢN CHUNG.txt`, metadata filter rất hữu ích, nhưng vẫn cần chiến lược chunking thông minh hơn nếu muốn top-1 thật sự chính xác.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ tạo thêm một custom chunker theo heading FAQ như `1.2`, `2.10`, `### Bước 1`, thay vì chỉ dùng ba built-in strategy. Tôi cũng sẽ chuẩn hóa metadata chi tiết hơn như `subcategory`, `service_type`, `doc_type`, vì với bộ XanhSM việc lọc theo `category` và `audience` đã giúp rất nhiều nhưng vẫn chưa đủ để top-1 luôn chính xác.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **87 / 100** |
