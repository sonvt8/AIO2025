# Document OCR Gate (Python)

Đây là một module phát triển trong dự án UC1-Thu thập và số hóa các tài liệu của AI-Incubator.\
Mục đích: **quyết định tài liệu có cần OCR hay không** trước khi đưa vào pipeline trích xuất.  
Hỗ trợ PDF, ảnh (jpg/png/tiff) và tệp nén (zip).

---

## Mục tiêu
- **Nhanh**: quyết định `requiresOCR` cho từng tài liệu.
- **An toàn đầu vào**: lọc tài liệu không tồn tại, không định dạng.
- **Gợi ý thủ công**: cung cấp `metadata.likeWatermark` và `metadata.likeSignature` để *tham khảo*, **không phải** ground truth.

> Lưu ý: `likeWatermark` và `likeSignature` chỉ là gợi ý ban đầu; để xác nhận chính xác, bạn cần kỹ thuật/phần mềm cao cấp hơn (nhận diện watermark, chữ ký số, model học sâu, …).

---

## Cách hoạt động (tóm tắt)

1. **Xác thực & chuẩn hóa đường dẫn**
   - Chuẩn hóa `documentPath` theo `config.docs_root` nếu đầu vào là tên file/đường dẫn tương đối.
   - **Loại bỏ mọi document có path không tồn tại** (kể cả khi `--skip-invalid-docs` **không** bật).  
   - Bỏ qua tài liệu **không thuộc** danh sách định dạng hỗ trợ.

2. **Quyết định `requiresOCR`**
   - **Ảnh (jpg/png/tiff)**: luôn `requiresOCR = true`.
   - **PDF**: phân tích nhanh:
     - Trích ký tự có thể chọn & mức độ phủ khối text (đã bỏ qua block *nghi ngờ chữ ký*).
     - Nếu “text-based” theo ngưỡng cấu hình → `requiresOCR = false`, ngược lại `true`.
   - **ZIP**: giải nén vào `data/unzipped/<customerID>/<documentID>/` rồi xử lý từng file con như trên.

3. **Metadata gợi ý**
   - `likeWatermark`: dựa trên từ khóa watermark (tiếng Việt/Anh) hoặc ảnh phủ trang (heuristic đơn giản).
   - `likeSignature`: dựa trên keyword + vị trí + kích thước block text để *ước lượng* khu vực chữ ký.

4. **Ghi kết quả**
   - Ghi JSON đầu ra với trường `requiresOCR` và `metadata` cho từng tài liệu.

---

## Cấu trúc & thành phần

- `main.py`: CLI, logging, nhận tham số, gọi bộ xử lý.
- `config.py`: tham số hệ thống (đường dẫn, ngưỡng, pattern…).
- `document_processor.py`: toàn bộ logic validate, phân tích PDF, xử lý zip và quyết định OCR.

---

## Cài đặt nhanh

```bash
pip install pymupdf
```

> Thư viện chính: [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/) để đọc nội dung PDF.

---

## Cách dùng CLI

```bash
python main.py --input /path/to/input.json
# hoặc chỉ định nơi lưu output
python main.py --input /path/to/input.json --output /path/to/output.json
# bỏ qua tài liệu lỗi (không dừng batch)
python main.py --input /path/to/input.json --skip-invalid-docs
```

- `--input` **bắt buộc**: đường dẫn file JSON đầu vào.
- `--output` **tùy chọn**: mặc định sẽ lưu `output.json` **cùng thư mục** với file input.
- `--skip-invalid-docs` **tùy chọn**: bỏ qua tài liệu lỗi thay vì raise.

Sau khi chạy, chương trình sẽ in ra **đường dẫn file log** để bạn tiện kiểm tra.

---

## Đầu vào (input JSON)

Mảng các `entry`:

```json
[
  {
    "type": "scan",           // hoặc "email"
    "customerID": "C001",
    "transactionID": "optional-uuid",
    "documents": [
      {
        "documentID": "DOC-1",
        "documentType": "INVOICE",
        "documentFormat": "pdf|jpg|png|tiff|zip",
        "documentPath": "DOC-QD-88888.pdf"
      }
    ]
  }
]
```

- `documentPath` có thể là **tương đối** (sẽ tìm trong thư mục `docs/`) hoặc **tuyệt đối**.
- Các định dạng hợp lệ mặc định: `pdf, jpg, png, tiff, zip`.

---

## Đầu ra (output JSON)

```json
[
  {
    "customerID": "C001",
    "transactionID": "generated-or-passed",
    "documents": [
      {
        "documentID": "DOC-1",
        "documentType": "INVOICE",
        "documentPath": "/abs/path/DOC-QD-88888.pdf",
        "requiresOCR": false,
        "metadata": {
          "likeWatermark": false,
          "likeSignature": true
        }
      }
    ]
  }
]
```

- Với **ZIP**, phần `documents` sẽ chứa **các file con** sau giải nén (kế thừa `documentType`, `documentID` con định danh theo `{parentID}-{tên file}`).

---

## Mặc định & cấu hình (trong `config.py`)

- **Đường dẫn**
  - `docs_root`: thư mục chứa tài liệu đầu vào tương đối (mặc định: `./docs`)
  - `unzip_root`: nơi giải nén ZIP (mặc định: `./data/unzipped`)
- **Ngưỡng quyết định PDF “text-based”**
  - `text_threshold`: tối thiểu ký tự có thể chọn sau lọc (mặc định `100`)
  - `min_coverage_ratio`: tỷ lệ diện tích text (mặc định `0.02`)
  - `min_blocks_non_sign`: số block text không phải chữ ký tối thiểu (mặc định `2`)
- **Heuristic**
  - `watermark_patterns`: danh sách từ khóa watermark VN/EN
  - `signature_keywords`: từ khóa nhận diện vùng chữ ký/tem số
  - Tỷ lệ vị trí/kích thước giúp ước lượng vùng chữ ký
- **Logging**
  - Ghi file theo timestamp, xoay vòng dung lượng.

Bạn có thể sửa các ngưỡng cho phù hợp dữ liệu thực tế.

---

## Thư mục tài liệu (docs/)
- Nếu `documentPath` chỉ là tên file/đường dẫn tương đối, chương trình sẽ tìm trong `docs_root`.
- ZIP sẽ được **giải nén** vào `unzip_root/<customerID>/<documentID>/`.

> Khuyến nghị: chỉ xử lý ZIP **đã tin cậy**. Công cụ này giải nén để đọc file con; không nhằm mục tiêu quét mã độc.

---

## Logging & chẩn đoán
- Log chi tiết theo từng tài liệu: bắt đầu/hoàn tất, quyết định `requiresOCR`, gợi ý watermark/signature, và lỗi (nếu có).
- Khi lỗi định dạng/đường dẫn:
  - Nếu **không** bật `--skip-invalid-docs`: raise lỗi (dừng batch ở entry lỗi).
  - Nếu **bật**: bỏ qua doc lỗi, tiếp tục các doc khác.

---

## Lưu ý quan trọng
- `likeWatermark` và `likeSignature` **chỉ là gợi ý** (keyword + vị trí/kích thước/ảnh phủ trang đơn giản).  
  → Dùng để ưu tiên kiểm tra thủ công hoặc gọi bước xử lý chuyên sâu tiếp theo.
- Một số PDF ngắn (biên nhận vài dòng) có thể cần tinh chỉnh `text_threshold` / `min_coverage_ratio` để tránh đánh OCR “oan”.

---

## Ví dụ nhanh

```bash
# 1) Mặc định: đọc input, ghi output.json cạnh input
python main.py --input /data/input.json

# 2) Ghi ra đích khác
python main.py --input /data/input.json --output /data/result/output.json

# 3) Bỏ qua tài liệu lỗi (không dừng batch)
python main.py --input /data/input.json --skip-invalid-docs
```

---