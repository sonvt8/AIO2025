
# Trợ lý RAG - Ứng dụng Hỏi Đáp Thông Minh

Chào mừng bạn đến với **Trợ lý RAG**, một ứng dụng thông minh sử dụng công nghệ Retrieval-Augmented Generation (RAG) để cung cấp câu trả lời chi tiết dựa trên các tài liệu đã tải lên. Ứng dụng này hỗ trợ cả mô hình ngôn ngữ OpenAI (ChatGPT) và mô hình cục bộ như Llama3.2 thông qua Ollama.

## Hướng dẫn tải thư mục `week4/rag` từ GitHub repository

Nếu bạn chỉ muốn tải thư mục `week4/rag` từ GitHub repository mà không cần tải toàn bộ nội dung repo, bạn có thể sử dụng `git sparse-checkout`. Dưới đây là các bước hướng dẫn:

1. Tạo thư mục cho dự án:
   ```bash
   mkdir my-rag-project
   cd my-rag-project
   ```

2. Khởi tạo git repository và kết nối với remote repository của bạn:
   ```bash
   git init
   git remote add origin https://github.com/sonvt8/AIO2025.git
   ```

3. Cấu hình `git sparse-checkout` để chỉ tải thư mục `week4/rag`:
   ```bash
   git sparse-checkout init --cone
   git sparse-checkout set week4/rag
   ```

4. Pull dữ liệu từ repository:
   ```bash
   git pull origin sonvt8
   ```

Sau khi thực hiện các bước trên, thư mục `week4/rag` sẽ được tải về mà không cần tải toàn bộ repo.

## Yêu cầu hệ thống

- **Hệ điều hành**: MacOS hoặc Windows.
- **Python**: Phiên bản 3.9 hoặc cao hơn.
- **Thư viện phụ thuộc**: Được liệt kê trong file `requirements.txt`.
- **Ollama** (nếu sử dụng Llama3.2).
- **API Key OpenAI** (nếu sử dụng ChatGPT).

## Cài đặt và cấu hình

### 1. Tải và cài đặt Ollama
Ollama là công cụ cần thiết để chạy mô hình ngôn ngữ cục bộ như Llama3.2. Thực hiện các bước sau:

- **Trên MacOS**:
  1. Mở terminal và chạy lệnh sau để cài đặt Ollama:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```
  Sau khi cài đặt, khởi động Ollama bằng lệnh:
  ```bash
  ollama serve
  ```
  Tải mô hình Llama3.2 bằng lệnh:
  ```bash
  ollama pull llama3.2
  ```

- **Trên Windows**:
  Truy cập trang chính thức của Ollama và tải file cài đặt cho Windows.
  Cài đặt bằng cách chạy file đã tải và làm theo hướng dẫn.
  Mở Command Prompt, khởi động Ollama bằng lệnh:
  ```bash
  ollama serve
  ```
  Tải mô hình Llama3.2 bằng lệnh:
  ```bash
  ollama pull llama3.2
  ```

Lưu ý: Đảm bảo Ollama đang chạy trong nền trước khi khởi động ứng dụng.

### 2. Cấu hình file .env
Tạo file `.env` dựa trên template mẫu từ file `.env.example`. Dưới đây là nội dung mẫu:

```env
# API TOKENS
OPENAI_API_KEY=""

# LOCAL PARAMETERS
DATA_DIR=""
DOCUMENTS_DIR=""
USERS='{"admin": "admin123", "user1": "pass123", "user2": "securepass"}'
```

Hướng dẫn điền:
- `OPENAI_API_KEY`: Thêm key API của OpenAI nếu bạn muốn sử dụng ChatGPT (lấy tại OpenAI API).
- `DATA_DIR`: Đường dẫn thư mục chứa dữ liệu (mặc định là data).
- `DOCUMENTS_DIR`: Đường dẫn thư mục chứa tài liệu (mặc định là documents trong cùng thư mục với mã nguồn).
- `USERS`: Dictionary JSON chứa thông tin đăng nhập (username:password). Bạn có thể giữ nguyên hoặc chỉnh sửa theo nhu cầu.

Lưu ý: Sao chép nội dung trên vào file `.env` trong cùng thư mục với mã nguồn và điền thông tin phù hợp.

### 3. Sử dụng ChatGPT (Tùy chọn)
Nếu muốn sử dụng ChatGPT thay vì Llama3.2:

1. Đăng ký tài khoản tại [OpenAI](https://platform.openai.com/docs/overview) và lấy API Key.
2. Thêm API Key vào biến `OPENAI_API_KEY` trong file `.env`.
3. Chọn mô hình "OpenAI GPT-4" trong giao diện ứng dụng khi sử dụng.

### 4. Cài đặt thư viện
Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```
Đảm bảo các thư viện như `streamlit`, `langchain`, `chromadb`, và `ollama` được cài đặt đúng.

## Khởi chạy ứng dụng
Khởi chạy ứng dụng bằng lệnh sau trong terminal hoặc Command Prompt:
```bash
streamlit run rag_chatbot_app.py > log.txt 2>&1
```

Giải thích:
- `streamlit run rag_chatbot_app.py`: Chạy ứng dụng Streamlit với file `rag_chatbot_app.py`.
- `> log.txt 2>&1`: Chuyển hướng output (bao gồm lỗi) vào file `log.txt` để theo dõi thông tin log.

Mở trình duyệt tại địa chỉ `http://localhost:8501` để sử dụng ứng dụng.

## Hướng dẫn sử dụng

### Đăng nhập:
Sử dụng thông tin tài khoản từ biến `USERS` trong file `.env` (mặc định: `admin/admin123`).
Nhập đúng username và password để truy cập giao diện chính.

### Tải tài liệu:
Trong sidebar, sử dụng nút "Tải lên tệp" để thêm các file .pdf, .txt, .md, hoặc .csv.
Nhấn "Xử lý Tài liệu" để xử lý và lưu vào ChromaDB.

### Truy vấn:
Nhập câu hỏi vào ô "Nhập câu hỏi của bạn".
Điều chỉnh số kết quả trả về bằng slider.
Nhấn "Tìm kiếm" để nhận câu trả lời chi tiết với trích dẫn từ tài liệu.

## Lưu ý quan trọng
- Đảm bảo file `.env` được cấu hình đúng trước khi chạy ứng dụng.
- Log hệ thống được ghi vào file `rag_system.log` và output chạy ứng dụng được lưu trong `log.txt`.
- Nếu gặp lỗi, kiểm tra file log hoặc liên hệ với nhà phát triển để được hỗ trợ.

## Ghi chú
Đây là project 1.2 đã cải tiến một số tính năng so với project gốc mà TA giới thiệu trong khóa học.

Chúc bạn sử dụng thành công Trợ lý RAG!