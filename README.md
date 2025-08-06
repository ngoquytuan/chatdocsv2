# Trò chuyện với tài liệu của bạn bằng RAG và các API LLM

Ứng dụng này là một công cụ Retrieval-Augmented Generation (RAG) cho phép người dùng tải lên nhiều loại tài liệu khác nhau và tương tác với chúng thông qua một giao diện chatbot. Hệ thống sử dụng các mô hình embedding và LLM có thể cấu hình (thông qua Ollama hoặc các nhà cung cấp API bên ngoài như OpenRouter) và một vector store trong bộ nhớ để trả lời câu hỏi một cách hiệu quả và chính xác.

## Tính năng

- **Tải lên nhiều loại tài liệu**: Hỗ trợ các định dạng PDF, DOCX, DOC, TXT, Markdown, và nhiều loại file mã nguồn khác.
- **Lựa chọn mô hình linh hoạt**: Chạy script `loadmodel.py` để chọn bất kỳ mô hình LLM và embedding nào có sẵn trên Ollama của bạn.
- **Hỗ trợ API LLM bên ngoài**: Dễ dàng tích hợp với các nhà cung cấp như OpenRouter, Groq, Google Gemini, hoặc OpenAI bằng cách cấu hình file `.env`.
- **Vector Store trong bộ nhớ**: Sử dụng ChromaDB trong bộ nhớ (RAM) để tránh các vấn đề về khóa file trên Windows và tăng tốc độ truy xuất.
- **Giao diện người dùng thân thiện**: Xây dựng bằng Streamlit để tương tác mượt mà.
- **Tùy chỉnh truy xuất**: Điều chỉnh số lượng kết quả được truy xuất (`k`) và ngưỡng điểm tương đồng để tinh chỉnh hiệu suất.

---

## Cài đặt

Làm theo các bước dưới đây để cài đặt và chạy ứng dụng:

### 1. Sao chép Repository

```bash
git clone <URL_REPOSITORY_CUA_BAN>
cd <TEN_THU_MUC_REPOSITORY>
```

### 2. Tạo môi trường ảo

```bash
# Dành cho Windows
python -m venv venv
venv\Scripts\activate

# Dành cho macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt các gói phụ thuộc

```bash
pip install -r requirements.txt
```

### 4. Cấu hình

1.  **API Keys (Tùy chọn)**: Nếu bạn muốn sử dụng một nhà cung cấp LLM bên ngoài (ví dụ: OpenRouter), hãy sao chép `.env.example` thành `.env` và điền API key của bạn vào.
    ```bash
    cp .env.example .env
    ```
2.  **Chọn mô hình**: Chạy script `loadmodel.py` để chọn LLM và mô hình embedding bạn muốn sử dụng từ danh sách các mô hình Ollama có sẵn trên máy của bạn. Lựa chọn sẽ được lưu vào `config.json`.
    ```bash
    python loadmodel.py
    ```

---

## Sử dụng

### 1. Khởi chạy ứng dụng

```bash
streamlit run app_api_main.py
```
pip install streamlit

### 2. Tải lên tài liệu

-   Sử dụng giao diện web để tải lên một hoặc nhiều tài liệu.
-   Ứng dụng sẽ tự động xử lý và nạp các tài liệu vào vector store.

### 3. Đặt câu hỏi

-   Nhập câu hỏi của bạn vào ô chat và nhấn Enter.
-   Sử dụng các thanh trượt trong mục "Settings" để điều chỉnh các tham số truy xuất nếu cần.

### 4. Xóa cuộc trò chuyện

-   Nhấn nút "Clear Chat" để xóa lịch sử trò chuyện và làm sạch vector store trong bộ nhớ.

---

## Cấu trúc dự án

```
.
├── app_api_main.py         # Giao diện người dùng Streamlit
├── rag_api.py              # Logic RAG chính (nạp tài liệu, trả lời câu hỏi)
├── llm_providers.py        # Quản lý việc lấy LLM từ các nhà cung cấp khác nhau
├── document_loader.py      # Xử lý việc tải và đọc các loại tài liệu
├── loadmodel.py            # Script để chọn và lưu cấu hình mô hình
├── config.json             # Lưu trữ các mô hình đã chọn (tự động tạo)
├── requirements.txt        # Các gói phụ thuộc của Python
├── .env                    # Lưu trữ API keys (cần tạo từ .env.example)
└── README.md               # Tài liệu dự án
```

---
---

# Chat with Your Docs using RAG and LLM APIs

This application is a Retrieval-Augmented Generation (RAG) tool that allows users to upload various documents and interact with them through a chatbot interface. The system uses configurable embedding and LLM models (via Ollama or external API providers like OpenRouter) and an in-memory vector store for efficient and accurate question-answering.

## Features

- **Multi-Format Document Upload**: Supports PDF, DOCX, DOC, TXT, Markdown, and various source code files.
- **Flexible Model Selection**: Run the `loadmodel.py` script to choose any LLM and embedding model available in your local Ollama instance.
- **External LLM API Support**: Easily integrate with providers like OpenRouter, Groq, Google Gemini, or OpenAI by configuring the `.env` file.
- **In-Memory Vector Store**: Uses an in-memory ChromaDB instance to avoid file-locking issues on Windows and for faster performance.
- **User-Friendly UI**: Built with Streamlit for seamless interaction.
- **Customizable Retrieval**: Adjust the number of retrieved results (`k`) and the similarity score threshold to fine-tune performance.

---

## Installation

Follow the steps below to set up and run the application:

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY_NAME>
```

### 2. Create a Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

1.  **API Keys (Optional)**: If you want to use an external LLM provider (e.g., OpenRouter), copy `.env.example` to `.env` and fill in your API key.
    ```bash
    cp .env.example .env
    ```
2.  **Select Models**: Run the `loadmodel.py` script to select the LLM and embedding models you want to use from your list of locally available Ollama models. Your choices will be saved to `config.json`.
    ```bash
    python loadmodel.py
    ```

---

## Usage

### 1. Start the Application

```bash
streamlit run app_api_main.py
```

### 2. Upload Documents

-   Use the web interface to upload one or more documents.
-   The application will automatically process and ingest the documents into the vector store.

### 3. Ask Questions

-   Type your question in the chat input box and press Enter.
-   Use the sliders in the "Settings" section to adjust retrieval parameters if needed.

### 4. Clear Chat

-   Click the "Clear Chat" button to reset the chat history and clear the in-memory vector store.

---

## Project Structure

```
.
├── app_api_main.py         # Streamlit user interface
├── rag_api.py              # Core RAG logic (ingestion, Q&A)
├── llm_providers.py        # Manages getting LLMs from different providers
├── document_loader.py      # Handles loading and reading different document types
├── loadmodel.py            # Script to select and save model configuration
├── config.json             # Stores selected models (auto-generated)
├── requirements.txt        # Python dependencies
├── .env                    # Stores API keys (created from .env.example)
└── README.md               # Project documentation
