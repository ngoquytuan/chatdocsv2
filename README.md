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



Tôi sẽ giúp bạn tạo file `rag_sentence_transformer.py` và cập nhật các file `config.json` và `app.py` để hỗ trợ chọn pipeline thông qua cấu hình. Dưới đây là các file được sửa đổi và tạo mới, với các thay đổi như sau:

1. **Cập nhật `config.json`**: Thêm trường `sentence_transformer_model` để chỉ định mô hình SentenceTransformer và trường `pipeline` để chọn pipeline (`ollama` hoặc `sentence_transformer`).
2. **Tạo `rag_sentence_transformer.py`**: Tạo một lớp `ChatPDFSentenceTransformer` dựa trên `rag.py`, sử dụng `SentenceTransformerEmbeddings` thay vì `OllamaEmbeddings`.
3. **Cập nhật `app.py`**: Thêm logic để chọn giữa `ChatPDF` và `ChatPDFSentenceTransformer` dựa trên trường `pipeline` trong `config.json`.

Dưới đây là các file:

<xaiArtifact artifact_id="54c03587-bf2b-4b8d-964f-5db7c1f1cd4e" artifact_version_id="7f76e450-0a20-4ed9-9cac-117abdb41bc7" title="config.json" contentType="application/json">
{
  "llm_model": "deepseek-r1:latest",
  "embedding_model": "mxbai-embed-large:latest",
  "sentence_transformer_model": "all-MiniLM-L6-v2",
  "pipeline": "ollama",
  "chunk_size": 1024,
  "chunk_overlap": 100
}
</xaiArtifact>

<xaiArtifact artifact_id="b2269551-a382-47fa-b7fb-113ffba06ef5" artifact_version_id="82416653-3c48-4d88-8b00-2f9287fbb9a5" title="rag_sentence_transformer.py" contentType="text/python">
import json
import os
import shutil
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDFSentenceTransformer:
    """A class for handling PDF ingestion and question answering using RAG with SentenceTransformer."""

    def __init__(self):
        """
        Initialize the ChatPDF instance with an LLM and SentenceTransformer embedding model from config.json.
        """
        with open("config.json") as f:
            config = json.load(f)
        llm_model = config.get("llm_model")
        embedding_model = config.get("sentence_transformer_model")

        self.model = ChatOllama(model=llm_model)
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        chunk_size = config.get("chunk_size", 1024)
        chunk_overlap = config.get("chunk_overlap", 100)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        with open("prompt.json") as f:
            prompt_config = json.load(f)
        prompt_template = prompt_config.get("prompt_template")
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db_sentence_transformer",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model_cont      # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        if self.vector_store:
            del self.vector_store
        if self.retriever:
            del self.retriever
        self.vector_store = None
        self.retriever = None
        if os.path.exists("chroma_db_sentence_transformer"):
            try:
                shutil.rmtree("chroma_db_sentence_transformer")
                logger.info("Chroma DB cleared.")
            except PermissionError as e:
                logger.error(f"Failed to clear Chroma DB: {e}. This often happens on Windows if files are still in use. Please restart the application if the issue persists.")
</xaiArtifact>

<xaiArtifact artifact_id="36314521-edc2-4238-920a-9eb62b9a53ec" artifact_version_id="eed9eaa4-6c75-40fa-979f-fd8628cbe2fb" title="app.py" contentType="text/python">
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF
from rag_sentence_transformer import ChatPDFSentenceTransformer
import json

st.set_page_config(page_title="RAG with Local DeepSeek R1")

def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            try:
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
        )
        os.remove(file_path)

def page():
    """Main app page layout."""
    if len(st.session_state) == 0:
        # Load pipeline configuration
        with open("config.json") as f:
            config = json.load(f)
        pipeline = config.get("pipeline", "ollama")
        
        # Initialize the appropriate assistant based on pipeline
        if pipeline == "sentence_transformer":
            st.session_state["assistant"] = ChatPDFSentenceTransformer()
        else:
            st.session_state["assistant"] = ChatPDF()
        st.session_state["messages"] = []

    st.header("RAG with Local DeepSeek R1")

    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    # Display messages and text input
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()

if __name__ == "__main__":
    page()
</xaiArtifact>

### Giải thích ngắn gọn:
1. **`config.json`**:
   - Thêm `sentence_transformer_model` với giá trị mặc định là `all-MiniLM-L6-v2` (mô hình nhẹ, phổ biến của SentenceTransformer).
   - Thêm `pipeline` để chọn giữa `ollama` và `sentence_transformer`. Mặc định là `ollama` để giữ tương thích với code gốc.

2. **`rag_sentence_transformer.py`**:
   - Tạo lớp `ChatPDFSentenceTransformer` dựa trên `rag.py`, nhưng sử dụng `SentenceTransformerEmbeddings`.
   - Sử dụng thư mục `chroma_db_sentence_transformer` để tránh xung đột với `chroma_db` của pipeline Ollama.
   - Giữ nguyên logic của `ingest`, `ask`, và `clear`, chỉ thay đổi phần embedding và tên thư mục lưu trữ.

3. **`app.py`**:
   - Thêm import `ChatPDFSentenceTransformer` từ `rag_sentence_transformer`.
   - Trong hàm `page()`, đọc `pipeline` từ `config.json` và khởi tạo `ChatPDF` hoặc `ChatPDFSentenceTransformer` tương ứng.
   - Phần còn lại của `app.py` không thay đổi, vì cả hai lớp đều có cùng interface (`ingest`, `ask`, `clear`).

### Lưu ý:
- Đảm bảo cài đặt thư viện `sentence-transformers` (`pip install sentence-transformers`).
- File `prompt.json` được giả định là đã tồn tại và không cần thay đổi.
- Bạn có thể thay đổi `sentence_transformer_model` trong `config.json` (ví dụ: `paraphrase-mpnet-base-v2` cho chất lượng tốt hơn nhưng nặng hơn).
- Để sử dụng pipeline SentenceTransformer, chỉnh `pipeline` trong `config.json` thành `sentence_transformer`.

Nếu bạn cần thêm chỉnh sửa hoặc giải thích chi tiết hơn, hãy cho tôi biết!