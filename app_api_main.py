# app_api_main.py
import os
import tempfile
import time
import json
import streamlit as st
from streamlit_chat import message
from rag_api import ChatPDFApi

st.set_page_config(page_title="RAG with LLM API")

def load_config():
    """Load configuration from config.json."""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("config.json not found. Please run loadmodel.py to create it.")
        return {}
    except json.JSONDecodeError:
        st.error("Error decoding config.json. Please check its format.")
        return {}

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
            except Exception as e:
                agent_text = f"An error occurred: {e}"

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    file_info = []  # Store (file_path, original_name) tuples
    temp_files = []

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_info.append((tf.name, file.name))  # Store temp path and original name
            temp_files.append(tf.name)

    if file_info:
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {len(file_info)} files..."):
            try:
                t0 = time.time()
                st.session_state["assistant"].ingest(file_info)  # Pass file info instead of just paths
                t1 = time.time()
                st.session_state["messages"].append(
                    (f"Ingested {len(file_info)} files in {t1 - t0:.2f} seconds", False)
                )
            except Exception as e:
                st.error(f"Error ingesting files: {e}")
                st.session_state["messages"].append((f"Error: {e}", False))
            finally:
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass  # Ignore cleanup errors

def page():
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        try:
            config = load_config()
            embedding_model = config.get("embedding_model", "mxbai-embed-large") # Fallback to default
            st.session_state["assistant"] = ChatPDFApi(embedding_model=embedding_model)
        except Exception as e:
            st.error(f"Failed to initialize the assistant: {e}")
            st.stop()

    st.header("RAG with External LLM API")

    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload a document",
        type=[
            "pdf", "doc", "docx", "txt", "md",
            "py", "java", "c", "cpp", "h", "html", "css", "js"
        ],
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