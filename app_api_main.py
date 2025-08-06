# app_api_main.py
import os
import tempfile
import time
import json
import streamlit as st
from streamlit_chat import message
from rag_api import ChatPDFApi
from document_summarizer import DocumentSummarizer
from chat_history import ChatHistoryManager

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
                agent_text, sources = st.session_state["assistant"].ask(
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
                st.session_state["assistant"].ingest(file_info)
                t1 = time.time()
                st.success(f"Ingested {len(file_info)} files in {t1 - t0:.2f} seconds")

                # Summarize the first document
                if st.session_state["assistant"].documents:
                    first_doc_text = st.session_state["assistant"].documents[0].page_content
                    with st.spinner("Generating summary..."):
                        summary = st.session_state["summarizer"].summarize(first_doc_text)
                        st.session_state["summary"] = summary
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
    if "assistant" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["summary"] = ""
        try:
            config = load_config()
            embedding_model = config.get("embedding_model", "mxbai-embed-large")
            st.session_state["assistant"] = ChatPDFApi(embedding_model=embedding_model)
            st.session_state["summarizer"] = DocumentSummarizer()
            st.session_state["history_manager"] = ChatHistoryManager()
        except Exception as e:
            st.error(f"Failed to initialize the application: {e}")
            st.stop()

    st.header("RAG with External LLM API")

    with st.sidebar:
        st.subheader("Upload Documents")
        st.file_uploader(
            "Upload documents",
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

        st.subheader("Settings")
        st.session_state["retrieval_k"] = st.slider(
            "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
        )
        st.session_state["retrieval_threshold"] = st.slider(
            "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )

        if st.button("Save Chat Session"):
            session_data = st.session_state["assistant"].memory.export_conversation()
            st.session_state["history_manager"].save_session(session_data)
            st.success("Chat session saved!")

        st.subheader("Chat History")
        sessions = st.session_state["history_manager"].list_sessions()
        for session in sessions:
            with st.expander(f"{session['session_id']} ({session['message_count']} msgs)"):
                st.write(f"Created: {session['created_at']}")
                st.write(f"Preview: {session['preview']}")
                if st.button("Load", key=f"load_{session['session_id']}"):
                    loaded_session = st.session_state["history_manager"].load_session(session['session_id'])
                    st.session_state["messages"] = [
                        (msg['content'], msg['type'] == 'human') for msg in loaded_session['messages']
                    ]
                    st.session_state["summary"] = "" # Clear summary when loading a session
                    st.rerun()
                if st.button("Delete", key=f"delete_{session['session_id']}"):
                    st.session_state["history_manager"].delete_session(session['session_id'])
                    st.rerun()

    if st.session_state["summary"]:
        with st.expander("Document Summary", expanded=True):
            st.markdown(st.session_state["summary"])

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["summary"] = ""
        st.session_state["assistant"].clear()
        st.rerun()

if __name__ == "__main__":
    page()