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
                search_type="similarity",
                search_kwargs={"k": k},
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
            | self.model           # Queries the LLM
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