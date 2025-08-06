# rag_api.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_loader import load_documents
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging
import os
import time
import gc
import chromadb
from llm_providers import get_llm_from_provider
from chat_memory import ChatMemoryManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDFApi:
    """
    A class for handling document ingestion and question answering using RAG
    with an external LLM API provider.
    """

    def __init__(self, embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the ChatPDFApi instance.
        """
        try:
            self.model = get_llm_from_provider()
            logger.info("LLM provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise
            
        try:
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            logger.info(f"Embedding model '{embedding_model}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{embedding_model}': {e}")
            raise
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                # Split by common code block separators
                "\nclass ", "\ndef ", "\n\tdef ",
                # Split by markdown headings
                "\n# ", "\n## ", "\n### ",
                # Fallback to splitting by lines
                "\n\n", "\n", " ", ""
            ]
        )
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Use the conversation history to understand the context of the user's question.
            
            Conversation History:
            {chat_history}
            
            Context from Documents:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately based on the context and conversation history.
            If the answer comes from the context, cite the source filename.
            """
        )
        
        self.vector_store = None
        self.retriever = None
        self.documents = []
        self.collection_name = "rag_collection_api"
        self.memory = ChatMemoryManager()
        
        # Initialize a persistent or in-memory ChromaDB client
        # Using an in-memory client to avoid file locking issues on Windows
        self.chroma_client = chromadb.Client()

    def _safe_remove_directory(self, directory_path, max_retries=3):
        """
        Safely remove directory with retries for Windows file locking issues.
        """
        if not os.path.exists(directory_path):
            return True
            
        for attempt in range(max_retries):
            try:
                import shutil
                shutil.rmtree(directory_path)
                logger.info(f"Successfully removed directory: {directory_path}")
                return True
            except PermissionError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to remove {directory_path}: {e}")
                if attempt < max_retries - 1:
                    # Force garbage collection and wait
                    gc.collect()
                    time.sleep(1)
                else:
                    logger.error(f"Failed to remove directory after {max_retries} attempts: {directory_path}")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error removing directory {directory_path}: {e}")
                return False
        
        return False

    def ingest(self, file_info):
        """
        Ingest files, split their contents, and store the embeddings.
        
        Args:
            file_info: Can be either:
                - List of file paths (strings) - for backward compatibility
                - List of tuples (file_path, original_filename)
        """
        # Handle both old format (list of paths) and new format (list of tuples)
        if isinstance(file_info[0], str):
            # Old format: list of file paths
            file_paths = file_info
            file_data = [(path, None) for path in file_paths]
        else:
            # New format: list of (file_path, original_filename) tuples
            file_data = file_info

        logger.info(f"Starting ingestion for {len(file_data)} files.")
        
        # Clear existing vector store first
        self._clear_vector_store()
        
        # Clear existing documents
        self.documents = []
        
        # Load documents from all files
        for file_path, original_filename in file_data:
            try:
                display_name = original_filename if original_filename else os.path.basename(file_path)
                logger.info(f"Loading documents from: {display_name} (path: {file_path})")
                
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    continue
                    
                docs = load_documents(file_path, original_filename)
                if docs:
                    # Add metadata about the original filename
                    for doc in docs:
                        if original_filename:
                            doc.metadata['source_filename'] = original_filename
                        else:
                            doc.metadata['source_filename'] = os.path.basename(file_path)
                    
                    self.documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {display_name}")
                else:
                    logger.warning(f"No documents loaded from {display_name}")
                    
            except Exception as e:
                display_name = original_filename if original_filename else file_path
                logger.error(f"Error loading documents from {display_name}: {e}")
                continue

        if not self.documents:
            raise ValueError("No documents were successfully loaded from the provided files.")

        logger.info(f"Total documents loaded: {len(self.documents)}")
        
        # Split documents into chunks
        try:
            chunks = self.text_splitter.split_documents(self.documents)
            logger.info(f"Created {len(chunks)} chunks from documents")
            
            if not chunks:
                raise ValueError("No chunks were created from the documents.")
                
            # Filter complex metadata that might cause issues
            chunks = filter_complex_metadata(chunks)
            logger.info(f"Filtered chunks: {len(chunks)}")
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

        # Create vector store
        try:
            # Use the initialized client and a consistent collection name
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=self.collection_name,
            )
            
            # Reset retriever to None so it gets recreated with new vector store
            self.retriever = None
            
            logger.info("Ingestion completed. Document embeddings stored successfully in memory.")
            
            # Verify vector store has documents
            try:
                collection = self.vector_store._collection
                doc_count = collection.count()
                logger.info(f"Vector store contains {doc_count} document chunks")
            except:
                logger.info("Vector store created successfully (document count check not available)")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def _clear_vector_store(self):
        """
        Clear the vector store and associated resources by deleting the collection.
        """
        self.vector_store = None
        self.retriever = None
        
        try:
            # Check if the collection exists before trying to delete it
            # This avoids errors if the collection was never created
            self.chroma_client.get_collection(name=self.collection_name)
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"Successfully deleted ChromaDB collection: {self.collection_name}")
        except ValueError:
            # This exception is often raised by ChromaDB if the collection doesn't exist
            logger.info(f"Collection '{self.collection_name}' not found, no need to delete.")
        except Exception as e:
            # Catch other potential exceptions during deletion
            logger.error(f"An error occurred while trying to delete collection '{self.collection_name}': {e}")

        # Force garbage collection
        gc.collect()

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline with an external LLM.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        # Add user message to memory
        self.memory.add_user_message(query)

        # Create or recreate retriever if needed
        try:
            if not self.retriever:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": k, "score_threshold": score_threshold},
                )
                logger.info(f"Created retriever with k={k}, threshold={score_threshold}")

            logger.info(f"Retrieving context for query: {query}")
            retrieved_docs = self.retriever.invoke(query)

            if not retrieved_docs:
                logger.warning("No relevant context found for the query")
                # Try with lower threshold if no docs found
                if score_threshold > 0.1:
                    logger.info("Retrying with lower similarity threshold...")
                    backup_retriever = self.vector_store.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": k, "score_threshold": 0.1},
                    )
                    retrieved_docs = backup_retriever.invoke(query)
                
                if not retrieved_docs:
                    response = "No relevant context found in the document to answer your question."
                    self.memory.add_ai_message(response)
                    return response, []

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Format context and get chat history
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            chat_history = self.memory.get_memory_context()
            
            formatted_input = {
                "context": context,
                "question": query,
                "chat_history": chat_history,
            }

            # Create chain and generate response
            chain = (
                RunnablePassthrough()
                | self.prompt
                | self.model
                | StrOutputParser()
            )

            logger.info("Generating response using the external LLM.")
            response = chain.invoke(formatted_input)
            
            # Add AI response to memory
            self.memory.add_ai_message(response)
            
            # Add source highlighting
            sources = [doc.metadata.get('source_filename', 'Unknown source') for doc in retrieved_docs]
            unique_sources = sorted(list(set(sources)))
            
            if unique_sources:
                response += f"\n\n**Sources:**\n- " + "\n- ".join(unique_sources)

            return response, unique_sources
            
        except Exception as e:
            logger.error(f"Error during question answering: {e}")
            error_message = f"An error occurred while processing your question: {e}"
            self.memory.add_ai_message(error_message)
            return error_message, []

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store, retriever, and documents.")
        self._clear_vector_store()
        self.documents = []
        self.memory.clear_memory()