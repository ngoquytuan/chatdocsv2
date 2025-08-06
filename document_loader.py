# document_loader.py
import os
import mimetypes
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

def detect_file_type(file_path: str, original_filename: str = None):
    """
    Detect file type from extension or MIME type.
    Falls back to original filename if available.
    """
    # First try to get extension from the file path
    _, file_extension = os.path.splitext(file_path)
    
    # If no extension and we have original filename, use that
    if not file_extension and original_filename:
        _, file_extension = os.path.splitext(original_filename)
    
    # If still no extension, try MIME type detection
    if not file_extension:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            extension_map = {
                'application/pdf': '.pdf',
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'text/plain': '.txt',
                'text/markdown': '.txt',  # Treat as text
                'text/x-python': '.txt',  # Treat as text
                'text/x-java': '.txt',    # Treat as text
                'text/x-c': '.txt',       # Treat as text
                'text/x-c++': '.txt',     # Treat as text
                'text/html': '.txt',      # Treat as text
                'text/css': '.txt',       # Treat as text
                'application/javascript': '.txt',  # Treat as text
            }
            file_extension = extension_map.get(mime_type, '')
    
    return file_extension.lower()

def get_document_loader(file_path: str, original_filename: str = None):
    """
    Selects the appropriate document loader based on the file extension.
    All text-based files (including markdown and code) are treated as text files.
    """
    file_extension = detect_file_type(file_path, original_filename)
    
    if not file_extension:
        # Try to read as text file as fallback
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(100)  # Try to read first 100 chars
            file_extension = '.txt'
        except UnicodeDecodeError:
            # If not text, might be PDF
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header == b'%PDF':
                        file_extension = '.pdf'
            except:
                pass

    # Simplified logic: only PDF, DOC/DOCX get special treatment
    # Everything else is treated as text
    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension in [".doc", ".docx"]:
        return UnstructuredWordDocumentLoader(file_path)
    else:
        # Treat everything else as text (including .md, .py, .java, .c, .cpp, .h, .html, .css, .js, .txt)
        return TextLoader(file_path, encoding="utf-8")

def load_documents(file_path: str, original_filename: str = None):
    """
    Loads documents from a file using the appropriate loader.
    """
    loader = get_document_loader(file_path, original_filename)
    return loader.load()