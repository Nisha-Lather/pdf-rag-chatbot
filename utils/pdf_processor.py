"""
utils/pdf_processor.py
-----------------------
This module is responsible for STEP 1 and STEP 2 of the RAG pipeline:
    1. Loading text out of an uploaded PDF file.
    2. Splitting ("chunking") that text into small overlapping pieces so
       they can later be converted into embeddings.

Why do we chunk instead of feeding the whole PDF to the LLM?
  - LLMs have a limited "context window" (they can't read infinite text at once).
  - Smaller chunks let FAISS retrieve ONLY the most relevant pieces of text
    for a given question, instead of the whole document.
"""

import os  # used to build file paths in a cross-platform way
from langchain_community.document_loaders import PyPDFLoader  # reads text out of PDF files, page by page
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into smaller chunks # splits long text into smaller chunks
from langchain_core.documents import Document  # the standard "Document" object LangChain uses everywhere# the standard "Document" object LangChain uses everywhere

import config  # our own settings file (CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR, etc.)


def save_uploaded_pdf(uploaded_file) -> str:
    """
    Saves a PDF uploaded through Streamlit's file_uploader widget onto disk,
    because PyPDFLoader needs an actual file PATH, not an in-memory object.

    Parameters
    ----------
    uploaded_file : streamlit UploadedFile object (the raw bytes of the PDF)

    Returns
    -------
    str : the full path where the PDF was saved
    """
    # Build the destination path: e.g. "uploaded_pdfs/resume.pdf"
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)

    # Open the destination file in "write binary" mode and dump the uploaded bytes into it.
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # getbuffer() gives raw bytes of the uploaded PDF

    return file_path  # give the caller the path so they can load it next


def load_pdf_text(file_path: str) -> list[Document]:
    """
    Reads a PDF from disk and returns its content as a list of LangChain
    Document objects (one Document per PDF page, roughly).

    Parameters
    ----------
    file_path : str : path to the PDF file on disk

    Returns
    -------
    list[Document] : the raw, un-chunked page content of the PDF
    """
    loader = PyPDFLoader(file_path)  # create a loader pointed at our PDF file
    pages = loader.load()            # actually reads the PDF and extracts text per page
    return pages                     # each item has .page_content (text) and .metadata (page no, source, etc.)


def split_into_chunks(pages: list[Document]) -> list[Document]:
    """
    Takes the raw pages of a PDF and splits them into smaller overlapping
    chunks, which is the unit we will actually embed and search over.

    Parameters
    ----------
    pages : list[Document] : output of load_pdf_text()

    Returns
    -------
    list[Document] : smaller chunks, each with the same metadata as its parent page
    """
    # RecursiveCharacterTextSplitter tries to split on paragraph/sentence
    # boundaries first, and only falls back to hard character cuts if needed.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,        # target size of each chunk (characters)
        chunk_overlap=config.CHUNK_OVERLAP,  # how much text overlaps between chunks
        length_function=len,                 # use plain character count to measure length
    )

    chunks = splitter.split_documents(pages)  # perform the actual splitting
    return chunks


def process_pdf(uploaded_file) -> list[Document]:
    """
    Convenience wrapper that runs the full pipeline for a single uploaded
    PDF: save -> load -> chunk. This is the single function app.py will call.

    Parameters
    ----------
    uploaded_file : streamlit UploadedFile object

    Returns
    -------
    list[Document] : ready-to-embed text chunks
    """
    file_path = save_uploaded_pdf(uploaded_file)  # step 1: persist PDF to disk
    pages = load_pdf_text(file_path)              # step 2: extract raw page text
    chunks = split_into_chunks(pages)              # step 3: split into overlapping chunks
    return chunks
