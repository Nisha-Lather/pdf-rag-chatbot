"""
utils/vector_store.py
----------------------
This module handles STEP 3 of the RAG pipeline: turning text chunks into
EMBEDDINGS (lists of numbers that capture meaning) and storing/searching
them using FAISS (Facebook AI Similarity Search) — a fast library for
finding the "closest" vectors to a given query vector.

Think of embeddings as coordinates on a giant map of "meaning": sentences
with similar meaning end up close together on that map. FAISS is the tool
that quickly finds the nearest points to our question's coordinates.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings  # wraps Gemini's embedding model for LangChain
from langchain_community.vectorstores import FAISS  # LangChain's wrapper around the FAISS library
from langchain_core.documents import Document  # standard document object (text + metadata) # standard document object (text + metadata)

import config  # our settings file (API key, model names, folder paths)


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Creates and returns the embedding model object we will use to convert
    text into vectors. Centralized here so every part of the app uses the
    exact same embedding model (mixing embedding models would break search).

    Returns
    -------
    GoogleGenerativeAIEmbeddings : the embedding model instance
    """
    return GoogleGenerativeAIEmbeddings(
        model=config.GEMINI_EMBEDDING_MODEL,   # which Gemini embedding model to call
        google_api_key=config.GOOGLE_API_KEY,  # our API key, loaded from .env via config.py
    )


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Embeds every chunk of text and stores the resulting vectors inside a
    brand-new FAISS index, held in memory (and later saved to disk).

    Parameters
    ----------
    chunks : list[Document] : the small text pieces produced by pdf_processor.py

    Returns
    -------
    FAISS : a searchable vector index built from the given chunks
    """
    embeddings = get_embedding_model()  # get our embedding model instance

    # FAISS.from_documents() does two things internally:
    #   1. Calls the embedding model on every chunk's text.
    #   2. Builds a FAISS index out of the resulting vectors.
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: str = config.VECTORSTORE_DIR) -> None:
    """
    Persists a FAISS index to disk so we don't have to re-embed the same
    PDF every time the Streamlit app restarts.

    Parameters
    ----------
    vectorstore : FAISS : the index to save
    path : str : folder to save it in (defaults to config.VECTORSTORE_DIR)
    """
    vectorstore.save_local(path)  # FAISS's built-in method to write the index + metadata to disk


def load_vectorstore(path: str = config.VECTORSTORE_DIR) -> FAISS:
    """
    Loads a previously-saved FAISS index back from disk.

    Parameters
    ----------
    path : str : folder where the index was saved

    Returns
    -------
    FAISS : the restored, searchable vector index
    """
    embeddings = get_embedding_model()  # must use the SAME embedding model used when the index was built

    # allow_dangerous_deserialization=True is required by LangChain because
    # loading a FAISS index involves unpickling files. This is safe here
    # because WE created the file ourselves (it's not from an untrusted source).
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def similarity_search(vectorstore: FAISS, query: str, k: int = config.TOP_K_RESULTS) -> list[Document]:
    """
    Given a user's question, finds the 'k' most semantically similar chunks
    stored in the FAISS index. These chunks become the "context" we hand
    to the Gemini LLM so it can answer using the PDF's actual content.

    Parameters
    ----------
    vectorstore : FAISS : the index to search
    query : str : the user's natural-language question
    k : int : how many chunks to retrieve (default comes from config.py)

    Returns
    -------
    list[Document] : the top-k most relevant chunks
    """
    results = vectorstore.similarity_search(query, k=k)  # FAISS embeds the query internally and finds neighbors
    return results
