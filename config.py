"""
config.py
----------
This file stores all the "settings" of our project in one place.
Keeping settings here (instead of scattering them across files) makes the
project easy to tune later — e.g. if you want bigger text chunks, you only
change ONE number, here.
"""

import os  # 'os' lets us read environment variables (like secret API keys) from the system
from dotenv import load_dotenv  # this function loads variables from a ".env" file into the environment

# Load the .env file (if it exists) so that os.getenv() below can find the values inside it.
load_dotenv()

# ---------------------------------------------------------------------------
# 1. API KEY
# ---------------------------------------------------------------------------
# We NEVER hard-code API keys directly in code (that's a security risk).
# Instead, we store it in a ".env" file (see .env.example) and read it here.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # returns None if not found in .env

# ---------------------------------------------------------------------------
# 2. GEMINI MODEL SETTINGS
# ---------------------------------------------------------------------------
GEMINI_CHAT_MODEL = "gemini-flash-latest"       # the LLM used to generate answers (always Google's current recommended Flash model)     # the LLM used to generate answers (fast + cheap)
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"  # the model used to turn text into vectors (embeddings)  # the model used to turn text into vectors (embeddings)

# ---------------------------------------------------------------------------
# 3. TEXT CHUNKING SETTINGS
# ---------------------------------------------------------------------------
# When we split a PDF into small pieces ("chunks"), these control how big
# each piece is, and how much neighboring chunks overlap (so we don't lose
# context at the edges of a chunk).
CHUNK_SIZE = 1000       # each chunk will have (roughly) this many characters
CHUNK_OVERLAP = 200     # this many characters are shared between consecutive chunks

# ---------------------------------------------------------------------------
# 4. RETRIEVAL SETTINGS
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 4   # how many of the most relevant chunks to fetch from FAISS for every question

# ---------------------------------------------------------------------------
# 5. FILE PATHS
# ---------------------------------------------------------------------------
# Folder where uploaded PDFs are temporarily stored.
UPLOAD_DIR = "uploaded_pdfs"

# Folder where the FAISS vector index is saved to disk, so we don't have to
# re-embed the same PDF every time the app restarts.
VECTORSTORE_DIR = "vectorstore"

# Make sure these folders actually exist on disk; create them if missing.
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
