"""
app.py
------
This is the MAIN entry point of the project. Running:
    streamlit run app.py
starts a web page where a user can:
    1. Upload a PDF.
    2. Wait while it gets chunked, embedded, and indexed in FAISS.
    3. Ask questions about the PDF and get answers from Gemini, grounded
       in the actual content of the document (RAG).

This file is intentionally kept "thin" — it mostly wires together the
functions defined in config.py and utils/*.py, and handles the UI.
"""

import streamlit as st  # the library that turns this script into a web app

import config  # our settings file
from utils.pdf_processor import process_pdf          # PDF -> chunks
from utils.vector_store import build_vectorstore, similarity_search  # chunks -> FAISS index -> search
from utils.rag_pipeline import generate_answer        # retrieved chunks + question -> Gemini answer


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
# st.set_page_config MUST be the first Streamlit command in the script.
st.set_page_config(
    page_title="AI PDF RAG Chatbot",  # text shown in the browser tab
    page_icon="📄",                    # emoji/icon shown in the browser tab
    layout="centered",                 # keeps content in the middle of the page
)

st.title("📄 AI PDF RAG Chatbot")  # big heading at the top of the page
st.caption("Upload a PDF and ask questions about it — powered by LangChain, FAISS & Google Gemini.")


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
# Streamlit re-runs this whole script on every user interaction (e.g. every
# button click). "st.session_state" is a dictionary-like object that PERSISTS
# across those re-runs, so we use it to remember things like the vectorstore
# and the chat history instead of losing them on every click.

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # will hold the FAISS index once a PDF is processed

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # will hold list of (question, answer) tuples


# ---------------------------------------------------------------------------
# SIDEBAR: API KEY CHECK
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    # Warn the user immediately if the Gemini API key is missing, since
    # nothing else in the app will work without it.
    if not config.GOOGLE_API_KEY:
        st.error(
            "GOOGLE_API_KEY not found!\n\n"
            "Create a `.env` file (see `.env.example`) and add your Gemini API key."
        )
    else:
        st.success("Gemini API key loaded ✅")

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Upload a PDF\n"
        "2. We split it into chunks & embed them\n"
        "3. FAISS finds the most relevant chunks for your question\n"
        "4. Gemini answers using only that context"
    )


# ---------------------------------------------------------------------------
# STEP 1: PDF UPLOAD
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])  # shows a drag-and-drop upload box

if uploaded_file is not None:
    # Only re-process the PDF if we haven't already processed this exact
    # file, so re-runs of the script (e.g. from asking a question) don't
    # waste time and API calls re-embedding the same document.
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        with st.spinner("Reading and indexing your PDF... this may take a moment."):
            chunks = process_pdf(uploaded_file)                 # PDF -> list of text chunks
            vectorstore = build_vectorstore(chunks)              # chunks -> embeddings -> FAISS index

            st.session_state.vectorstore = vectorstore           # remember the index for later questions
            st.session_state.last_uploaded_name = uploaded_file.name  # remember which file we processed
            st.session_state.chat_history = []                   # reset chat history for the new document

        st.success(f"'{uploaded_file.name}' processed into {len(chunks)} chunks and indexed! Ask a question below.")


# ---------------------------------------------------------------------------
# STEP 2: QUESTION & ANSWER
# ---------------------------------------------------------------------------
if st.session_state.vectorstore is not None:
    # st.chat_input renders a fixed input box at the bottom of the page, like a chat app.
    user_question = st.chat_input("Ask a question about your PDF...")

    if user_question:
        with st.spinner("Thinking..."):
            # Retrieve the most relevant chunks for this specific question.
            relevant_chunks = similarity_search(
                st.session_state.vectorstore,
                user_question,
                k=config.TOP_K_RESULTS,
            )

            # Generate the final answer using Gemini, grounded in those chunks.
            answer = generate_answer(user_question, relevant_chunks)

        # Save this question/answer pair into our persistent chat history.
        st.session_state.chat_history.append((user_question, answer, relevant_chunks))

    # Render the full chat history (oldest first) using Streamlit's chat bubble UI.
    for question, answer, relevant_chunks in st.session_state.chat_history:
        with st.chat_message("user"):        # renders a "user" style chat bubble
            st.write(question)

        with st.chat_message("assistant"):   # renders an "assistant" style chat bubble
            st.write(answer)

            # Let the user expand and see exactly which PDF text was used,
            # which builds trust and makes the RAG process transparent.
            with st.expander("📎 Sources used for this answer"):
                for i, chunk in enumerate(relevant_chunks, start=1):
                    page = chunk.metadata.get("page", "unknown")  # page number, if available
                    st.markdown(f"**Chunk {i} (page {page}):**")
                    st.text(chunk.page_content[:500] + "...")  # show first 500 chars of the chunk

else:
    st.info("👆 Upload a PDF above to get started.")
