# 📄 AI PDF RAG Chatbot

A PDF-based AI chatbot that uses **Retrieval-Augmented Generation (RAG)** to
answer questions about the content of an uploaded PDF, using **LangChain**,
**FAISS**, and the **Google Gemini API**, with an interactive **Streamlit** UI.

This matches the resume bullet:
> Developed a PDF-based AI chatbot using Retrieval-Augmented Generation (RAG) to
> answer questions from uploaded documents. Implemented document chunking,
> embeddings, and FAISS-based vector retrieval to provide relevant context to
> the Gemini LLM. Built an interactive Streamlit interface for PDF upload and
> real-time question answering.

---

## 🧠 How it works (RAG pipeline)

```
 PDF Upload
     │
     ▼
[1] Load & Extract Text  (utils/pdf_processor.py -> PyPDFLoader)
     │
     ▼
[2] Split into Chunks     (utils/pdf_processor.py -> RecursiveCharacterTextSplitter)
     │
     ▼
[3] Embed + Store in FAISS (utils/vector_store.py -> GoogleGenerativeAIEmbeddings + FAISS)
     │
     ▼
 User asks a question
     │
     ▼
[4] Retrieve top-k relevant chunks (utils/vector_store.py -> similarity_search)
     │
     ▼
[5] Generate answer with Gemini using retrieved chunks as context
     (utils/rag_pipeline.py -> ChatGoogleGenerativeAI)
     │
     ▼
 Answer shown in Streamlit chat UI, with sources
```

---

## 📁 Project Structure

```
pdf_rag_chatbot/
├── app.py                   # Streamlit UI — main entry point
├── config.py                 # All project settings (API key, model names, chunk sizes...)
├── requirements.txt          # Python dependencies
├── .env.example               # Template for your API key file
├── README.md                  # This file
└── utils/
    ├── __init__.py
    ├── pdf_processor.py       # PDF loading + text chunking
    ├── vector_store.py        # Embeddings + FAISS index (build/save/load/search)
    └── rag_pipeline.py        # Prompt template + Gemini answer generation
```

---

## 🚀 Setup & Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Gemini API key**
   - Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Copy `.env.example` to a new file named `.env`
   - Paste your key in:
     ```
     GOOGLE_API_KEY=your_actual_key_here
     ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints (usually `http://localhost:8501`),
   upload a PDF, and start asking questions!

---

## 🔧 Key Concepts Used

| Concept | Where | Why |
|---|---|---|
| **Chunking** | `pdf_processor.py` | LLMs and vector search work better on small, focused pieces of text than one giant document. |
| **Embeddings** | `vector_store.py` | Converts text into numeric vectors that capture semantic meaning, so "similar meaning" text can be found mathematically. |
| **FAISS** | `vector_store.py` | A fast library for searching through thousands of vectors to find the ones closest to your question. |
| **RAG Prompting** | `rag_pipeline.py` | Injects only the most relevant retrieved text into the LLM prompt, which reduces hallucination and keeps answers grounded in the actual PDF. |
| **Session State** | `app.py` | Streamlit reruns the script on every interaction; `st.session_state` lets us persist the FAISS index and chat history across reruns. |

---

## 📝 Notes

- The FAISS index currently lives in memory per session (rebuilt every time
  you upload a new PDF). `vector_store.py` also includes `save_vectorstore()`
  / `load_vectorstore()` if you want to persist an index across app restarts.
- Swap `GEMINI_CHAT_MODEL` / `GEMINI_EMBEDDING_MODEL` in `config.py` to try
  different Gemini model versions.
- This project structure is intentionally modular (separate `utils/` files)
  so each part of the RAG pipeline can be explained, tested, or swapped out
  independently — useful for interviews and further development.


  Live Link:- https://pdf-rag-chatbot-blswekfskzkzavwtcja2jd.streamlit.app/
