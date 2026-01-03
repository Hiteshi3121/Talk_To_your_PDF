# Talk_To_your_PDF
# 🧠 AI RAG Chatbot for Insurance Policy Documents

An **AI-powered Retrieval-Augmented Generation (RAG) chatbot** that allows users to **upload PDFs** and ask **natural-language questions** to get **accurate, context-grounded answers**.  
The system uses **document embeddings + vector search + LLM reasoning** to avoid hallucinations and ensure responses come strictly from provided content.

---

## 🚀 Key Features

- 📄 Upload **insurance policy PDF documents**
- 🔍 Semantic search using **vector embeddings**
- 🧠 **RAG (Retrieval-Augmented Generation)** for accurate answers
- 💬 Chat-style conversational UI
- 🧾 Answers grounded in **source policy text**
- ♻️ Cached vector store for fast repeated queries
- 🧹 Option to clear cache and rebuild embeddings

---

## 🧠 How It Works (RAG Pipeline)

1. **PDF Upload**
   - User uploads an insurance policy document via Streamlit UI

2. **Document Processing**
   - PDF is loaded and split into overlapping text chunks
   - Chunking ensures context continuity

3. **Vector Embeddings**
   - Each chunk is converted into embeddings using  
     `sentence-transformers/all-MiniLM-L6-v2`

4. **Vector Store**
   - Embeddings are stored in a **Chroma vector database**
   - Persisted locally for reuse

5. **Retrieval + LLM**
   - Relevant chunks are retrieved (`top-k = 3`)
   - Passed to Groq LLM using **RetrievalQA chain**
   - Final answer is generated strictly from retrieved context

---

## 🏗️ Project Structure

```text
.
├── Phase3.py              # Main Streamlit RAG application
├── main.py                # Entry-point / placeholder
├── vectorstore_cache/     # Persisted Chroma vector DB
├── demo_output_2.pdf      # Sample chatbot output screenshots
├── Project Layout.rtf     # Project documentation
└── README.md
