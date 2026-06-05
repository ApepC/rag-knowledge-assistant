# RAG-Powered Knowledge Base Assistant

**Author: CS** | [GitHub](https://github.com/ApepC)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi)

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that enables natural language Q&A over any custom document collection. Built as part of the DataCamp AI Engineering curriculum.

---

## Architecture

```
Documents (PDF / TXT)
        │
        ▼
 ┌─────────────────┐
 │  Text Chunking  │  RecursiveCharacterTextSplitter
 │  chunk=512      │  overlap=64
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   Embeddings    │  sentence-transformers/all-MiniLM-L6-v2
 │  (local, free)  │  No API cost for embedding
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │    ChromaDB     │  Persistent local vector store
 │  Vector Store   │  cosine similarity search
 └────────┬────────┘
          │  top-k=4 relevant chunks retrieved
          ▼
 ┌─────────────────┐
 │  GPT-4o-mini   │  LangChain RetrievalQA chain
 │  + RAG Prompt  │  Grounded answer generation
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   FastAPI REST  │  POST /ask  GET /stats  GET /health
 └─────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ApepC/rag-knowledge-assistant.git
cd rag-knowledge-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Ingest your documents
python src/rag_assistant.py ingest --path ./docs/

# 5. Ask a question
python src/rag_assistant.py query --question "What is the refund policy?"

# 6. Start REST API
python src/rag_assistant.py serve
# API docs: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question, get grounded answer + sources |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Vector count, model info |

**Example request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

**Example response:**
```json
{
  "question": "What is the return policy?",
  "answer": "Customers may request a full refund within 30 days of purchase...",
  "sources": ["docs/sample.txt"],
  "model": "gpt-4o-mini",
  "retrieval_k": 4
}
```

---

## Key Concepts Demonstrated

- **RAG Architecture** — retrieval-augmented generation to ground LLM output in real documents
- **Semantic Chunking** — recursive text splitting with overlap for context preservation
- **Local Embeddings** — `sentence-transformers` for cost-free embedding generation
- **Vector Similarity Search** — ChromaDB cosine similarity for top-k retrieval
- **Prompt Engineering** — custom system prompt to minimize hallucination
- **LangChain Orchestration** — RetrievalQA chain wiring retriever → LLM → response
- **FastAPI Deployment** — production-ready REST API with Pydantic validation

---

## Project Structure

```
rag-knowledge-assistant/
├── src/
│   └── rag_assistant.py   # Main pipeline: ingest, query, serve
├── docs/
│   └── sample.txt         # Sample knowledge base document
├── requirements.txt
├── .env.example
└── README.md
```

---

*Built by CS —*
