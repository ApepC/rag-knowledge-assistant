"""
rag_assistant.py - RAG-Powered Knowledge Base Assistant
========================================================
Author: CS
GitHub: github.com/BobboB

Natural language Q&A over custom document collections using
Retrieval-Augmented Generation (RAG).

Stack: LangChain | ChromaDB | OpenAI GPT-4o-mini | FastAPI

Usage:
    python rag_assistant.py ingest --path ./docs/
    python rag_assistant.py query  --question "What is the return policy?"
    python rag_assistant.py serve  --port 8000
"""

import argparse, json, os, sys
from pathlib import Path

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, DirectoryLoader
    )
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as e:
    print(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST  = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "gpt-4o-mini"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K           = 4
TEMPERATURE     = 0.1

# ── System Prompt ────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based strictly
on the provided context documents. If the answer is not contained in the context,
say "I don't have enough information in the provided documents to answer that."
Never fabricate or assume information not present in the context.

Context:
{context}

Question: {question}

Answer:"""
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load local sentence-transformer embedding model (no API cost)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(path: str) -> list:
    """Load PDF and TXT files from a file path or directory."""
    p = Path(path)
    docs = []
    if p.is_file():
        loader = PyPDFLoader(path) if path.endswith(".pdf") \
                 else TextLoader(path, encoding="utf-8")
        docs = loader.load()
    elif p.is_dir():
        for loader_cls, glob in [(PyPDFLoader, "**/*.pdf"),
                                  (TextLoader,  "**/*.txt")]:
            try:
                docs.extend(
                    DirectoryLoader(path, glob=glob, loader_cls=loader_cls).load()
                )
            except Exception:
                pass
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    print(f"Loaded {len(docs)} document(s) from {path}")
    return docs


def ingest(path: str) -> None:
    """Load, chunk, embed, and store documents in ChromaDB."""
    print("=" * 55)
    print("RAG INGESTION PIPELINE — CS")
    print("=" * 55)
    docs   = load_documents(path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(docs)
    print(f"Split into {len(chunks)} chunks — generating embeddings...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PERSIST,
        collection_name=COLLECTION_NAME,
    )
    vs.persist()
    print(f"Stored {vs._collection.count()} vectors in {CHROMA_PERSIST}")
    print("=" * 55)


def build_chain() -> RetrievalQA:
    """Build the LangChain RAG chain."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    vs = Chroma(
        persist_directory=CHROMA_PERSIST,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=1024,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )


def answer(chain: RetrievalQA, question: str) -> dict:
    """Run a question through the RAG chain."""
    result  = chain({"query": question})
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result.get("source_documents", [])
    })
    return {
        "question":    question,
        "answer":      result["result"],
        "sources":     sources,
        "model":       LLM_MODEL,
        "retrieval_k": TOP_K,
    }


# ── FastAPI REST API ─────────────────────────────────────────
def create_app() -> FastAPI:
    app   = FastAPI(
        title="RAG Knowledge Base Assistant",
        description="Natural language Q&A over custom document collections.",
        version="1.0.0",
        contact={"name": "CS", "url": "https://github.com/BobboB"},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
    chain = build_chain()

    class QuestionRequest(BaseModel):
        question: str

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": LLM_MODEL, "top_k": TOP_K}

    @app.post("/ask")
    def ask_endpoint(req: QuestionRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        try:
            return answer(chain, req.question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    def stats():
        vs = Chroma(
            persist_directory=CHROMA_PERSIST,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
        return {
            "collection":      COLLECTION_NAME,
            "vector_count":    vs._collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model":       LLM_MODEL,
        }

    return app


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base Assistant — Author: CS"
    )
    sub = parser.add_subparsers(dest="command")

    ip = sub.add_parser("ingest", help="Ingest documents into vector store")
    ip.add_argument("--path", required=True, help="File or directory to ingest")

    qp = sub.add_parser("query", help="Ask a single question")
    qp.add_argument("--question", required=True)

    sp = sub.add_parser("serve", help="Start FastAPI server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if   args.command == "ingest": ingest(args.path)
    elif args.command == "query":
        print(json.dumps(answer(build_chain(), args.question), indent=2))
    elif args.command == "serve":
        print(f"Starting server at http://{args.host}:{args.port}")
        print("API docs: http://localhost:8000/docs")
        uvicorn.run(create_app(), host=args.host, port=args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
