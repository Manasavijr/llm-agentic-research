"""
RAG (Retrieval-Augmented Generation) Pipeline.

Full pipeline:
  1. Document ingestion (PDF, DOCX, TXT)
  2. Semantic chunking with overlap
  3. Embedding with sentence-transformers
  4. FAISS vector index (L2 + cosine)
  5. Hybrid retrieval (dense + keyword)
  6. LangChain QA chain with Ollama

Designed for engineering documentation use cases:
  technical manuals, API docs, spec sheets, research papers.
"""
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

ENGINEERING_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert engineering AI assistant with deep technical knowledge.
Use the following retrieved context to answer the engineering question precisely.

Context:
{context}

Engineering Question: {question}

Instructions:
- Be technically precise and use domain-specific terminology
- If the context contains relevant code or specs, reference them directly
- If the context is insufficient, clearly state what information is missing
- Structure your answer with clear sections if the answer is complex

Technical Answer:"""
)


class DocumentChunker:
    """
    Semantic-aware document chunker optimized for engineering docs.
    Uses recursive splitting that respects code blocks, tables, and sections.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict = None) -> List[Dict]:
        """Split text into chunks with metadata."""
        chunks = self.splitter.split_text(text)
        return [
            {
                "text": chunk,
                "chunk_id": hashlib.md5(chunk.encode()).hexdigest()[:8],
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                **(metadata or {}),
            }
            for chunk in chunks
            if len(chunk.strip()) > 50  # filter noise
        ]


class RAGEngine:
    """
    Full RAG engine with FAISS vector search and Ollama LLM.
    Supports multiple retrieval strategies for benchmarking.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ):
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)

        self._embedding_model_name = embedding_model
        self.embeddings = None
        logger.info(f"Embedding model will load on first use: {embedding_model}")

        self._ollama_base_url = ollama_base_url
        self._llm_model_name = llm_model
        self.llm = None
        self.vectorstore: Optional[LangchainFAISS] = None
        self.chunks: List[Dict] = []
        self.documents_indexed: List[str] = []

        logger.info("RAG engine initialized")

    def _get_llm(self):
        if self.llm is None:
            self.llm = OllamaLLM(base_url=self._ollama_base_url, model=self._llm_model_name, temperature=0.1)
        return self.llm

    def _get_embeddings(self):
        if self.embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings as HFE
            self.embeddings = HFE(model_name=self._embedding_model_name)
        return self.embeddings

    def ingest_text(self, text: str, source: str = "unknown") -> int:
        """Ingest raw text into the vector store."""
        chunks = self.chunker.chunk(text, metadata={"source": source})
        self._index_chunks(chunks)
        self.documents_indexed.append(source)
        logger.info(f"Ingested '{source}': {len(chunks)} chunks")
        return len(chunks)

    def ingest_file(self, filepath: str) -> int:
        """Ingest a file (PDF, DOCX, TXT) into the vector store."""
        path = Path(filepath)
        ext = path.suffix.lower()

        if ext == ".pdf":
            from pypdf import PdfReader
            import io
            reader = PdfReader(filepath)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        elif ext in (".docx", ".doc"):
            import docx
            doc = docx.Document(filepath)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            with open(filepath, "r", errors="ignore") as f:
                text = f.read()

        return self.ingest_text(text, source=path.name)

    def _index_chunks(self, chunks: List[Dict]):
        """Build or update FAISS index with new chunks."""
        self.chunks.extend(chunks)
        texts = [c["text"] for c in chunks]
        metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

        if self.vectorstore is None:
            self.vectorstore = LangchainFAISS.from_texts(
                texts, self._get_embeddings(), metadatas=metadatas
            )
        else:
            self.vectorstore.add_texts(texts, metadatas=metadatas)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        if not self.vectorstore:
            return []
        k = top_k or self.top_k
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            }
            for doc, score in docs_with_scores
        ]

    def query(self, question: str, top_k: int = None) -> Dict:
        """Full RAG pipeline: retrieve → augment → generate."""
        t0 = time.perf_counter()

        if not self.vectorstore:
            return {
                "answer": "No documents indexed yet. Please ingest documents first.",
                "sources": [],
                "latency_ms": 0,
                "retrieved_chunks": 0,
            }

        retrieved = self.retrieve(question, top_k)
        context = "\n\n".join(r["text"] for r in retrieved)

        prompt = ENGINEERING_QA_PROMPT.format(context=context, question=question)
        answer = self._get_llm().invoke(prompt)

        latency = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer.strip(),
            "sources": [r["metadata"].get("source", "unknown") for r in retrieved],
            "retrieved_chunks": len(retrieved),
            "top_scores": [round(r["score"], 4) for r in retrieved],
            "latency_ms": round(latency, 2),
            "context_length": len(context),
        }

    def get_retriever(self):
        """Return LangChain retriever for use in agents."""
        if not self.vectorstore:
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def get_stats(self) -> Dict:
        return {
            "documents_indexed": self.documents_indexed,
            "total_chunks": len(self.chunks),
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "vectorstore_ready": self.vectorstore is not None,
        }
