from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch

from advanced_rag import AdvancedRAGEngine, RAGConfig

from .config import ServiceSettings
from .llm import build_llm_callable
from .vector_store import ChromaVectorStore


class RAGService:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._engine: AdvancedRAGEngine | None = None
        self._vector_store: ChromaVectorStore | None = None
        self._index_path: Path | None = None
        self.settings.documents_dir.mkdir(parents=True, exist_ok=True)
        self.settings.documents_registry_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def engine(self) -> AdvancedRAGEngine:
        if self._engine is None:
            raise RuntimeError("RAG index is not loaded. Call /index/load first.")
        return self._engine

    def status(self) -> dict[str, Any]:
        if self._engine is None:
            return {
                "loaded": False,
                "chunk_count": 0,
                "embedding_dim": None,
                "index_path": str(self._index_path) if self._index_path else None,
                "vector_backend": self.settings.vector_backend,
                "vector_store_path": str(self.settings.chroma_persist_dir)
                if self.settings.vector_backend == "chroma"
                else None,
                "collection_name": self.settings.chroma_collection_name
                if self.settings.vector_backend == "chroma"
                else None,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }

        embedding_dim = None
        if self._engine.embeddings is not None and self._engine.embeddings.ndim == 2:
            embedding_dim = int(self._engine.embeddings.shape[1])
        elif self._vector_store is not None:
            embedding_dim = self._vector_store.embedding_dim

        return {
            "loaded": True,
            "chunk_count": self._vector_store.count if self._vector_store is not None else len(self._engine.chunks),
            "embedding_dim": embedding_dim,
            "index_path": str(self._index_path) if self._index_path else None,
            "vector_backend": self.settings.vector_backend,
            "vector_store_path": str(self.settings.chroma_persist_dir)
            if self._vector_store is not None
            else None,
            "collection_name": self.settings.chroma_collection_name
            if self._vector_store is not None
            else None,
            "device": self._engine.device,
        }

    def load_index(
        self,
        csv_path: str | None = None,
        embedding_model_name: str | None = None,
        reranker_model_name: str | None = None,
        use_default_reranker: bool = True,
        reset_vector_store: bool | None = None,
    ) -> dict[str, Any]:
        path = Path(csv_path) if csv_path else self.settings.index_path
        resolved_reranker = self.settings.reranker_model_name if use_default_reranker else reranker_model_name
        config = RAGConfig(
            embedding_model_name=embedding_model_name or self.settings.embedding_model_name,
            reranker_model_name=resolved_reranker,
            model_local_files_only=self.settings.model_local_files_only,
        )

        engine = AdvancedRAGEngine(config=config)
        vector_store: ChromaVectorStore | None = None

        if self.settings.vector_backend == "chroma":
            vector_store = ChromaVectorStore(
                persist_dir=self.settings.chroma_persist_dir,
                collection_name=self.settings.chroma_collection_name,
            )
            vector_store.import_embeddings_csv(
                path,
                reset=self.settings.reset_vector_store_on_load
                if reset_vector_store is None
                else reset_vector_store,
            )
        elif self.settings.vector_backend == "memory":
            engine.load_index_from_csv(path)
        else:
            raise ValueError(f"Unsupported RAG_VECTOR_BACKEND: {self.settings.vector_backend}")

        with self._lock:
            self._engine = engine
            self._vector_store = vector_store
            self._index_path = path

        return self.status()

    def list_documents(self) -> dict[str, Any]:
        return {"documents": list(self._load_registry().values())}

    def get_document(self, document_id: str) -> dict[str, Any]:
        registry = self._load_registry()
        if document_id not in registry:
            raise KeyError(f"Document not found: {document_id}")
        return registry[document_id]

    def upload_document(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        if not content:
            raise ValueError("Uploaded document is empty.")

        safe_filename = self._safe_filename(filename)
        document_id = uuid.uuid4().hex
        saved_path = self.settings.documents_dir / f"{document_id}-{safe_filename}"
        saved_path.write_bytes(content)

        pages = self._extract_document_pages(saved_path, content_type)
        if not pages:
            saved_path.unlink(missing_ok=True)
            raise ValueError("No text could be extracted from the uploaded document.")

        resolved_source = source.strip() if source and source.strip() else safe_filename
        docs = [
            {
                "text": page["text"],
                "document_id": document_id,
                "filename": safe_filename,
                "source": resolved_source,
                "page_number": page.get("page_number"),
            }
            for page in pages
        ]

        engine = self._ensure_engine()
        chunks = engine.build_chunks_from_texts(docs, min_tokens=8)
        if not chunks:
            saved_path.unlink(missing_ok=True)
            raise ValueError("Document text was too small to create searchable chunks.")

        for index, chunk in enumerate(chunks):
            chunk.chunk_id = f"{document_id}-chunk-{index:05d}"
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = safe_filename
            chunk.metadata["source"] = resolved_source

        texts = [chunk.text for chunk in chunks]
        embeddings = engine._get_embedding_model().encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding_matrix = AdvancedRAGEngine._normalize_embedding_matrix(
            [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]
        )

        if self.settings.vector_backend == "chroma":
            vector_store = self._ensure_vector_store()
            vector_store.upsert_texts(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=texts,
                embeddings=[embedding.astype(np.float32).tolist() for embedding in embedding_matrix],
                metadatas=[chunk.metadata for chunk in chunks],
            )
        elif self.settings.vector_backend == "memory":
            for index, embedding in enumerate(embedding_matrix):
                chunks[index].embedding = embedding
            engine.build_index_from_chunks([*engine.chunks, *chunks])
        else:
            raise ValueError(f"Unsupported RAG_VECTOR_BACKEND: {self.settings.vector_backend}")

        record = {
            "document_id": document_id,
            "filename": safe_filename,
            "content_type": content_type,
            "size_bytes": len(content),
            "source": resolved_source,
            "status": "indexed",
            "chunk_count": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        registry = self._load_registry()
        registry[document_id] = record
        self._save_registry(registry)
        return {**record, "message": "Document uploaded and indexed."}

    def delete_document(self, document_id: str) -> dict[str, Any]:
        registry = self._load_registry()
        if document_id not in registry:
            raise KeyError(f"Document not found: {document_id}")

        if self._vector_store is not None:
            self._vector_store.delete_document(document_id)
        elif self.settings.vector_backend == "chroma":
            self._ensure_vector_store().delete_document(document_id)

        for path in self.settings.documents_dir.glob(f"{document_id}-*"):
            path.unlink(missing_ok=True)

        removed = registry.pop(document_id)
        self._save_registry(registry)
        return removed

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if self._vector_store is not None:
            return self._retrieve_from_vector_store(
                query=query,
                top_k=top_k,
                filters=filters,
            )
        return self.engine.retrieve(query=query, top_k=top_k, filters=filters, use_cache=use_cache)

    def query(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        max_context_chars: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        engine = self.engine
        original_max_context_chars = engine.config.max_context_chars
        if max_context_chars is not None:
            engine.config.max_context_chars = max_context_chars

        try:
            if self._vector_store is not None:
                return self._answer_from_vector_store(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    use_cache=use_cache,
                )

            llm_callable = build_llm_callable(self.settings)
            return engine.answer(
                query=query,
                llm_callable=llm_callable,
                filters=filters,
                top_k=top_k,
                use_cache=use_cache,
            )
        finally:
            engine.config.max_context_chars = original_max_context_chars

    def _retrieve_from_vector_store(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        engine = self.engine
        if self._vector_store is None:
            raise RuntimeError("Vector store is not initialized.")

        rewritten_query = engine.rewrite_query(query)
        chosen_top_k = top_k or engine.adaptive_top_k(query)
        if chosen_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = engine._get_embedding_model().encode(
            rewritten_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        search_results = self._vector_store.query(
            query_embedding=query_embedding,
            top_k=chosen_top_k,
            filters=filters,
        )

        results = []
        for index, item in enumerate(search_results):
            text = engine._mask_pii(item.text) if engine.config.pii_masking_enabled else item.text
            results.append(
                {
                    "chunk_id": item.chunk_id,
                    "index": index,
                    "text": text,
                    "metadata": item.metadata,
                    "dense_score": item.score,
                    "bm25_score": 0.0,
                    "fusion_score": item.score,
                    "rerank_score": item.score,
                }
            )

        confidence = 0.0
        if results:
            confidence = float(sum(item["rerank_score"] for item in results) / len(results))

        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "top_k": chosen_top_k,
            "confidence": confidence,
            "results": results,
        }

    def _answer_from_vector_store(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        engine = self.engine
        retrieval = self._retrieve_from_vector_store(query=query, top_k=top_k, filters=filters)
        confidence = retrieval["confidence"]

        if not retrieval["results"] or confidence < engine.config.abstain_threshold:
            return {
                "query": query,
                "rewritten_query": retrieval["rewritten_query"],
                "answer": "I do not have enough grounded evidence in the retrieved context to answer reliably.",
                "abstained": True,
                "confidence": confidence,
                "citations": [],
                "retrieval": retrieval,
            }

        context = engine.build_cited_context(retrieval)
        prompt = (
            "You are a grounded assistant. Use only the cited context below. "
            "If the answer is not present, say so clearly.\n\n"
            f"Question: {query}\n\n"
            "CITED CONTEXT:\n"
            f"{context}\n\n"
            "Return:\n"
            "1) Direct answer\n"
            "2) Evidence bullets with chunk IDs\n"
            "3) Unknowns\n"
        )

        raw_answer = build_llm_callable(self.settings)(prompt)
        citations = [
            {
                "chunk_id": item["chunk_id"],
                "page_number": item.get("metadata", {}).get("page_number"),
                "source": item.get("metadata", {}).get("source"),
                "rerank_score": item.get("rerank_score"),
            }
            for item in retrieval["results"]
        ]

        return {
            "query": query,
            "rewritten_query": retrieval["rewritten_query"],
            "answer": raw_answer,
            "abstained": False,
            "confidence": confidence,
            "citations": citations,
            "retrieval": retrieval,
        }

    def _ensure_engine(self) -> AdvancedRAGEngine:
        if self._engine is None:
            config = RAGConfig(
                embedding_model_name=self.settings.embedding_model_name,
                reranker_model_name=self.settings.reranker_model_name,
                model_local_files_only=self.settings.model_local_files_only,
            )
            with self._lock:
                if self._engine is None:
                    self._engine = AdvancedRAGEngine(config=config)
        return self._engine

    def _ensure_vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    self._vector_store = ChromaVectorStore(
                        persist_dir=self.settings.chroma_persist_dir,
                        collection_name=self.settings.chroma_collection_name,
                    )
        return self._vector_store

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if not self.settings.documents_registry_path.exists():
            return {}
        with self.settings.documents_registry_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {}
        return {str(key): dict(value) for key, value in data.items()}

    def _save_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        with self.settings.documents_registry_path.open("w", encoding="utf-8") as file:
            json.dump(registry, file, indent=2, sort_keys=True)

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename or "document.txt").name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        return name or "document.txt"

    @staticmethod
    def _extract_document_pages(path: Path, content_type: str | None = None) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".pdf" or content_type == "application/pdf":
            try:
                import fitz
            except ImportError as exc:
                raise RuntimeError("PyMuPDF is required for PDF uploads.") from exc

            pages = []
            with fitz.open(path) as doc:
                for page_index, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append({"page_number": page_index, "text": text})
            return pages

        if suffix in {".txt", ".md", ".markdown"} or (content_type or "").startswith("text/"):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            return [{"page_number": None, "text": text}] if text else []

        raise ValueError("Unsupported document type. Upload PDF, TXT, or Markdown files.")
