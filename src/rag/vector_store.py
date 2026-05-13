"""
Vector Store
============
FAISS-based vector store for the RAG knowledge base.
Stores radiology report embeddings and enables fast similarity search.

Supports two embedding sources:
  - CLIP:    single-vector per document (standard dense retrieval)
  - ColPali: multi-vector per document (stored as flattened; MaxSim at query time)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS vector store for radiology report retrieval.

    Features:
    - Build index from CLIP embeddings (single-vector, IndexFlatIP)
    - Persist/load index + metadata to/from disk
    - Similarity search returning top-K results with metadata
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        embedding_dim: int = 1024,
    ):
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.embedding_dim = embedding_dim

        self._index = None
        self._metadata: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> "FAISSVectorStore":
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: (N, dim) float32 array, L2-normalized
            metadata: N metadata dicts (must include "text" key)

        Returns:
            self
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        assert len(embeddings) == len(metadata), "Embeddings and metadata must have same length"
        assert embeddings.ndim == 2

        self.embedding_dim = embeddings.shape[1]
        embeddings = embeddings.astype(np.float32)

        # Inner product index (equivalent to cosine sim for normalized vectors)
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(embeddings)
        self._metadata = metadata
        self._embeddings = embeddings

        logger.info(f"Built FAISS index with {len(metadata)} documents (dim={self.embedding_dim})")
        return self

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """
        Search for top-K similar documents.

        Args:
            query_embeddings: (Q, dim) float32 array
            top_k: Number of results per query

        Returns:
            List of Q lists, each with top_k result dicts
            Each result has: rank, score, doc_idx, text, ...metadata
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        query_embeddings = query_embeddings.astype(np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[np.newaxis, :]

        scores, indices = self._index.search(query_embeddings, min(top_k, len(self._metadata)))

        results = []
        for q_idx in range(len(query_embeddings)):
            q_results = []
            for rank, (idx, score) in enumerate(zip(indices[q_idx], scores[q_idx])):
                if idx == -1:
                    continue
                result = {
                    "rank": rank + 1,
                    "score": float(score),
                    "doc_idx": int(idx),
                    **self._metadata[idx],
                }
                q_results.append(result)
            results.append(q_results)

        return results

    #  Persist 

    def save(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed.")

        idx_path = Path(index_path or self.index_path)
        meta_path = Path(metadata_path or self.metadata_path)

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(idx_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index → {idx_path}")
        logger.info(f"Saved metadata → {meta_path}")

    def load(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> "FAISSVectorStore":
        """Load FAISS index and metadata from disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed.")

        idx_path = Path(index_path or self.index_path)
        meta_path = Path(metadata_path or self.metadata_path)

        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")

        self._index = faiss.read_index(str(idx_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self.embedding_dim = self._index.d
        logger.info(f"Loaded FAISS index from {idx_path} ({len(self._metadata)} docs)")
        return self

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._metadata)

    @property
    def is_built(self) -> bool:
        return self._index is not None


class ColPaliVectorStore:
    """
    In-memory store for ColPali multi-vector document embeddings.

    ColPali produces variable-length multi-vector embeddings (one vector
    per image patch). These cannot be stored in a standard FAISS flat index
    without flattening. Instead we store them as a list of tensors and
    compute MaxSim at query time (exact, not approximate).

    For large corpora (>10k docs), consider using PLAID (ColBERT FAISS) or
    approximate MaxSim.
    """

    def __init__(self):
        self._doc_embeddings: list = []  # list of (Ld, dim) tensors
        self._metadata: list[dict] = []

    def add(
        self,
        doc_embeddings: list,
        metadata: list[dict],
    ) -> None:
        """
        Add document embeddings to the store.

        Args:
            doc_embeddings: List of tensors (Ld_i, dim) for each document
            metadata: List of metadata dicts
        """
        assert len(doc_embeddings) == len(metadata)
        self._doc_embeddings.extend(doc_embeddings)
        self._metadata.extend(metadata)
        logger.info(f"ColPali store now has {len(self._metadata)} documents")

    def search_with_query_embs(
        self,
        query_embs,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """
        Retrieve top-K documents using MaxSim against stored embeddings.

        Args:
            query_embs: (Q, Lq, dim) tensor
            top_k: Number of results

        Returns:
            List of Q lists with top_k result dicts
        """
        if not self._doc_embeddings:
            raise RuntimeError("No documents in store.")

        import torch

        # Stack docs: list of (Ld_i, dim) → pad to same length
        Ld_max = max(e.shape[0] for e in self._doc_embeddings)
        dim = self._doc_embeddings[0].shape[-1]
        D = len(self._doc_embeddings)

        padded = torch.zeros(D, Ld_max, dim)
        for i, emb in enumerate(self._doc_embeddings):
            padded[i, : emb.shape[0], :] = emb

        # MaxSim scoring
        q = torch.nn.functional.normalize(query_embs.float(), dim=-1)   # (Q, Lq, dim)
        d = torch.nn.functional.normalize(padded.float(), dim=-1)       # (D, Ld, dim)

        q_exp = q.unsqueeze(1)  # (Q, 1, Lq, dim)
        d_exp = d.unsqueeze(0)  # (1, D, Ld, dim)

        sim = torch.matmul(q_exp, d_exp.transpose(-1, -2))  # (Q, D, Lq, Ld)
        max_sim = sim.max(dim=-1).values.sum(dim=-1)         # (Q, D)

        results = []
        for q_idx in range(max_sim.shape[0]):
            scores = max_sim[q_idx]
            topk = scores.topk(min(top_k, D))
            q_results = [
                {
                    "rank": r + 1,
                    "score": float(topk.values[r]),
                    "doc_idx": int(topk.indices[r]),
                    **self._metadata[int(topk.indices[r])],
                }
                for r in range(len(topk.indices))
            ]
            results.append(q_results)

        return results

    def save(self, path: str) -> None:
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self._doc_embeddings, "metadata": self._metadata}, f)
        logger.info(f"Saved ColPali store → {path}")

    def load(self, path: str) -> "ColPaliVectorStore":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._doc_embeddings = data["embeddings"]
        self._metadata = data["metadata"]
        logger.info(f"Loaded ColPali store from {path} ({len(self._metadata)} docs)")
        return self

    @property
    def size(self) -> int:
        return len(self._metadata)
