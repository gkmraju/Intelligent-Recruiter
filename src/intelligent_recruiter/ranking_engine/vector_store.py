"""FAISS-backed vector store for the dense re-ranking layer.

Indexes L2-normalized embeddings in a ``faiss.IndexFlatIP`` (exact inner
product == cosine similarity on normalized vectors). For the shortlist
sizes used here (~1.5K vectors) flat exact search is the right tool: no
training step, no recall loss, sub-millisecond queries.

If ``faiss`` is not installed, falls back to a NumPy matrix product with
the identical API, so the pipeline never breaks.

Usage::

    store = FaissVectorStore(dim=384)
    store.add(doc_vecs)                  # [n, dim] float32, L2-normalized
    sims = store.similarities(q_vec)     # [n] cosine sims, input order
    ids, scores = store.search(q_vec, k=100)   # top-k retrieval
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def _load_faiss() -> Any | None:
    try:
        return importlib.import_module("faiss")
    except ImportError:  # pragma: no cover
        return None


_FAISS_MODULE = _load_faiss()
FAISS_AVAILABLE: bool = _FAISS_MODULE is not None


class FaissVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        # _vecs is used only when FAISS is unavailable (NumPy fallback).
        # When FAISS is present we do NOT duplicate storage there.
        self._vecs: np.ndarray | None = None
        faiss_module = _FAISS_MODULE
        self.index: Any | None = (
            faiss_module.IndexFlatIP(self.dim) if faiss_module is not None else None
        )
        self.backend = "faiss" if FAISS_AVAILABLE else "numpy"
        self._ntotal: int = 0

    # ------------------------------------------------------------------ #
    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected [n, {self.dim}] matrix, got {vectors.shape}")
        if self.index is not None:
            self.index.add(vectors)
        else:
            # NumPy fallback: accumulate vectors for dot-product search.
            self._vecs = vectors if self._vecs is None else np.vstack([self._vecs, vectors])
        self._ntotal += len(vectors)

    @property
    def ntotal(self) -> int:
        return self._ntotal

    # ------------------------------------------------------------------ #
    def search(self, query: np.ndarray, k: int = 100):
        """Top-k nearest (by cosine) -> (ids [k], scores [k])."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, self.dim)
        k = min(k, self._ntotal)
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return ids[0], scores[0]
        assert self._vecs is not None
        sims = (self._vecs @ q[0])
        ids = np.argsort(-sims)[:k]
        return ids, sims[ids]

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query to every stored vector,
        returned in insertion order (used for score blending)."""
        ids, scores = self.search(query, k=self.ntotal)
        out = np.zeros(self.ntotal, dtype=np.float32)
        out[ids] = scores
        return out


def build_vector_index(vectors: np.ndarray) -> FaissVectorStore:
    """Convenience constructor kept for backwards compatibility."""
    store = FaissVectorStore(dim=vectors.shape[1])
    store.add(vectors)
    return store
