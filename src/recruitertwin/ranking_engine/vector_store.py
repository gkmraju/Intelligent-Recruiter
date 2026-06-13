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

import numpy as np

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None
    FAISS_AVAILABLE = False


class FaissVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._vecs: np.ndarray | None = None  # numpy fallback storage
        self.index = faiss.IndexFlatIP(self.dim) if FAISS_AVAILABLE else None
        self.backend = "faiss" if FAISS_AVAILABLE else "numpy"

    # ------------------------------------------------------------------ #
    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected [n, {self.dim}] matrix, got {vectors.shape}")
        if self.index is not None:
            self.index.add(vectors)
        if self._vecs is None:
            self._vecs = vectors
        else:
            self._vecs = np.vstack([self._vecs, vectors])

    @property
    def ntotal(self) -> int:
        return 0 if self._vecs is None else int(self._vecs.shape[0])

    # ------------------------------------------------------------------ #
    def search(self, query: np.ndarray, k: int = 100):
        """Top-k nearest (by cosine) → (ids [k], scores [k])."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, self.dim)
        k = min(k, self.ntotal)
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return ids[0], scores[0]
        sims = (self._vecs @ q[0])
        ids = np.argsort(-sims)[:k]
        return ids, sims[ids]

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query to *every* stored vector,
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
