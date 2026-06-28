"""Lightweight embedding layer for Stage 2 re-ranking.

Two backends, picked automatically:

1. MiniLM, loaded from the local ``models/`` directory when available.
2. LSA fallback, a pure-scikit-learn TF-IDF plus TruncatedSVD embedding.

Either way the output is a matrix of L2-normalized vectors, so cosine
similarity equals inner product, which is exactly what the FAISS store in
``vector_store.py`` indexes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "all-MiniLM-L6-v2"
MAX_CHARS = 2000  # roughly 512 tokens; summary + recent roles carry the signal
LSA_DIM = 256
_MINILM_MODEL: Any | None = None


def minilm_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return MODEL_DIR.exists()


def _get_minilm_model() -> Any:
    global _MINILM_MODEL

    if _MINILM_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MINILM_MODEL = SentenceTransformer(str(MODEL_DIR), device="cpu")
    return _MINILM_MODEL


class LightweightEmbedder:
    """Embeds the JD query + candidate texts into one normalized space."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "auto":
            self.backend = "minilm" if minilm_available() else "lsa"
        elif backend == "minilm" and not minilm_available():
            self.backend = "lsa"
        else:
            self.backend = backend
        self._model = None

    def encode_corpus_and_query(
        self,
        texts: list[str],
        query: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (doc_matrix [n, d], query_vector [d]), both L2-normalized."""
        if self.backend == "minilm":
            return self._encode_minilm(texts, query)
        return self._encode_lsa(texts, query)

    def _encode_minilm(
        self,
        texts: list[str],
        query: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            self._model = _get_minilm_model()
        docs = [t[:MAX_CHARS] for t in texts]
        doc_vecs = self._model.encode(
            docs,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        q_vec = self._model.encode(
            [query[:MAX_CHARS]],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)
        return doc_vecs, q_vec

    def _encode_lsa(
        self,
        texts: list[str],
        query: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts + [query])
        return _encode_lsa_from_tfidf_matrix(X)


def _encode_lsa_from_tfidf_matrix(X: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized LSA document/query vectors from a TF-IDF matrix.

    The matrix must contain candidate rows followed by the query as the final
    row, matching ``TfidfVectorizer.fit_transform(texts + [query])``.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    dim = min(LSA_DIM, X.shape[1] - 1, X.shape[0] - 1)
    if dim < 2:  # degenerate tiny corpus: use raw TF-IDF rows
        normalized = cast(Any, normalize(X))
        dense = normalized.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=dim, random_state=42)
        projected = cast(Any, svd.fit_transform(X))
        dense = cast(Any, normalize(projected)).astype(np.float32)
    return dense[:-1], dense[-1]


def _minmax_similarities(doc_vecs: np.ndarray, q_vec: np.ndarray) -> list[float]:
    from intelligent_recruiter.ranking_engine.vector_store import FaissVectorStore

    store = FaissVectorStore(dim=doc_vecs.shape[1])
    store.add(doc_vecs)
    sims = store.similarities(q_vec)  # aligned with input order

    lo, hi = float(sims.min()), float(sims.max())
    if hi <= lo:
        return [0.0] * len(doc_vecs)
    return [float((s - lo) / (hi - lo)) for s in sims]


def semantic_similarities_from_tfidf_matrix(X: Any) -> list[float]:
    """LSA semantic similarities from an existing TF-IDF matrix.

    This avoids fitting a second vectorizer in the ranking pipeline while
    preserving the same LSA projection used by the fallback embedder.
    """
    doc_vecs, q_vec = _encode_lsa_from_tfidf_matrix(X)
    return _minmax_similarities(doc_vecs, q_vec)


def semantic_similarities(
    texts: list[str],
    query: str,
    backend: str = "auto",
) -> list[float] | None:
    """Cosine similarity of each text to the query, min-max normalized."""
    if backend == "none":
        return None

    emb = LightweightEmbedder(backend=backend)
    doc_vecs, q_vec = emb.encode_corpus_and_query(texts, query)
    return _minmax_similarities(doc_vecs, q_vec)
