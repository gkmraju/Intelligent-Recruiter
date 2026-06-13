"""Lightweight embedding layer for Stage 2 re-ranking.

Two backends, picked automatically:

1. **MiniLM (preferred)** — sentence-transformers/all-MiniLM-L6-v2 loaded
   from the local ``models/`` directory (run ``scripts/download_model.py``
   once with internet access). 384-dim, CPU-only, no network at rank time.

2. **LSA fallback (always available)** — a pure-scikit-learn lightweight
   embedding: TF-IDF (1-2 grams, sublinear TF) projected to a dense
   256-dim space with TruncatedSVD, then L2-normalized. Zero downloads,
   deterministic, builds in ~1s for a 1.5K-document shortlist. Captures
   term co-occurrence semantics ("vector db" ~ "FAISS index") that raw
   lexical matching misses.

Either way the output is a matrix of L2-normalized vectors, so cosine
similarity == inner product, which is exactly what the FAISS store in
``vector_store.py`` indexes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "all-MiniLM-L6-v2"
MAX_CHARS = 2000  # ~512 tokens; summary + recent roles carry the signal
LSA_DIM = 256


def minilm_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return MODEL_DIR.exists()


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
        self._pipe = None

    # ------------------------------------------------------------------ #
    def encode_corpus_and_query(
        self, texts: list[str], query: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (doc_matrix [n, d], query_vector [d]), both L2-normalized."""
        if self.backend == "minilm":
            return self._encode_minilm(texts, query)
        return self._encode_lsa(texts, query)

    # ------------------------------------------------------------------ #
    def _encode_minilm(self, texts, query):
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(str(MODEL_DIR), device="cpu")
        docs = [t[:MAX_CHARS] for t in texts]
        doc_vecs = self._model.encode(
            docs, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        q_vec = self._model.encode(
            [query[:MAX_CHARS]], convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)
        return doc_vecs, q_vec

    def _encode_lsa(self, texts, query):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                              max_features=60000, sublinear_tf=True)
        X = vec.fit_transform(texts + [query])
        dim = min(LSA_DIM, X.shape[1] - 1, X.shape[0] - 1)
        if dim < 2:  # degenerate tiny corpus — use raw tf-idf rows
            dense = normalize(X).toarray().astype(np.float32)
        else:
            svd = TruncatedSVD(n_components=dim, random_state=42)
            dense = normalize(svd.fit_transform(X)).astype(np.float32)
        return dense[:-1], dense[-1]


def semantic_similarities(
    texts: list[str],
    query: str,
    backend: str = "auto",
) -> list[float] | None:
    """Cosine similarity (via the FAISS store) of each text to the query,
    min-max normalized to [0, 1] for blending with lexical scores."""
    from recruitertwin.ranking_engine.vector_store import FaissVectorStore

    if backend == "none":
        return None

    emb = LightweightEmbedder(backend=backend)
    doc_vecs, q_vec = emb.encode_corpus_and_query(texts, query)

    store = FaissVectorStore(dim=doc_vecs.shape[1])
    store.add(doc_vecs)
    sims = store.similarities(q_vec)  # aligned with input order

    lo, hi = float(sims.min()), float(sims.max())
    if hi <= lo:
        return [0.0] * len(texts)
    return [float((s - lo) / (hi - lo)) for s in sims]
