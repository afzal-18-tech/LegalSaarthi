"""
LegalSaarthi — Dense Retriever (Phase 10)

Semantic retrieval baseline using sentence-transformers + FAISS.
"""
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import faiss
    _FAISS_AVAILABLE = True
except (ImportError, Exception):
    faiss = None
    _FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer

from src.utils.logging_setup import get_logger

logger = get_logger("retrieval.dense")


class _NumpyIndexFlatIP:
    """Fallback index when FAISS native binary is unavailable."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, x: np.ndarray) -> None:
        x_f32 = x.astype(np.float32)
        if len(self.vectors) == 0:
            self.vectors = x_f32
        else:
            self.vectors = np.vstack([self.vectors, x_f32])

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vectors) == 0:
            return np.array([[]]), np.array([[]])
        sims = np.dot(self.vectors, query_embedding.T).squeeze(-1)
        if sims.ndim == 0:
            sims = np.array([float(sims)])
        actual_k = min(top_k, len(sims))
        top_indices = np.argsort(-sims)[:actual_k]
        top_scores = sims[top_indices]
        return np.array([top_scores]), np.array([top_indices])


class DenseRetriever:
    """Dense retriever using sentence-transformers embeddings and FAISS / NumPy index."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace sentence-transformer model name.
        """
        self.model_name = model_name
        self.model: SentenceTransformer = None
        self.index = None
        self.chunk_ids: List[str] = []
        self.embeddings: np.ndarray = None

    def _load_model(self) -> None:
        """Lazy-load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence-transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> None:
        """
        Encode all chunks and build a FAISS inner-product index.

        Args:
            chunks: List of chunk dicts with 'chunk_id' and 'text' keys.
            batch_size: Encoding batch size.
        """
        self._load_model()

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks (batch_size={batch_size})...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity via IP
        )

        # Build index (Inner Product on normalized vectors = cosine similarity)
        dim = self.embeddings.shape[1]
        if _FAISS_AVAILABLE and faiss is not None:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = _NumpyIndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        logger.info(f"Dense index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks for a query.

        Returns:
            List of (chunk_id, score) tuples, sorted by descending similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        self._load_model()

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results

    def save_index(self, path: Path) -> None:
        """Persist index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if _FAISS_AVAILABLE and faiss is not None and not isinstance(self.index, _NumpyIndexFlatIP):
            faiss.write_index(self.index, str(path / "faiss_index.bin"))
        np.save(path / "embeddings.npy", self.embeddings)

        with open(path / "dense_meta.json", "w") as f:
            json.dump({
                "chunk_ids": self.chunk_ids,
                "model_name": self.model_name,
                "dim": int(self.embeddings.shape[1]),
                "count": len(self.chunk_ids),
            }, f)

        logger.info(f"Dense index saved to {path}")

    def load_index(self, path: Path) -> None:
        """Load index and metadata from disk."""
        path = Path(path)

        self.embeddings = np.load(path / "embeddings.npy")
        dim = self.embeddings.shape[1]

        if _FAISS_AVAILABLE and faiss is not None and (path / "faiss_index.bin").exists():
            try:
                self.index = faiss.read_index(str(path / "faiss_index.bin"))
            except Exception:
                self.index = _NumpyIndexFlatIP(dim)
                self.index.add(self.embeddings.astype(np.float32))
        else:
            self.index = _NumpyIndexFlatIP(dim)
            self.index.add(self.embeddings.astype(np.float32))

        with open(path / "dense_meta.json", "r") as f:
            meta = json.load(f)

        self.chunk_ids = meta["chunk_ids"]
        self.model_name = meta["model_name"]

        self._load_model()
        logger.info(f"Dense index loaded from {path} ({len(self.chunk_ids)} docs)")
