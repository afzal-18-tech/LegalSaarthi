"""
LegalSaarthi — BM25 Retriever (Phase 10)

Lexical retrieval baseline using BM25Okapi over tokenized chunk texts.
"""
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi

from src.utils.logging_setup import get_logger

logger = get_logger("retrieval.bm25")


def simple_tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """BM25Okapi-based lexical retriever."""

    def __init__(self):
        self.index: BM25Okapi = None
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from chunk dicts.

        Args:
            chunks: List of chunk dicts with 'chunk_id' and 'text' keys.
        """
        self.chunk_ids = []
        self.corpus_tokens = []

        for chunk in chunks:
            self.chunk_ids.append(chunk["chunk_id"])
            tokens = simple_tokenize(chunk["text"])
            self.corpus_tokens.append(tokens)

        self.index = BM25Okapi(self.corpus_tokens)
        logger.info(f"BM25 index built with {len(self.chunk_ids)} documents.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks for a query.

        Returns:
            List of (chunk_id, score) tuples, sorted by descending score.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_tokens = simple_tokenize(query)
        scores = self.index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]

    def save_index(self, path: Path) -> None:
        """Persist index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25_index.pkl", "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "corpus_tokens": self.corpus_tokens,
            }, f)
        logger.info(f"BM25 index saved to {path}")

    def load_index(self, path: Path) -> None:
        """Load index from disk."""
        path = Path(path)
        with open(path / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.corpus_tokens = data["corpus_tokens"]
        self.index = BM25Okapi(self.corpus_tokens)
        logger.info(f"BM25 index loaded from {path} ({len(self.chunk_ids)} docs)")
