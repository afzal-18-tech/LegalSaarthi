"""
LegalSaarthi — Hybrid Retriever (Phase 11)

Fuses results from BM25 and Dense retrievers.
Implements Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.
"""
from typing import Dict, List, Tuple

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.logging_setup import get_logger

logger = get_logger("retrieval.hybrid")


class HybridRetriever:
    """Fuses results from lexical and semantic retrievers."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fusion_method: str = "rrf",
        k_rrf: int = 60,
        alpha: float = 0.5,
        candidate_k: int = 50,
    ):
        """
        Args:
            bm25_retriever: Initialized and built BM25 retriever.
            dense_retriever: Initialized and built Dense retriever.
            fusion_method: 'rrf' or 'weighted'.
            k_rrf: Constant for RRF formula (default 60 is standard).
            alpha: Weight for BM25 in 'weighted' fusion (1 - alpha for Dense).
            candidate_k: Number of top results to retrieve from each base retriever before fusion.
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.fusion_method = fusion_method
        self.k_rrf = k_rrf
        self.alpha = alpha
        self.candidate_k = candidate_k

        if self.fusion_method not in ["rrf", "weighted"]:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks using hybrid fusion.
        """
        # Retrieve candidates
        bm25_results = self.bm25.retrieve(query, top_k=self.candidate_k)
        dense_results = self.dense.retrieve(query, top_k=self.candidate_k)

        if self.fusion_method == "rrf":
            fused = self._rrf_fusion(bm25_results, dense_results)
        elif self.fusion_method == "weighted":
            fused = self._weighted_fusion(bm25_results, dense_results)
        else:
            fused = []

        # Sort descending and return top_k
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def _rrf_fusion(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}

        # BM25 ranks
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (self.k_rrf + rank)

        # Dense ranks
        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (self.k_rrf + rank)

        return list(scores.items())

    def _weighted_fusion(
        self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Weighted score fusion (requires min-max normalization)."""
        scores: Dict[str, float] = {}

        def _normalize(results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
            if not results:
                return []
            vals = [s for _, s in results]
            min_v, max_v = min(vals), max(vals)
            if max_v - min_v == 0:
                return [(cid, 1.0) for cid, _ in results]
            return [(cid, (s - min_v) / (max_v - min_v)) for cid, s in results]

        norm_bm25 = _normalize(bm25_results)
        norm_dense = _normalize(dense_results)

        # Add BM25
        for chunk_id, score in norm_bm25:
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += self.alpha * score

        # Add Dense
        for chunk_id, score in norm_dense:
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += (1.0 - self.alpha) * score

        return list(scores.items())
