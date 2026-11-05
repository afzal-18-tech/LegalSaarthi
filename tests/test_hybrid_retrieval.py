"""
LegalSaarthi — Hybrid Retrieval Tests (Phase 11)
"""
import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


class MockRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k):
        return self.results[:top_k]


class TestHybridRetriever:
    def test_rrf_fusion(self):
        bm25 = MockRetriever([("c1", 10.0), ("c2", 8.0), ("c3", 5.0)])
        dense = MockRetriever([("c3", 0.9), ("c1", 0.8), ("c4", 0.7)])

        hybrid = HybridRetriever(bm25, dense, fusion_method="rrf", k_rrf=1)

        # Expected RRF calculation (k=1):
        # c1: 1/(1+1) + 1/(1+2) = 1/2 + 1/3 = 5/6 = 0.833
        # c2: 1/(1+2) + 0 = 1/3 = 0.333
        # c3: 1/(1+3) + 1/(1+1) = 1/4 + 1/2 = 3/4 = 0.75
        # c4: 0 + 1/(1+3) = 1/4 = 0.25
        # Order should be: c1, c3, c2, c4

        results = hybrid.retrieve("query", top_k=4)
        assert len(results) == 4
        assert results[0][0] == "c1"
        assert results[1][0] == "c3"
        assert results[2][0] == "c2"
        assert results[3][0] == "c4"
        assert results[0][1] == pytest.approx(5 / 6)

    def test_weighted_fusion(self):
        bm25 = MockRetriever([("c1", 10.0), ("c2", 0.0)])
        dense = MockRetriever([("c2", 1.0), ("c1", 0.0)])

        hybrid = HybridRetriever(bm25, dense, fusion_method="weighted", alpha=0.5)

        # Normalized:
        # bm25: c1=1.0, c2=0.0
        # dense: c2=1.0, c1=0.0
        # With alpha=0.5:
        # c1 = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # c2 = 0.5 * 0.0 + 0.5 * 1.0 = 0.5

        results = hybrid.retrieve("query", top_k=2)
        assert len(results) == 2
        scores = {r[0]: r[1] for r in results}
        assert scores["c1"] == pytest.approx(0.5)
        assert scores["c2"] == pytest.approx(0.5)

    def test_invalid_fusion(self):
        bm25 = MockRetriever([])
        dense = MockRetriever([])
        with pytest.raises(ValueError):
            HybridRetriever(bm25, dense, fusion_method="invalid")
