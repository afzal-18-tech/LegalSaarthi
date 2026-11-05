"""
LegalSaarthi — Retrieval Tests (Phase 10)

Tests for BM25Retriever, DenseRetriever, and retrieval metrics.
"""
import pytest

from src.evaluation.retrieval_metrics import recall_at_k, mrr, ndcg_at_k


# ─── Metric Tests ───────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c", "d"]
        gold = {"a", "b"}
        assert recall_at_k(retrieved, gold, k=4) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "y", "b"]
        gold = {"a", "b", "c"}
        assert recall_at_k(retrieved, gold, k=2) == pytest.approx(1 / 3)

    def test_no_recall(self):
        retrieved = ["x", "y", "z"]
        gold = {"a", "b"}
        assert recall_at_k(retrieved, gold, k=3) == 0.0

    def test_empty_gold(self):
        retrieved = ["a", "b"]
        assert recall_at_k(retrieved, set(), k=2) == 0.0


class TestMRR:
    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        gold = {"a"}
        assert mrr(retrieved, gold) == 1.0

    def test_second_position(self):
        retrieved = ["x", "a", "c"]
        gold = {"a"}
        assert mrr(retrieved, gold) == 0.5

    def test_no_match(self):
        retrieved = ["x", "y", "z"]
        gold = {"a"}
        assert mrr(retrieved, gold) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        # Both gold items at positions 1 and 2
        retrieved = ["a", "b", "x", "y"]
        gold = {"a", "b"}
        assert ndcg_at_k(retrieved, gold, k=4) == pytest.approx(1.0)

    def test_imperfect_ndcg(self):
        # Gold item "a" at position 3 instead of 1
        retrieved = ["x", "y", "a"]
        gold = {"a"}
        # DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
        expected = (1.0 / 2.0) / 1.0  # 0.5
        assert ndcg_at_k(retrieved, gold, k=3) == pytest.approx(expected)

    def test_no_relevant(self):
        retrieved = ["x", "y", "z"]
        gold = {"a"}
        assert ndcg_at_k(retrieved, gold, k=3) == 0.0


# ─── BM25 Retriever Tests ───────────────────────────────────

class TestBM25Retriever:
    @pytest.fixture
    def mini_corpus(self):
        return [
            {"chunk_id": "c1", "text": "The Supreme Court held that the appeal is dismissed."},
            {"chunk_id": "c2", "text": "Under Section 302 of the Indian Penal Code, the accused was convicted."},
            {"chunk_id": "c3", "text": "The plaintiff filed a civil suit for recovery of damages."},
            {"chunk_id": "c4", "text": "The High Court upheld the order of the trial court."},
            {"chunk_id": "c5", "text": "Constitutional validity of the amendment was challenged."},
        ]

    def test_build_and_retrieve(self, mini_corpus):
        from src.retrieval.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        retriever.build_index(mini_corpus)

        results = retriever.retrieve("Supreme Court appeal dismissed", top_k=3)
        assert len(results) == 3
        # The first result should be c1 (most relevant)
        assert results[0][0] == "c1"
        # Each result is (chunk_id, score) tuple
        assert all(isinstance(r[1], float) for r in results)

    def test_save_and_load(self, mini_corpus, tmp_path):
        from src.retrieval.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        retriever.build_index(mini_corpus)

        retriever.save_index(tmp_path / "bm25_test")

        loaded = BM25Retriever()
        loaded.load_index(tmp_path / "bm25_test")

        r1 = retriever.retrieve("appeal dismissed", top_k=3)
        r2 = loaded.retrieve("appeal dismissed", top_k=3)
        assert [r[0] for r in r1] == [r[0] for r in r2]


# ─── Dense Retriever Tests ──────────────────────────────────

class TestDenseRetriever:
    @pytest.fixture
    def mini_corpus(self):
        return [
            {"chunk_id": "c1", "text": "The Supreme Court held that the appeal is dismissed."},
            {"chunk_id": "c2", "text": "Under Section 302 of the Indian Penal Code, the accused was convicted."},
            {"chunk_id": "c3", "text": "The plaintiff filed a civil suit for recovery of damages."},
            {"chunk_id": "c4", "text": "The High Court upheld the order of the trial court."},
            {"chunk_id": "c5", "text": "Constitutional validity of the amendment was challenged."},
        ]

    @pytest.mark.slow
    def test_build_and_retrieve(self, mini_corpus):
        from src.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
        retriever.build_index(mini_corpus, batch_size=2)

        results = retriever.retrieve("appeal dismissed by Supreme Court", top_k=3)
        assert len(results) == 3
        # c1 should rank highly
        retrieved_ids = [r[0] for r in results]
        assert "c1" in retrieved_ids
        # Scores should be floats
        assert all(isinstance(r[1], float) for r in results)

    @pytest.mark.slow
    def test_save_and_load(self, mini_corpus, tmp_path):
        from src.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
        retriever.build_index(mini_corpus, batch_size=2)

        retriever.save_index(tmp_path / "dense_test")

        loaded = DenseRetriever()
        loaded.load_index(tmp_path / "dense_test")

        r1 = retriever.retrieve("appeal", top_k=3)
        r2 = loaded.retrieve("appeal", top_k=3)
        assert [r[0] for r in r1] == [r[0] for r in r2]
