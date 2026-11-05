"""
LegalSaarthi — Retrieval Metrics (Phase 10)

Standard IR evaluation metrics: Recall@k, MRR, nDCG@k.
"""
import math
from typing import Any, Dict, List, Set, Tuple

from src.utils.logging_setup import get_logger

logger = get_logger("evaluation.retrieval_metrics")


def recall_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """
    Fraction of gold evidence chunks found in top-k retrieved results.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        gold_ids: Set of gold (relevant) chunk IDs.
        k: Cutoff rank.

    Returns:
        Recall@k value in [0.0, 1.0].
    """
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for rid in top_k if rid in gold_ids)
    return found / len(gold_ids)


def mrr(retrieved_ids: List[str], gold_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank — reciprocal of the rank of the first relevant result.

    Returns:
        MRR value in [0.0, 1.0]. 0.0 if no relevant result found.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at rank k.
    Uses binary relevance (1 if in gold, 0 otherwise).

    Returns:
        nDCG@k value in [0.0, 1.0].
    """
    # DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG: all relevant documents at the top
    ideal_count = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_retriever(
    retriever,
    benchmark: List[Dict[str, Any]],
    top_k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a retriever against the gold benchmark.

    Args:
        retriever: Object with a `retrieve(query, top_k)` method.
        benchmark: List of benchmark query dicts from qa_ground_truth.json.
        top_k_values: List of k values to evaluate.

    Returns:
        Dict with per-query results and aggregate metrics.
    """
    if top_k_values is None:
        top_k_values = [5, 10, 20]

    max_k = max(top_k_values)
    per_query_results = []

    for query_entry in benchmark:
        query_id = query_entry["query_id"]
        query_text = query_entry["question_text"]
        gold_ids = set(query_entry["gold_evidence"])
        difficulty = query_entry.get("difficulty", "Unknown")

        # Retrieve with max k
        retrieved = retriever.retrieve(query_text, top_k=max_k)
        retrieved_ids = [rid for rid, _ in retrieved]

        # Compute metrics at each k
        query_metrics = {
            "query_id": query_id,
            "difficulty": difficulty,
            "gold_count": len(gold_ids),
            "mrr": mrr(retrieved_ids, gold_ids),
        }
        for k in top_k_values:
            query_metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
            query_metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, gold_ids, k)

        per_query_results.append(query_metrics)

    # Aggregate metrics
    aggregates = _compute_aggregates(per_query_results, top_k_values)

    return {
        "per_query": per_query_results,
        "aggregates": aggregates,
    }


def _compute_aggregates(
    per_query: List[Dict[str, Any]],
    top_k_values: List[int],
) -> Dict[str, Any]:
    """Compute mean metrics overall and by difficulty."""
    difficulties = sorted(set(q["difficulty"] for q in per_query))

    def _mean_metrics(queries: List[Dict]) -> Dict[str, float]:
        n = len(queries)
        if n == 0:
            return {}
        result = {"count": n, "mrr": sum(q["mrr"] for q in queries) / n}
        for k in top_k_values:
            result[f"recall@{k}"] = sum(q[f"recall@{k}"] for q in queries) / n
            result[f"ndcg@{k}"] = sum(q[f"ndcg@{k}"] for q in queries) / n
        return result

    agg = {"overall": _mean_metrics(per_query)}

    for diff in difficulties:
        subset = [q for q in per_query if q["difficulty"] == diff]
        agg[diff] = _mean_metrics(subset)

    return agg
