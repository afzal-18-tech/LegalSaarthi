#!/usr/bin/env python3
"""
LegalSaarthi — Retrieval Baselines Script (Phase 10)

Builds BM25 and Dense indices, evaluates both against the gold benchmark,
and generates a comparative report.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.retrieval_metrics import evaluate_retriever
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger("retrieval.baselines")


def load_chunks(chunks_path: Path):
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def generate_report(bm25_results, dense_results, hybrid_results, output_path: Path):
    """Generate a Markdown comparison report."""
    lines = [
        "# Retrieval Baselines Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overall Comparison",
        "",
        "| Metric | BM25 | Dense (MiniLM) | Hybrid (RRF) |",
        "|--------|------|----------------|--------------|",
    ]

    bm25_agg = bm25_results["aggregates"]["overall"]
    dense_agg = dense_results["aggregates"]["overall"]
    hybrid_agg = hybrid_results["aggregates"]["overall"] if hybrid_results else {}

    for metric_key in ["mrr", "recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]:
        bm25_val = bm25_agg.get(metric_key, 0)
        dense_val = dense_agg.get(metric_key, 0)
        hybrid_val = hybrid_agg.get(metric_key, 0) if hybrid_agg else 0
        lines.append(f"| {metric_key} | {bm25_val:.4f} | {dense_val:.4f} | {hybrid_val:.4f} |")

    # Breakdown by difficulty
    lines.extend(["", "## Breakdown by Difficulty", ""])

    for diff in ["Easy", "Medium", "Hard"]:
        bm25_d = bm25_results["aggregates"].get(diff, {})
        dense_d = dense_results["aggregates"].get(diff, {})
        hybrid_d = hybrid_results["aggregates"].get(diff, {}) if hybrid_results else {}
        if not bm25_d:
            continue

        lines.extend([
            f"### {diff} (n={bm25_d.get('count', 0)})",
            "",
            f"| Metric | BM25 | Dense | Hybrid |",
            f"|--------|------|-------|--------|",
        ])

        for metric_key in ["mrr", "recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]:
            bm25_val = bm25_d.get(metric_key, 0)
            dense_val = dense_d.get(metric_key, 0)
            hybrid_val = hybrid_d.get(metric_key, 0) if hybrid_d else 0
            lines.append(f"| {metric_key} | {bm25_val:.4f} | {dense_val:.4f} | {hybrid_val:.4f} |")

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval baselines.")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense retrieval (useful for quick testing).")
    parser.add_argument("--output-dir", type=str, default="experiments/phase19", help="Directory for indices and results")
    parser.add_argument("--report-path", type=str, default=None, help="Path for report markdown")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    # Paths
    chunks_path = PROJECT_ROOT / "data/processed/chunks/strategy=section_aware/chunks.jsonl"
    benchmark_path = PROJECT_ROOT / "data/benchmark/qa_ground_truth.json"
    experiment_dir = PROJECT_ROOT / args.output_dir
    report_path = Path(args.report_path) if args.report_path else experiment_dir / "retrieval_report.md"

    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading chunks...")
    chunks = load_chunks(chunks_path)
    logger.info(f"Loaded {len(chunks)} chunks.")

    logger.info("Loading benchmark...")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(benchmark)} queries.")

    top_k_values = config.get("retrieval", {}).get("evaluation", {}).get("top_k_values", [5, 10, 20])

    # ─── BM25 Baseline ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BM25 BASELINE")
    logger.info("=" * 60)

    bm25 = BM25Retriever()
    bm25_index_dir = experiment_dir / "bm25_index"
    if (bm25_index_dir / "corpus_tokens.pkl").exists() and (bm25_index_dir / "chunk_ids.json").exists():
        logger.info(f"Loading existing BM25 index from {bm25_index_dir}...")
        bm25.load_index(bm25_index_dir)
    else:
        bm25.build_index(chunks)
        bm25.save_index(bm25_index_dir)

    bm25_results = evaluate_retriever(bm25, benchmark, top_k_values)

    bm25_agg = bm25_results["aggregates"]["overall"]
    logger.info(f"BM25 — MRR: {bm25_agg['mrr']:.4f}, Recall@10: {bm25_agg['recall@10']:.4f}, nDCG@10: {bm25_agg['ndcg@10']:.4f}")

    # Save raw results
    with open(experiment_dir / "bm25_results.json", "w") as f:
        json.dump(bm25_results, f, indent=2)

    # ─── Dense Baseline ────────────────────────────────────
    if args.skip_dense:
        logger.info("Skipping dense retrieval (--skip-dense flag).")
        dense_results = None
    else:
        logger.info("=" * 60)
        logger.info("DENSE BASELINE (sentence-transformers)")
        logger.info("=" * 60)

        model_name = config.get("retrieval", {}).get("dense", {}).get("model_name", "all-MiniLM-L6-v2")
        batch_size = config.get("retrieval", {}).get("dense", {}).get("batch_size", 64)

        dense = DenseRetriever(model_name=model_name)
        dense_index_dir = experiment_dir / "dense_index"
        if (dense_index_dir / "faiss_index.bin").exists() and (dense_index_dir / "metadata.json").exists():
            logger.info(f"Loading existing Dense index from {dense_index_dir}...")
            dense.load_index(dense_index_dir)
        else:
            dense.build_index(chunks, batch_size=batch_size)
            dense.save_index(dense_index_dir)

        dense_results = evaluate_retriever(dense, benchmark, top_k_values)

        dense_agg = dense_results["aggregates"]["overall"]
        logger.info(f"Dense — MRR: {dense_agg['mrr']:.4f}, Recall@10: {dense_agg['recall@10']:.4f}, nDCG@10: {dense_agg['ndcg@10']:.4f}")

        with open(experiment_dir / "dense_results.json", "w") as f:
            json.dump(dense_results, f, indent=2)
            
    # ─── Hybrid Baseline ───────────────────────────────────
    if args.skip_dense:
        hybrid_results = None
    else:
        logger.info("=" * 60)
        logger.info("HYBRID BASELINE (RRF)")
        logger.info("=" * 60)
        
        hybrid_config = config.get("retrieval", {}).get("hybrid", {})
        fusion_method = hybrid_config.get("fusion_method", "rrf")
        k_rrf = hybrid_config.get("k_rrf", 60)
        alpha = hybrid_config.get("alpha", 0.5)
        candidate_k = hybrid_config.get("candidate_k", 50)
        
        hybrid = HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            fusion_method=fusion_method,
            k_rrf=k_rrf,
            alpha=alpha,
            candidate_k=candidate_k
        )
        
        hybrid_results = evaluate_retriever(hybrid, benchmark, top_k_values)
        
        hybrid_agg = hybrid_results["aggregates"]["overall"]
        logger.info(f"Hybrid — MRR: {hybrid_agg['mrr']:.4f}, Recall@10: {hybrid_agg['recall@10']:.4f}, nDCG@10: {hybrid_agg['ndcg@10']:.4f}")

        with open(experiment_dir / "hybrid_results.json", "w") as f:
            json.dump(hybrid_results, f, indent=2)

    # ─── Report ────────────────────────────────────────────
    if dense_results and hybrid_results:
        generate_report(bm25_results, dense_results, hybrid_results, report_path)

    logger.info("=" * 60)
    logger.info("RETRIEVAL BASELINES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
