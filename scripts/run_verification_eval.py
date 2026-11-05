"""
LegalSaarthi — Phase 16 Verification Evaluation Harness.

Evaluates the robustness, precision, recall, and adversarial sensitivity
of the CitationVerifier and ClaimVerifier against synthetic adversarial
perturbations constructed from the validation split.

Usage:
  python -m scripts.run_verification_eval --limit 10   # fast smoke test
  python -m scripts.run_verification_eval              # full evaluation
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

from src.evaluation.verification_benchmark import (
    generate_citation_test_cases,
    generate_claim_test_cases,
)
from src.evaluation.verification_eval import (
    evaluate_citation_verifier,
    evaluate_claim_verifier,
)
from src.generation.llm_client import LLMClient
from src.generation.verified_rag_pipeline import ClaimVerifier
from src.utils.config import load_config
from src.utils.logging_setup import get_logger, setup_logging

logger = get_logger("evaluation.verification_eval")

CHUNKS_PATH = Path("data/processed/chunks/strategy=section_aware/chunks.jsonl")
SPLITS_PATH = Path("data/metadata/split_manifest.csv")


def model_tag(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def load_val_chunks() -> List[Dict[str, Any]]:
    """Load chunks belonging strictly to the validation split."""
    splits = pd.read_csv(SPLITS_PATH)
    val_cases = set(splits.loc[splits["split"] == "val", "case_id"])

    val_chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("case_id") in val_cases:
                val_chunks.append(chunk)
    return val_chunks


def render_markdown_report(
    run_id: str,
    model_name: str,
    cit_metrics: Dict[str, Any],
    claim_metrics: Dict[str, Any],
) -> str:
    cm = claim_metrics["confusion_matrix"]
    lines = [
        f"# Phase 16 Verification Evaluation Report — {run_id}",
        "",
        f"- Verifier Model: `{model_name}`",
        f"- Timestamp: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 1. Citation Verifier Evaluation (Deterministic)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Test Cases | {cit_metrics['total_test_cases']} |",
        f"| Exact Match Accuracy | {cit_metrics['exact_match_accuracy']*100:.1f}% |",
        f"| Invalid Citations Total | {cit_metrics['invalid_citations_total']} |",
        f"| Invalid Citations Detected | {cit_metrics['invalid_citations_detected']} |",
        f"| Invalid Detection Recall | {cit_metrics['invalid_detection_recall']*100:.1f}% |",
        f"| Invalid Detection Precision | {cit_metrics['invalid_detection_precision']*100:.1f}% |",
        f"| Invalid Detection F1 | {cit_metrics['invalid_detection_f1']*100:.1f}% |",
        f"| Stripping Accuracy | {cit_metrics['stripping_accuracy']*100:.1f}% |",
        "",
        "### Citation Accuracy by Perturbation Type",
        "| Perturbation Type | Accuracy |",
        "|---|---|",
    ]
    for ptype, acc in cit_metrics["by_perturbation_type"].items():
        lines.append(f"| {ptype} | {acc*100:.1f}% |")

    lines += [
        "",
        "## 2. Claim-Level Entailment Verifier Evaluation (Adversarial)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Claim Test Cases | {claim_metrics['total_test_cases']} |",
        f"| Accuracy | {claim_metrics['accuracy']*100:.1f}% |",
        f"| Precision (PPV) | {claim_metrics['precision']*100:.1f}% |",
        f"| Recall / Sensitivity | {claim_metrics['recall_sensitivity']*100:.1f}% |",
        f"| Specificity (True Negative Rate) | {claim_metrics['specificity_tnr']*100:.1f}% |",
        f"| F1 Score | {claim_metrics['f1_score']*100:.1f}% |",
        f"| Median Latency (s) | {claim_metrics['latency_median_s']:.2f}s |",
        "",
        "### Confusion Matrix",
        "| | Predicted SUPPORTED | Predicted UNSUPPORTED |",
        "|---|---|---|",
        f"| **Actual SUPPORTED** | **TP: {cm['true_positives']}** | FN: {cm['false_negatives']} |",
        f"| **Actual UNSUPPORTED** | FP (Hallucination Missed): {cm['false_positives']} | **TN (Hallucination Caught): {cm['true_negatives']}** |",
        "",
        "### Claim Verifier Accuracy by Perturbation Class",
        "| Perturbation Class | Count | Verifier Accuracy |",
        "|---|---|---|",
    ]
    for ptype, acc in claim_metrics["accuracy_by_perturbation_type"].items():
        cnt = claim_metrics["counts_by_perturbation_type"].get(ptype, 0)
        lines.append(f"| {ptype} | {cnt} | {acc*100:.1f}% |")

    lines += [
        "",
        "## 3. Analysis & Key Insights (Hypothesis H3)",
        "- **Citation Verifier**: Evaluates deterministic boundary detection and stripping of hallucinated tokens.",
        "- **Claim Verifier Robustness**: Specificity directly measures whether the verifier catches adversarial modifications (entity swaps, numerical tampering, polarity inversion, and unrelated legal doctrines).",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Run Phase 16 Verification Evaluation")
    p.add_argument("--limit", type=int, default=None, help="Limit number of test cases for fast testing")
    p.add_argument("--model", default=None, help="Override verifier model")
    args = p.parse_args()

    load_dotenv()
    setup_logging()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config().get("generation", {})

    model_name = args.model or cfg["model_name"]
    n_claims = args.limit or 40
    n_citations = args.limit or 30

    run_id = f"verification-eval__{model_tag(model_name)}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.limit:
        run_id += f"__limit{args.limit}"

    out_dir = Path("experiments/phase16") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Phase 16 Verification Evaluation: {run_id}")
    logger.info(f"Model: {model_name} | Claims: {n_claims} | Citations: {n_citations}")
    logger.info("=" * 60)

    val_chunks = load_val_chunks()
    logger.info(f"Loaded {len(val_chunks)} validation chunks.")

    # 1. Generate adversarial benchmarks
    citation_tests = generate_citation_test_cases(val_chunks, n_cases=n_citations)
    claim_tests = generate_claim_test_cases(val_chunks, n_cases=n_claims)

    (out_dir / "citation_test_benchmark.json").write_text(
        json.dumps(citation_tests, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "claim_test_benchmark.json").write_text(
        json.dumps(claim_tests, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2. Evaluate CitationVerifier
    logger.info("Evaluating CitationVerifier...")
    cit_results = evaluate_citation_verifier(citation_tests)

    # 3. Evaluate ClaimVerifier
    logger.info("Evaluating ClaimVerifier...")
    verify_client = LLMClient(model_name, temperature=0.0, max_tokens=256)
    claim_verifier = ClaimVerifier(verify_client)
    claim_results = evaluate_claim_verifier(claim_verifier, claim_tests)

    # 4. Save results and report
    full_metrics = {
        "run_id": run_id,
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "citation_verifier": cit_results,
        "claim_verifier": {k: v for k, v in claim_results.items() if k != "detailed_results"},
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(full_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "claim_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in claim_results["detailed_results"]),
        encoding="utf-8",
    )

    report_md = render_markdown_report(run_id, model_name, cit_results, claim_results)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    logger.info(f"Phase 16 Evaluation complete. Saved artifacts to: {out_dir}")
    print(report_md)


if __name__ == "__main__":
    main()
