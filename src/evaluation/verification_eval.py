"""
LegalSaarthi — Phase 16 Verification Evaluation Engine.

Evaluates CitationVerifier and ClaimVerifier against ground-truth
adversarial perturbation benchmarks. Computes confusion matrix,
precision, recall, specificity, F1, and breakdown by perturbation class.
"""
from typing import Any, Dict, List, Optional
import time

from src.generation.verified_rag_pipeline import CitationVerifier, ClaimVerifier


def evaluate_citation_verifier(
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate CitationVerifier against citation test cases.

    Checks:
      - Detection of invalid (hallucinated / out-of-pool) chunk IDs.
      - Preservation of valid chunk IDs.
      - Exact count of stripped hallucinated tokens.
    """
    total = len(test_cases)
    exact_matches = 0
    total_true_invalid = 0
    detected_invalid = 0
    false_alarm_invalid = 0
    total_stripped_correct = 0

    per_type_stats: Dict[str, Dict[str, int]] = {}

    for tc in test_cases:
        ptype = tc["perturbation_type"]
        if ptype not in per_type_stats:
            per_type_stats[ptype] = {"total": 0, "correct": 0}
        per_type_stats[ptype]["total"] += 1

        res = CitationVerifier.verify(tc["answer_text"], tc["retrieved_ids"])

        valid_match = set(res["valid_ids"]) == set(tc["true_valid_ids"])
        invalid_match = set(res["invalid_ids"]) == set(tc["true_invalid_ids"])
        strip_match = res["citations_stripped"] == tc["expected_stripped_count"]

        if valid_match and invalid_match and strip_match:
            exact_matches += 1
            per_type_stats[ptype]["correct"] += 1

        # Track invalid detection
        true_inv = set(tc["true_invalid_ids"])
        pred_inv = set(res["invalid_ids"])
        total_true_invalid += len(true_inv)
        detected_invalid += len(true_inv & pred_inv)
        false_alarm_invalid += len(pred_inv - true_inv)
        if strip_match:
            total_stripped_correct += 1

    inv_recall = detected_invalid / total_true_invalid if total_true_invalid else 1.0
    inv_prec = (
        detected_invalid / (detected_invalid + false_alarm_invalid)
        if (detected_invalid + false_alarm_invalid)
        else 1.0
    )
    inv_f1 = (
        2 * inv_prec * inv_recall / (inv_prec + inv_recall)
        if (inv_prec + inv_recall)
        else 1.0
    )

    return {
        "total_test_cases": total,
        "exact_match_accuracy": exact_matches / total if total else 0.0,
        "invalid_citations_total": total_true_invalid,
        "invalid_citations_detected": detected_invalid,
        "invalid_detection_recall": inv_recall,
        "invalid_detection_precision": inv_prec,
        "invalid_detection_f1": inv_f1,
        "stripping_accuracy": total_stripped_correct / total if total else 0.0,
        "by_perturbation_type": {
            ptype: s["correct"] / s["total"] if s["total"] else 0.0
            for ptype, s in per_type_stats.items()
        },
    }


def evaluate_claim_verifier(
    verifier: ClaimVerifier,
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate ClaimVerifier against ground-truth claims and adversarial mutations.

    Computes:
      - Confusion matrix: TP, FP, TN, FN
      - Precision, Recall, Specificity (True Negative Rate), F1
      - Accuracy by perturbation class (entity_swap, numeric_swap, negation_inversion, unrelated_doctrine)
    """
    tp = 0  # True: SUPPORTED, Pred: SUPPORTED
    fp = 0  # True: UNSUPPORTED, Pred: SUPPORTED (critical failure: hallucination accepted)
    tn = 0  # True: UNSUPPORTED, Pred: UNSUPPORTED (success: hallucination caught)
    fn = 0  # True: SUPPORTED, Pred: UNSUPPORTED (false alarm)
    unknowns = 0

    results: List[Dict[str, Any]] = []
    by_ptype: Dict[str, Dict[str, int]] = {}
    latencies: List[float] = []

    for tc in test_cases:
        ptype = tc["perturbation_type"]
        if ptype not in by_ptype:
            by_ptype[ptype] = {"total": 0, "correct": 0}
        by_ptype[ptype]["total"] += 1

        t0 = time.time()
        res = verifier.verify_claims([tc["claim"]], tc["evidence_text"])
        lat = time.time() - t0
        latencies.append(lat)

        verdict = res["verdicts"][0] if res["verdicts"] else "UNKNOWN"
        gt = tc["ground_truth"]

        is_correct = (verdict == gt)
        if is_correct:
            by_ptype[ptype]["correct"] += 1

        if gt == "SUPPORTED":
            if verdict == "SUPPORTED":
                tp += 1
            elif verdict == "UNSUPPORTED":
                fn += 1
            else:
                unknowns += 1
        else:  # ground_truth == "UNSUPPORTED"
            if verdict == "UNSUPPORTED":
                tn += 1
            elif verdict == "SUPPORTED":
                fp += 1
            else:
                unknowns += 1

        results.append({
            "test_id": tc["test_id"],
            "perturbation_type": ptype,
            "claim": tc["claim"],
            "ground_truth": gt,
            "predicted_verdict": verdict,
            "correct": is_correct,
            "latency_s": lat,
        })

    total = len(test_cases)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "total_test_cases": total,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "unknowns": unknowns,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity_tnr": specificity,
        "f1_score": f1,
        "latency_median_s": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
        "accuracy_by_perturbation_type": {
            ptype: s["correct"] / s["total"] if s["total"] else 0.0
            for ptype, s in by_ptype.items()
        },
        "counts_by_perturbation_type": {
            ptype: s["total"] for ptype, s in by_ptype.items()
        },
        "detailed_results": results,
    }
