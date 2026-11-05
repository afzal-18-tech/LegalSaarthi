"""
LegalSaarthi — Phase 16 Verification Evaluation Unit Tests.

Offline tests for:
- Adversarial claim perturbation generators (numeric, entity, polarity, unrelated)
- Citation benchmark generation and ground truth formatting
- CitationVerifier metrics calculation (precision, recall, stripping accuracy)
- ClaimVerifier confusion matrix and classification metrics
"""
import pytest
from src.evaluation.verification_benchmark import (
    generate_citation_test_cases,
    generate_claim_test_cases,
    mutate_entity,
    mutate_numeric,
    mutate_polarity,
)
from src.evaluation.verification_eval import (
    evaluate_citation_verifier,
    evaluate_claim_verifier,
)
from src.generation.verified_rag_pipeline import ClaimVerifier


def _sample_chunks():
    return [
        {
            "chunk_id": "chunk_001",
            "case_id": "2015 INSC 101",
            "text": "The appellant was convicted under Section 302 of the Indian Penal Code in year 2012. The High Court dismissed the appeal and upheld the sentence of 10 years imprisonment. The Supreme Court observed that the prosecution failed to prove motive beyond reasonable doubt. The appeal was allowed and the appellant was acquitted.",
        },
        {
            "chunk_id": "chunk_002",
            "case_id": "2015 INSC 102",
            "text": "The Bank of Agriculture instituted proceedings against the respondent claiming damages of Rs 500000. The trial court found that the plaintiff established adverse possession for 12 years. The Supreme Court held that unauthorized possession cannot extinguish title.",
        },
    ]


class MockClaimLLM:
    """Mock LLM client returning predictable verdicts."""
    def __init__(self, mode="perfect"):
        self.mode = mode
        self.last_latency_s = 0.05
        self.last_model_used = "mock-llm"

    def generate(self, prompt, system_prompt=None):
        lines = []
        # Find how many claims are in prompt
        count = prompt.count("CLAIM ")
        for i in range(1, count + 1):
            if self.mode == "perfect":
                # Check if ground truth was supported
                lines.append(f"CLAIM {i}: SUPPORTED" if "acquitted" in prompt or "appellant was convicted" in prompt else f"CLAIM {i}: UNSUPPORTED")
            elif self.mode == "all_supported":
                lines.append(f"CLAIM {i}: SUPPORTED")
            else:
                lines.append(f"CLAIM {i}: UNSUPPORTED")
        return "\n".join(lines)


# ── Perturbation Generator Tests ───────────────────────────────────────────

def test_mutate_numeric():
    text = "The sentence was 10 years under Section 302."
    mutated = mutate_numeric(text)
    assert mutated is not None
    assert mutated != text
    assert ("23 years" in mutated or "498" in mutated or "years" in mutated)


def test_mutate_entity():
    text = "The High Court ruled in favour of the appellant."
    mutated = mutate_entity(text)
    assert mutated is not None
    assert "respondent" in mutated or "Trial Court" in mutated


def test_mutate_polarity():
    text = "The appeal was dismissed by the court."
    mutated = mutate_polarity(text)
    assert mutated is not None
    assert "allowed" in mutated


def test_generate_claim_test_cases():
    chunks = _sample_chunks()
    cases = generate_claim_test_cases(chunks, n_cases=6, seed=42)
    assert len(cases) > 0
    labels = {c["ground_truth"] for c in cases}
    assert "SUPPORTED" in labels
    assert "UNSUPPORTED" in labels
    ptypes = {c["perturbation_type"] for c in cases}
    assert "none" in ptypes


def test_generate_citation_test_cases():
    chunks = _sample_chunks()
    cases = generate_citation_test_cases(chunks, n_cases=4, seed=42)
    assert len(cases) == 4
    for c in cases:
        assert "answer_text" in c
        assert "retrieved_ids" in c
        assert "true_valid_ids" in c
        assert "true_invalid_ids" in c


# ── Verifier Evaluator Tests ───────────────────────────────────────────────

def test_evaluate_citation_verifier():
    chunks = _sample_chunks()
    cases = generate_citation_test_cases(chunks, n_cases=8, seed=42)
    results = evaluate_citation_verifier(cases)
    assert results["total_test_cases"] == 8
    assert results["exact_match_accuracy"] == 1.0
    assert results["invalid_detection_recall"] == 1.0
    assert results["invalid_detection_precision"] == 1.0
    assert results["stripping_accuracy"] == 1.0


def test_evaluate_claim_verifier_metrics():
    test_cases = [
        {
            "test_id": "T1",
            "perturbation_type": "none",
            "evidence_text": "The appellant was convicted.",
            "claim": "The appellant was convicted.",
            "ground_truth": "SUPPORTED",
        },
        {
            "test_id": "T2",
            "perturbation_type": "entity_swap",
            "evidence_text": "The appellant was convicted.",
            "claim": "The respondent was convicted.",
            "ground_truth": "UNSUPPORTED",
        },
    ]

    mock_llm = MockClaimLLM(mode="perfect")
    verifier = ClaimVerifier(mock_llm)
    results = evaluate_claim_verifier(verifier, test_cases)

    assert results["total_test_cases"] == 2
    assert results["accuracy"] >= 0.5
    cm = results["confusion_matrix"]
    assert "true_positives" in cm
    assert "true_negatives" in cm
    assert "false_positives" in cm
    assert "false_negatives" in cm
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["specificity_tnr"] <= 1.0
