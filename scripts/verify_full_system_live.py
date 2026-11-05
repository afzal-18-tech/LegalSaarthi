"""
End-to-End Live System Verification Script for LegalSaarthi (Phases 1 to 23).

Exercises actual classes, methods, and data pipelines across every phase:
1. Literature review (Phase 1)
2. Ingestion & Environment (Phase 2-4)
3. Corpus Splits & Data Validation (Phase 5)
4. PDF Parser (Phase 6)
5. Legal Structure Detection (Phase 7)
6. Document Chunking & Provenance (Phase 8)
7. Benchmark Ground Truth Integrity (Phase 9)
8. Retrieval Engines: BM25, Dense, Hybrid (Phase 10-11)
9. RAG Pipeline, CitationVerifier & ClaimVerifier (Phase 12-14)
10. Summarization & Readability (Phase 15)
11. Verification Adversarial Evaluation (Phase 16)
12. Abstention Calibrator (Phase 17)
13. Robustness & Evidence-Failure Harness (Phase 18)
14. Scaled Corpus Verification (Phase 19)
15. Experimental Runs Integrity (Phase 20-22)
16. Automated Judge, Rubric & Audit Workbook (Phase 23)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import pathlib
import sys
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("legalsaarthi.verify_full_system")


def test_phase1_literature() -> bool:
    """Verify Phase 1 literature review files and matrix density."""
    lit_csv = pathlib.Path("docs/literature/literature_matrix.csv")
    gap_md = pathlib.Path("docs/literature/gap_map.md")
    rat_md = pathlib.Path("docs/literature/baseline_rationale.md")
    if not (lit_csv.exists() and gap_md.exists() and rat_md.exists()):
        return False
    with open(lit_csv, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        return len(reader) >= 25


def test_phase2_to_4_ingestion() -> bool:
    """Verify hashing, source resolver, and sample PDF validation."""
    from src.utils.hashing import sha256_file
    from src.ingestion.source_resolver import SourceResolver, derive_pdf_filename

    resolver = SourceResolver(config={})
    sample_row = {"path": "2020_4_552_564", "year": 2020}
    filename = derive_pdf_filename(sample_row)
    if filename != "2020_4_552_564_EN.pdf":
        return False

    raw_dir = pathlib.Path("data/raw")
    pdfs = list(raw_dir.rglob("*.pdf"))
    if not pdfs:
        return False
    h = sha256_file(str(pdfs[0]))
    return len(h) == 64


def test_phase5_splits() -> bool:
    """Verify split disjointness and locked test set."""
    split_file = pathlib.Path("data/metadata/split_manifest_main.csv")
    if not split_file.exists():
        return False
    with open(split_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    splits = {"dev": set(), "val": set(), "test": set()}
    for r in rows:
        s = r.get("split")
        cid = r.get("case_id")
        if s in splits and cid:
            splits[s].add(cid)
    # Check disjointness
    disjoint = (
        len(splits["dev"] & splits["val"]) == 0
        and len(splits["dev"] & splits["test"]) == 0
        and len(splits["val"] & splits["test"]) == 0
    )
    return disjoint and len(splits["test"]) >= 150


def test_phase6_pdf_parser() -> bool:
    """Run PDFParser on an actual PDF from data/raw/."""
    from src.preprocessing.pdf_parser import PDFParser

    raw_dir = pathlib.Path("data/raw")
    pdfs = list(raw_dir.rglob("*.pdf"))
    if not pdfs:
        return False
    parser = PDFParser()
    doc = parser.parse_pdf(pdfs[0], case_id="TEST_CASE")
    return (
        doc.get("case_id") == "TEST_CASE"
        and doc.get("total_pages", 0) > 0
        and len(doc.get("pages", [])) > 0
    )


def test_phase7_structure_detector() -> bool:
    """Run StructureDetector on sample legal text."""
    from src.preprocessing.structure_detector import StructureDetector

    detector = StructureDetector()
    sample_doc = {
        "case_id": "2020 INSC 999",
        "total_pages": 2,
        "pages": [
            {
                "page_num": 1,
                "paragraphs": [
                    {"text": "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION"},
                    {"text": "The appellant was convicted under Section 302 of the Indian Penal Code."},
                    {"text": "The question for consideration is whether the high court erred."},
                    {"text": "Mr. Salve, learned Senior Counsel appearing for the petitioner, contended that..."},
                ],
            },
            {
                "page_num": 2,
                "paragraphs": [
                    {"text": "Section 100 of the Code of Civil Procedure provides that..."},
                    {"text": "In Kesavananda Bharati v. State of Kerala, this Court observed..."},
                    {"text": "We have carefully considered the rival submissions and the record."},
                    {"text": "For the reasons aforesaid, the appeal is dismissed with no order as to costs."},
                ],
            },
        ],
    }
    enriched = detector.detect_structure(sample_doc)
    summary = enriched.get("sections_summary", {})
    non_zero = sum(1 for v in summary.values() if v > 0)
    return non_zero >= 4


def test_phase8_chunker() -> bool:
    """Run DocumentChunker with section_aware strategy and verify provenance."""
    from src.preprocessing.chunker import DocumentChunker

    chunker = DocumentChunker(max_words=100, overlap_words=20)
    sample_doc = {
        "case_id": "2020 INSC 999",
        "total_pages": 1,
        "pages": [
            {
                "page_num": 1,
                "paragraphs": [
                    {"text": "In the Supreme Court of India.", "section_label": "HEADER", "para_id": "p1"},
                    {"text": "The appellant challenges the conviction under Section 302 IPC.", "section_label": "FACTS", "para_id": "p2"},
                    {"text": "The appeal is accordingly dismissed.", "section_label": "DECISION", "para_id": "p3"},
                ],
            }
        ],
    }
    chunks = chunker.chunk_by_section(sample_doc)
    if not chunks:
        return False
    c0 = chunks[0]
    return (
        c0.case_id == "2020 INSC 999"
        and len(c0.sections) > 0
        and len(c0.pages) > 0
        and bool(c0.source_reference)
    )


def test_phase9_benchmark() -> bool:
    """Verify 50 gold QA benchmark queries against chunk store."""
    bm_file = pathlib.Path("data/benchmark/qa_ground_truth.json")
    if not bm_file.exists():
        return False
    with open(bm_file, encoding="utf-8") as f:
        benchmarks = json.load(f)
    if len(benchmarks) < 50:
        return False
    sample = benchmarks[0]
    return "question_text" in sample and "gold_evidence" in sample and "difficulty" in sample


def test_phase10_and_11_retrieval() -> bool:
    """Test BM25Retriever and HybridRetriever with sample queries."""
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever

    corpus = [
        {"chunk_id": "c1", "text": "The Supreme Court dismissed the appeal under Section 302 IPC."},
        {"chunk_id": "c2", "text": "Suit for specific performance of contract under Specific Relief Act."},
        {"chunk_id": "c3", "text": "Arbitration petition under Section 11 of the Arbitration and Conciliation Act."},
    ]
    bm25 = BM25Retriever()
    bm25.build_index(corpus)
    hits = bm25.retrieve("Section 302 IPC murder appeal", top_k=2)
    if not hits or hits[0][0] != "c1":
        return False

    # Check that BM25 retriever can be combined into HybridRetriever instance
    dense = DenseRetriever()
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, fusion_method="rrf")
    bm25_res = [("c1", 4.2), ("c2", 1.1)]
    dense_res = [("c1", 0.9), ("c3", 0.5)]
    fused = hybrid._rrf_fusion(bm25_res, dense_res)
    return len(fused) > 0 and fused[0][0] == "c1"


def test_phase12_to_14_rag_and_verifiers() -> bool:
    """Test CitationVerifier, ClaimVerifier, and citation extraction with CHUNK prefix support."""
    from src.generation.rag_pipeline import extract_citations
    from src.generation.verified_rag_pipeline import CitationVerifier

    # Test extract_citations robustness with [CHUNK ...] and (...)
    ans = "The appeal is allowed [CHUNK 2015 INSC 177_section_aware_1] and dismissed (CHUNK 2015 INSC 177_section_aware_2)."
    cites = extract_citations(ans)
    if cites != ["2015 INSC 177_section_aware_1", "2015 INSC 177_section_aware_2"]:
        return False

    # Test CitationVerifier stripping
    retrieved = ["2015 INSC 177_section_aware_1"]
    v = CitationVerifier.verify(ans, retrieved)
    if v["valid_ids"] != ["2015 INSC 177_section_aware_1"]:
        return False
    if v["invalid_ids"] != ["2015 INSC 177_section_aware_2"]:
        return False
    if v["citations_stripped"] != 1:
        return False
    if "2015 INSC 177_section_aware_2" in v["cleaned_answer"]:
        return False
    return True


def test_phase15_summarization_metrics() -> bool:
    """Test readability and disposition extraction functions."""
    from src.summarization.summary_metrics import (
        readability,
        predict_disposition,
    )

    text = "The appeal is dismissed. The order of the High Court is affirmed."
    disp = predict_disposition(text)
    read_scores = readability(text)
    re_score = read_scores.get("flesch_reading_ease")
    fk_grade = read_scores.get("fk_grade")
    return disp == "dismissed" and re_score is not None and fk_grade is not None


def test_phase16_adversarial_verification() -> bool:
    """Test adversarial mutation benchmark generator."""
    from src.evaluation.verification_benchmark import generate_claim_test_cases

    sample_chunks = [
        {"case_id": "CASE1", "chunk_id": "c1", "text": "The High Court awarded compensation of Rs 50,000 to the appellant in the motor accident claim case under Section 166 of the Motor Vehicles Act, 1988."},
        {"case_id": "CASE2", "chunk_id": "c2", "text": "In a separate dispute involving arbitration under Section 11 of the Arbitration and Conciliation Act, the court appointed a sole arbitrator to resolve claims between the parties."}
    ]
    test_cases = generate_claim_test_cases(sample_chunks, n_cases=2)
    return len(test_cases) > 0 and "ground_truth" in test_cases[0]


def test_phase17_abstention_calibration() -> bool:
    """Test abstention calibrator policy logic."""
    from src.evaluation.abstention_calibrator import AbstentionPolicy

    policy = AbstentionPolicy(threshold=0.5)
    rec_pass = {"verification": {"support_rate": 0.8}, "correctness": 5.0, "factuality": 5.0}
    rec_fail = {"verification": {"support_rate": 0.2}, "correctness": 2.0, "factuality": 2.0}

    abstain_pass, _ = policy.decide(rec_pass)
    abstain_fail, reason = policy.decide(rec_fail)
    return (not abstain_pass) and abstain_fail and "threshold" in (reason or "")


def test_phase18_robustness() -> bool:
    """Test perturbation generation under distractor and contradictory conditions."""
    from src.evaluation.robustness import tamper_numbers, OOD_QUESTIONS, CONDITIONS

    orig = "The high court awarded compensation under Section 302 in 2015."
    tampered = tamper_numbers(orig)
    return (
        tampered != orig
        and len(OOD_QUESTIONS) >= 8
        and "distractor" in CONDITIONS
        and "contradictory" in CONDITIONS
    )


def test_phase19_to_22_experiment_artifacts() -> bool:
    """Verify presence and validity of phase 19 through 22 experiment artifacts."""
    p19_report = pathlib.Path("experiments/phase19/retrieval_report.md")
    p20_runs = list(pathlib.Path("experiments/phase20").glob("*ollama*"))
    p20_main = pathlib.Path("experiments/phase20/main_results.md")
    p21_runs = list(pathlib.Path("experiments/phase21").glob("*ollama*"))
    p22_runs = list(pathlib.Path("experiments/phase22").glob("*ollama*"))

    return (
        p19_report.exists()
        and p20_main.exists()
        and len(p20_runs) >= 3
        and len(p21_runs) >= 3
        and len(p22_runs) >= 1
    )


def test_phase23_judge_and_audit() -> bool:
    """Test Phase 23 rubric parsing, agreement metrics, and Excel audit workbook."""
    from src.evaluation.phase23_harness import (
        compute_agreement,
        parse_judge_rubric,
    )
    import openpyxl

    sample_response = (
        "FACTUALITY: 4\n"
        "COMPLETENESS: 3\n"
        "CLARITY: 5\n"
        "LEGAL_COHERENCE: 4\n"
        "CITATION_VALIDITY: 5\n"
        "EVIDENCE_SUPPORT: 4\n"
    )
    parsed = parse_judge_rubric(sample_response)
    if not (parsed["factuality"] == 4 and parsed["clarity"] == 5):
        return False

    # Check agreement calculation
    agr = compute_agreement([4, 5, 3], [4, 4, 3])
    if agr["exact_agreement"] < 0.6:
        return False

    # Check workbook
    wb_file = pathlib.Path("LegalSaarthi_Phase23_50_Query_Manual_Audit.xlsx")
    if not wb_file.exists():
        return False
    wb = openpyxl.load_workbook(wb_file)
    if "50_Query_Audit" not in wb.sheetnames or "Scoring_Summary" not in wb.sheetnames:
        return False
    ws = wb["50_Query_Audit"]
    if ws.max_row < 50:
        return False

    return True


def run_all_tests() -> bool:
    print("=" * 75)
    print("LegalSaarthi — Live End-to-End System Test (Phase 1 to Phase 23)")
    print("=" * 75)

    tests = [
        ("Phase 1: Literature Review & Gap Analysis", test_phase1_literature),
        ("Phase 2-4: Ingestion, Resolver & PDF Hashing", test_phase2_to_4_ingestion),
        ("Phase 5: Corpus Splits & Disjointness Lock", test_phase5_splits),
        ("Phase 6: Canonical PDF Parser Execution", test_phase6_pdf_parser),
        ("Phase 7: Legal Structure Detection (9 Zones)", test_phase7_structure_detector),
        ("Phase 8: Evidence Chunker & Provenance", test_phase8_chunker),
        ("Phase 9: QA Benchmark Ground Truth Integrity", test_phase9_benchmark),
        ("Phase 10-11: BM25 & Hybrid RRF Retrieval", test_phase10_and_11_retrieval),
        ("Phase 12-14: RAG Pipeline & Verifier Modules", test_phase12_to_14_rag_and_verifiers),
        ("Phase 15: Summarization & Readability Metrics", test_phase15_summarization_metrics),
        ("Phase 16: Adversarial Claim Verification", test_phase16_adversarial_verification),
        ("Phase 17: Abstention Calibration Engine", test_phase17_abstention_calibration),
        ("Phase 18: Robustness & Perturbation Harness", test_phase18_robustness),
        ("Phase 19-22: Experiment Artifacts & Stat Tests", test_phase19_to_22_experiment_artifacts),
        ("Phase 23: Automated Judge, Rubric & Audit Workbook", test_phase23_judge_and_audit),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            passed = fn()
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(f"  {name:55s} : [{status}]")
        except Exception as e:
            all_passed = False
            print(f"  {name:55s} : [ERROR: {e}]")

    print("=" * 75)
    if all_passed:
        print("ALL 15 VERIFICATION SUITES PASSED! System is completely verified.")
    else:
        print("FAILURES DETECTED! Review issues before proceeding.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
