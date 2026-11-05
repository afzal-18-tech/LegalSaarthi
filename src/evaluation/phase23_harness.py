"""
Phase 23 — LLM-as-a-Judge evaluation harness.

For each of the 50 Phase 23 audit queries the workflow is:
1. Select a test-split case, retrieve its chunks.
2. Generate answers from E1 (Generic LLM), E3 (Basic RAG BM25), E5 (Verified-RAG BM25).
3. Run deterministic checks (citation validity, statutory reference presence, evidence traceability).
4. Score each answer with the LLM judge on the Phase 23 rubric (6 dimensions, 1-5 each).
5. Record per-system per-query results.
"""
from __future__ import annotations

import json
import re
import pathlib
import logging
from typing import Any

log = logging.getLogger("legalsaarthi.evaluation.phase23_harness")

# ---------------------------------------------------------------------------
# 50 Phase 23 audit question templates (from Excel)
# ---------------------------------------------------------------------------
AUDIT_QUESTIONS: list[dict[str, str]] = [
    # Facts & background (Q01-Q10)
    {"qid": "Q01", "category": "Facts & background", "prompt": "What are the material facts of this case?"},
    {"qid": "Q02", "category": "Facts & background", "prompt": "Who are the parties, and what roles do they play in the dispute?"},
    {"qid": "Q03", "category": "Facts & background", "prompt": "What key events led to the litigation?"},
    {"qid": "Q04", "category": "Facts & background", "prompt": "What procedural history is relevant to this case?"},
    {"qid": "Q05", "category": "Facts & background", "prompt": "Which facts did the court treat as established or undisputed?"},
    {"qid": "Q06", "category": "Facts & background", "prompt": "Which important facts were disputed by the parties?"},
    {"qid": "Q07", "category": "Facts & background", "prompt": "What relief or remedy did the petitioner/appellant seek?"},
    {"qid": "Q08", "category": "Facts & background", "prompt": "What was the lower court or tribunal's decision, if discussed?"},
    {"qid": "Q09", "category": "Facts & background", "prompt": "Which dates or events are essential to the court's analysis?"},
    {"qid": "Q10", "category": "Facts & background", "prompt": "Summarize the case background without adding facts absent from the judgment."},
    # Legal issues (Q11-Q18)
    {"qid": "Q11", "category": "Legal issues", "prompt": "What principal legal issue did the court decide?"},
    {"qid": "Q12", "category": "Legal issues", "prompt": "What secondary legal issues did the court consider?"},
    {"qid": "Q13", "category": "Legal issues", "prompt": "What question of law was framed or addressed?"},
    {"qid": "Q14", "category": "Legal issues", "prompt": "Which arguments or issues did the appellant raise?"},
    {"qid": "Q15", "category": "Legal issues", "prompt": "Which arguments or issues did the respondent raise?"},
    {"qid": "Q16", "category": "Legal issues", "prompt": "Did the court distinguish questions of fact from questions of law?"},
    {"qid": "Q17", "category": "Legal issues", "prompt": "Which issue was decisive to the final outcome?"},
    {"qid": "Q18", "category": "Legal issues", "prompt": "Were any issues left undecided or considered unnecessary?"},
    # Law & precedents (Q19-Q26)
    {"qid": "Q19", "category": "Law & precedents", "prompt": "Which statutory provisions did the court expressly discuss?"},
    {"qid": "Q20", "category": "Law & precedents", "prompt": "How did the court interpret the principal statutory provision?"},
    {"qid": "Q21", "category": "Law & precedents", "prompt": "Which precedents did the court rely on?"},
    {"qid": "Q22", "category": "Law & precedents", "prompt": "What principle did the court draw from a cited precedent?"},
    {"qid": "Q23", "category": "Law & precedents", "prompt": "Did the court distinguish, limit, or overrule a precedent?"},
    {"qid": "Q24", "category": "Law & precedents", "prompt": "What legal test, standard, or rule did the court apply?"},
    {"qid": "Q25", "category": "Law & precedents", "prompt": "Did the judgment discuss a constitutional provision? How was it used?"},
    {"qid": "Q26", "category": "Law & precedents", "prompt": "Which statutory conditions or exceptions mattered to the analysis?"},
    # Reasoning (Q27-Q35)
    {"qid": "Q27", "category": "Reasoning", "prompt": "What were the main steps in the court's reasoning?"},
    {"qid": "Q28", "category": "Reasoning", "prompt": "Why did the court accept or reject the key argument?"},
    {"qid": "Q29", "category": "Reasoning", "prompt": "How did the court apply the law to the facts?"},
    {"qid": "Q30", "category": "Reasoning", "prompt": "Which evidence or finding most influenced the reasoning?"},
    {"qid": "Q31", "category": "Reasoning", "prompt": "Did the court identify a limitation, exception, or qualification?"},
    {"qid": "Q32", "category": "Reasoning", "prompt": "Was there a dissenting or concurring opinion? Distinguish it from the majority."},
    {"qid": "Q33", "category": "Reasoning", "prompt": "Did the decision rely on procedural grounds, substantive grounds, or both?"},
    {"qid": "Q34", "category": "Reasoning", "prompt": "What is the strongest source-supported explanation for the conclusion?"},
    {"qid": "Q35", "category": "Reasoning", "prompt": "Did the court give separate or alternative grounds for its decision?"},
    # Decision & relief (Q36-Q40)
    {"qid": "Q36", "category": "Decision & relief", "prompt": "What was the final disposition of the case or appeal?"},
    {"qid": "Q37", "category": "Decision & relief", "prompt": "What specific order or relief did the court grant or refuse?"},
    {"qid": "Q38", "category": "Decision & relief", "prompt": "Did the court remand the matter or direct further proceedings?"},
    {"qid": "Q39", "category": "Decision & relief", "prompt": "Were costs, deadlines, conditions, or directions imposed?"},
    {"qid": "Q40", "category": "Decision & relief", "prompt": "Which party succeeded on the principal issue?"},
    # Summary (Q41-Q45)
    {"qid": "Q41", "category": "Summary", "prompt": "Summarize the judgment as facts, issues, reasoning, and decision."},
    {"qid": "Q42", "category": "Summary", "prompt": "Explain the judgment in plain language without changing its legal meaning."},
    {"qid": "Q43", "category": "Summary", "prompt": "List key findings and give source page/paragraph references."},
    {"qid": "Q44", "category": "Summary", "prompt": "What is the central holding of the judgment?"},
    {"qid": "Q45", "category": "Summary", "prompt": "Distinguish the legal principle applied from case-specific facts."},
    # Citation & evidence (Q46-Q50)
    {"qid": "Q46", "category": "Citation & evidence", "prompt": "For each major factual claim, identify the supporting passage and page/paragraph."},
    {"qid": "Q47", "category": "Citation & evidence", "prompt": "Do cited passages directly support the claims attached to them? Identify unsupported claims."},
    {"qid": "Q48", "category": "Citation & evidence", "prompt": "Are statutory sections and case citations accurate against the judgment?"},
    {"qid": "Q49", "category": "Citation & evidence", "prompt": "Are statements attributed to the correct judge/opinion, including any dissent?"},
    {"qid": "Q50", "category": "Citation & evidence", "prompt": "If evidence is insufficient, does the system acknowledge the limitation rather than guess?"},
]

# ---------------------------------------------------------------------------
# LLM-as-Judge rubric prompt
# ---------------------------------------------------------------------------
JUDGE_RUBRIC_PROMPT = """You are an expert legal evaluator assessing an AI system's answer to a question about an Indian Supreme Court judgment.

ANSWER TO EVALUATE:
{answer}

RETRIEVED EVIDENCE (chunks the system had access to):
{evidence}

QUESTION:
{question}

Score this answer on SIX dimensions. For each, give ONLY an integer from 1 to 5:
1 = Very poor: major errors or serious unsupported claims
2 = Poor: several important issues
3 = Acceptable: useful but noticeable weaknesses
4 = Good: mostly correct, minor issues
5 = Excellent: criterion met very well with source support

Scoring criteria and calibration guidelines:
- FACTUALITY: Are material facts, dates, parties, findings, and outcomes strictly aligned with the evidence? (5 = fully verified by excerpts; 3 = partial; 1-2 = material fabrication).
- COMPLETENESS: Is key information needed to answer the question present? If the answer is brief, superficial, or omits essential legal context present in evidence, score <= 3.
- CLARITY: Is the answer understandable, unambiguous, and legally precise without distortion?
- LEGAL_COHERENCE: Are facts, statutory rules, reasoning, and final outcome logically consistent?
- CITATION_VALIDITY: Do cited chunk IDs (e.g. [CHUNK xyz]) exist in the retrieved evidence? If no chunk IDs are cited at all, score 1-2. If cited IDs are valid, score 5.
- EVIDENCE_SUPPORT: Do the cited excerpts actually substantiate the specific claims made? Deduct if claims are asserted without backing.

If the answer is an abstention ("I don't know", "Insufficient information", etc.):
- Score FACTUALITY=3, COMPLETENESS=1, CLARITY=3, LEGAL_COHERENCE=3, CITATION_VALIDITY=3, EVIDENCE_SUPPORT=3
  (Abstention is honest but provides no information.)

Output ONLY these six lines and nothing else:
FACTUALITY: <1-5>
COMPLETENESS: <1-5>
CLARITY: <1-5>
LEGAL_COHERENCE: <1-5>
CITATION_VALIDITY: <1-5>
EVIDENCE_SUPPORT: <1-5>"""


def parse_judge_rubric(raw: str) -> dict[str, int | None]:
    """Parse the 6-dimension rubric output from the judge."""
    dims = ["FACTUALITY", "COMPLETENESS", "CLARITY", "LEGAL_COHERENCE", "CITATION_VALIDITY", "EVIDENCE_SUPPORT"]
    scores: dict[str, int | None] = {d.lower(): None for d in dims}
    for dim in dims:
        m = re.search(rf"{dim}:\s*([1-5])", raw, re.IGNORECASE)
        if m:
            scores[dim.lower()] = int(m.group(1))
    scores["parse_failure"] = any(v is None for v in scores.values())  # type: ignore
    return scores


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def check_citation_validity(answer: str, retrieved_chunk_ids: list[str], cited_ids: list[str]) -> dict[str, Any]:
    """Check which cited chunk IDs actually exist in the retrieved set."""
    cited_set = set(cited_ids)
    retrieved_set = set(retrieved_chunk_ids)
    valid = cited_set & retrieved_set
    invalid = cited_set - retrieved_set
    rate = len(valid) / len(cited_set) if cited_set else 1.0
    return {
        "n_cited": len(cited_set),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "valid_ids": sorted(valid),
        "invalid_ids": sorted(invalid),
        "citation_precision": round(rate, 4),
        "has_citation": bool(cited_set),
    }


def check_statutory_references(answer: str) -> dict[str, Any]:
    """Heuristic detection of statutory and case references in the answer."""
    # Statutory: "Section X", "Sec. X", "Article N", "Rule N", "Order N Rule N", core Acts
    statutory_patterns = [
        r"\bSec(?:tion)?\s*\.?\s*\d+[A-Za-z]?\b",
        r"\bArt(?:icle)?\s*\.?\s*\d+[A-Za-z]?\b",
        r"\bRule\s+\d+\b",
        r"\bOrder\s+(?:[IVXLC]+|\d+)(?:\s+Rule\s+\d+)?\b",
        r"\b(?:IPC|CrPC|CPC|Constitution of India|Indian Penal Code|Code of Criminal Procedure|Code of Civil Procedure|Evidence Act|Companies Act|Arbitration Act|Specific Relief Act|Limitation Act|Negotiable Instruments Act|Income Tax Act)\b",
        r"\b[A-Z][\w\s]{2,30}Act[,\s]+(?:\d{4}|of\s+\d{4})\b",
    ]
    found: list[str] = []
    for pat in statutory_patterns:
        found.extend(re.findall(pat, answer, re.IGNORECASE))
    # Case citations: "YYYY INSC NNNN" or "(YYYY) N SCC NNNN"
    case_pattern = r"\b\d{4}\s+INSC\s+\d+\b|\(\d{4}\)\s*\d+\s+SCC\s+\d+"
    case_refs = re.findall(case_pattern, answer, re.IGNORECASE)
    return {
        "statutory_refs": list(set(found)),
        "case_refs": list(set(case_refs)),
        "n_statutory": len(set(found)),
        "n_case_refs": len(set(case_refs)),
        "has_statutory": bool(found),
        "has_case_refs": bool(case_refs),
    }


def check_evidence_traceability(answer: str, chunks: list[dict]) -> dict[str, Any]:
    """Check how much of the answer text can be traced to retrieved chunk text."""
    if not answer or not chunks:
        return {"traceable_ngrams": 0, "total_ngrams": 0, "traceability_rate": 0.0}

    def ngrams(text: str, n: int = 5) -> set[tuple]:
        words = re.findall(r"\b\w+\b", text.lower())
        return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}

    answer_ng = ngrams(answer)
    if not answer_ng:
        return {"traceable_ngrams": 0, "total_ngrams": 0, "traceability_rate": 0.0}

    all_chunk_text = " ".join(c.get("text", "") or c.get("content", "") for c in chunks)
    chunk_ng = ngrams(all_chunk_text)
    traceable = answer_ng & chunk_ng
    rate = len(traceable) / len(answer_ng)
    return {
        "traceable_ngrams": len(traceable),
        "total_ngrams": len(answer_ng),
        "traceability_rate": round(rate, 4),
    }


def is_abstention(answer: str) -> bool:
    """Detect if the model abstained."""
    if not answer:
        return True
    low = answer.lower().strip()
    patterns = [
        "i don't know", "i do not know", "insufficient information",
        "i cannot", "i can't", "not enough information",
        "no information", "unable to determine", "insufficient evidence",
    ]
    return any(p in low for p in patterns) and len(answer) < 250


# ---------------------------------------------------------------------------
# Agreement analysis (judge scores vs manual ratings)
# ---------------------------------------------------------------------------

def compute_agreement(auto_scores: list[int], manual_scores: list[int]) -> dict[str, float]:
    """Compute agreement statistics between automated judge and manual ratings.

    Returns:
        exact_agreement: fraction of exact matches
        adjacent_agreement: fraction within ±1
        mae: mean absolute error
        pearson_r: linear correlation
    """
    n = len(auto_scores)
    if n == 0:
        return {"exact_agreement": 0.0, "adjacent_agreement": 0.0, "mae": 0.0, "pearson_r": 0.0, "n": 0}

    exact = sum(a == m for a, m in zip(auto_scores, manual_scores)) / n
    adjacent = sum(abs(a - m) <= 1 for a, m in zip(auto_scores, manual_scores)) / n
    mae = sum(abs(a - m) for a, m in zip(auto_scores, manual_scores)) / n

    # Pearson r
    import math
    mean_a = sum(auto_scores) / n
    mean_m = sum(manual_scores) / n
    num = sum((a - mean_a) * (m - mean_m) for a, m in zip(auto_scores, manual_scores))
    den_a = math.sqrt(sum((a - mean_a) ** 2 for a in auto_scores))
    den_m = math.sqrt(sum((m - mean_m) ** 2 for m in manual_scores))
    r = num / (den_a * den_m) if den_a * den_m > 0 else 0.0

    return {
        "exact_agreement": round(exact, 4),
        "adjacent_agreement": round(adjacent, 4),
        "mae": round(mae, 4),
        "pearson_r": round(r, 4),
        "n": n,
    }


def aggregate_rubric_scores(records: list[dict]) -> dict[str, Any]:
    """Aggregate rubric scores across multiple records for one system."""
    dims = ["factuality", "completeness", "clarity", "legal_coherence", "citation_validity", "evidence_support"]
    totals: dict[str, list[float]] = {d: [] for d in dims}
    n_abstained = 0
    n_parse_failures = 0

    for rec in records:
        scores = rec.get("judge_scores") or {}
        if rec.get("abstained"):
            n_abstained += 1
        if scores.get("parse_failure"):
            n_parse_failures += 1
        for d in dims:
            v = scores.get(d)
            if v is not None:
                totals[d].append(float(v))

    result: dict[str, Any] = {
        "n": len(records),
        "n_abstained": n_abstained,
        "n_parse_failures": n_parse_failures,
        "abstention_rate": round(n_abstained / len(records), 4) if records else 0.0,
    }
    for d in dims:
        vals = totals[d]
        result[f"mean_{d}"] = round(sum(vals) / len(vals), 4) if vals else None

    # Composite (mean of all 6 dims across all records with scores)
    all_vals = [v for d in dims for v in totals[d]]
    result["mean_composite"] = round(sum(all_vals) / len(all_vals), 4) if all_vals else None
    return result
