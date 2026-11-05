"""
LegalSaarthi — Shared benchmark question construction and generation-metric aggregation.

Used by Phase 12 (generic LLM) and Phase 13 (basic RAG) so both systems are
evaluated on byte-identical questions with identical metric definitions.
"""
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

QUESTION_TEMPLATES = {
    "factual": "What are the facts of case {case_id} involving: '{text}'?",
    "dispositional": "What was the final decision in case {case_id} regarding: '{text}'?",
    "arguments": "What were the arguments in case {case_id} related to: '{text}'?",
    "legal_reasoning": "What was the legal reasoning in case {case_id} involving: '{text}'?",
}
DEFAULT_TEMPLATE = "Provide details about case {case_id} regarding: '{text}'?"


def build_question(item: Dict[str, Any]) -> str:
    """Natural-language question from a Phase 9 benchmark item (same wording as Phase 12)."""
    tmpl = QUESTION_TEMPLATES.get(item["question_type"], DEFAULT_TEMPLATE)
    return tmpl.format(case_id=item["case_id"], text=item["question_text"])


def load_chunk_store(path: Path) -> Dict[str, Dict[str, Any]]:
    store = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            store[c["chunk_id"]] = c
    return store


def load_split_map(path: Path) -> Dict[str, str]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {r["case_id"]: r.get("split", "unknown") for r in csv.DictReader(f)}


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _p95(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def aggregate(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-query results.

    Definitions (explicit to avoid conflating abstention with error):
      - abstention_rate: abstained / count
      - correctness_answered / factuality_answered: mean judge score (1-5) over
        answered queries whose judge output parsed. Abstentions excluded.
      - correctness_all: mean over all scorable queries with abstention = 0
        (the Phase 12 definition, kept for comparability).
      - judge_parse_failures / generation_failures: counted, never imputed.
    RAG-only (present when results carry retrieval/citation fields):
      - gold_hit_rate: >=1 gold chunk retrieved
      - gold_recall: mean fraction of gold chunks retrieved
      - citation_rate: answered queries with >=1 citation
      - invalid_citation_rate: cited IDs not in retrieved evidence / all cited IDs
      - cited_gold_rate: answered queries citing >=1 gold chunk
    """
    results = list(results)
    n = len(results)
    out: Dict[str, Any] = {"count": n}
    if n == 0:
        return out

    abst = [r for r in results if r["evaluation"]["abstention"] == 1]
    gen_fail = [r for r in results if r.get("generation_failed")]
    answered = [r for r in results if r["evaluation"]["abstention"] == 0 and not r.get("generation_failed")]
    parse_fail = [r for r in answered if r["evaluation"].get("judge_parse_failure")]
    scored = [r for r in answered if not r["evaluation"].get("judge_parse_failure")]

    out.update({
        "abstained": len(abst),
        "abstention_rate": len(abst) / n,
        "answered": len(answered),
        "generation_failures": len(gen_fail),
        "judge_parse_failures": len(parse_fail),
        "correctness_answered": _mean([r["evaluation"]["correctness"] for r in scored]),
        "factuality_answered": _mean([r["evaluation"]["factuality"] for r in scored]),
        "correctness_all": _mean([r["evaluation"]["correctness"] for r in scored + abst]),
        "factuality_all": _mean([r["evaluation"]["factuality"] for r in scored + abst]),
    })

    lat = [r["generation_latency_s"] for r in results if r.get("generation_latency_s") is not None]
    out["latency_median_s"] = statistics.median(lat) if lat else None
    out["latency_p95_s"] = _p95(lat)

    if results and "retrieval_eval" in results[0]:
        out["gold_hit_rate"] = _mean([1.0 if r["retrieval_eval"]["gold_hit"] else 0.0 for r in results])
        out["gold_recall"] = _mean([r["retrieval_eval"]["gold_recall"] for r in results])
        cited_total = sum(len(r["citations"]["cited_ids"]) for r in answered)
        invalid_total = sum(len(r["citations"]["invalid_ids"]) for r in answered)
        out["citation_rate"] = _mean([1.0 if r["citations"]["has_citation"] else 0.0 for r in answered])
        out["invalid_citation_rate"] = (invalid_total / cited_total) if cited_total else None
        out["cited_gold_rate"] = _mean([1.0 if r["retrieval_eval"]["cited_gold"] else 0.0 for r in answered])
        # Verified-RAG citation stripping
        stripped = [r["citations"].get("citations_stripped", 0) for r in answered]
        if any(s > 0 for s in stripped):
            out["total_citations_stripped"] = sum(stripped)

    # Verified-RAG claim verification metrics (Phase 14)
    if results and "verification" in results[0]:
        v_results = [r for r in answered if "verification" in r]
        if v_results:
            out["mean_support_rate"] = _mean([r["verification"]["support_rate"] for r in v_results])
            out["total_supported_claims"] = sum(r["verification"]["supported_count"] for r in v_results)
            out["total_unsupported_claims"] = sum(r["verification"]["unsupported_count"] for r in v_results)
            out["total_unknown_claims"] = sum(r["verification"]["unknown_count"] for r in v_results)
            out["verification_abstentions"] = sum(
                1 for r in v_results if r["verification"].get("abstained")
            )

    return out


def aggregate_by(results: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(str(r.get(key, "unknown")), []).append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def fmt(x: Any, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    if pct:
        return f"{x:.1%}"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)
