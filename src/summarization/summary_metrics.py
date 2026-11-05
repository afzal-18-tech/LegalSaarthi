"""
Phase 15 reference-free summary metrics.

No gold summaries exist for this corpus, so ROUGE/BERTScore are NOT computed.
Metrics:
  - disposition_correct : outcome stated in summary matches metadata disposal_nature
  - support_rate        : share of summary claims entailed by the source judgment
                          (LLM verifier over top-overlap source blocks)
  - citation validity   : cited block IDs exist in the system's input (LLM-structured only)
  - headings_filled     : required headings present with content (structured only)
  - readability         : Flesch Reading Ease, Flesch-Kincaid Grade
  - length / compression

Phase 1 Reliability Fixes:
  - predict_disposition now prefers DECISION-section text over last-match anywhere in document.
  - detect_document_type prevents contract/statute rules being applied to court judgments.
  - extract_parties returns structured fallback instead of arbitrary sentence fragments.
"""
import re
from typing import Any, Dict, List, Optional

from src.generation.rag_pipeline import extract_citations

# ── Disposition ────────────────────────────────────────────────────────────
GOLD_DISPOSITION_MAP = {
    "appeal(s) allowed": "allowed",
    "case allowed": "allowed",
    "allowed": "allowed",
    "case partly allowed": "partly_allowed",
    "partly allowed": "partly_allowed",
    "dismissed": "dismissed",
    "disposed off": "disposed",
    "disposed of": "disposed",
}
SCORABLE = {"allowed", "partly_allowed", "dismissed", "disposed"}

_DISP_PATTERNS = [
    ("partly_allowed", re.compile(r"\b(partly|partially|in part)\s+allowed\b|\ballowed\s+in\s+part\b", re.I)),
    ("dismissed", re.compile(r"\bdismiss(ed|al|es)?\b|\brejected\b", re.I)),
    ("disposed", re.compile(r"\bdisposed\s+(of|off)\b", re.I)),
    ("allowed", re.compile(r"(?<!not\s)(?<!partly\s)(?<!partially\s)\ballowed\b|\bset\s+aside\b|\bin favou?r of the appellant", re.I)),
]


def normalize_gold_disposition(raw: Optional[str]) -> str:
    return GOLD_DISPOSITION_MAP.get((raw or "").strip().lower(), "other")


def predict_disposition(summary: str, decision_text: str = "") -> str:
    """Return the operative outcome from the judgment.

    Phase 1 Fix: Preferentially scans ``decision_text`` (the text from
    DECISION-labelled sections if available), then falls back to scanning
    the full summary. This prevents intermediate mentions (e.g., "the High
    Court dismissed the appeal") from being misread as the final order.

    Args:
        summary: Full generated summary text.
        decision_text: Text extracted only from DECISION-section paragraphs.
            Pass non-empty to activate the section-aware search path.

    Returns:
        One of ``allowed``, ``partly_allowed``, ``dismissed``, ``disposed``,
        or ``not_stated``.
    """
    def _scan(text: str) -> tuple:
        best, best_pos = "not_stated", -1
        for label, pat in _DISP_PATTERNS:
            for m in pat.finditer(text or ""):
                if m.end() > best_pos or (m.end() == best_pos and label == "partly_allowed"):
                    best, best_pos = label, m.end()
        return best, best_pos

    # 1. Try explicit DECISION section first (most reliable operative text)
    if decision_text and decision_text.strip():
        label, _ = _scan(decision_text)
        if label != "not_stated":
            return label

    # 2. Try DECISION / ORDER section within summary if present
    dec_match = re.search(r"(?:DECISION|ORDER|DISPOSITION|OPERATIVE ORDER)\s*:\s*(.*)", summary or "", re.I | re.DOTALL)
    if dec_match:
        label, _ = _scan(dec_match.group(1))
        if label != "not_stated":
            return label

    # 3. For long summaries (>= 200 chars), scan tail starting at a word boundary
    # to avoid slicing mid-word (which breaks negative lookbehinds like 'partly allowed')
    if len(summary or "") >= 200:
        tail_start = (summary or "").find(" ", int(len(summary) * 0.6))
        if tail_start != -1:
            tail = summary[tail_start:]
            label, _ = _scan(tail)
            if label != "not_stated":
                return label

    # 4. Scan full summary
    label, _ = _scan(summary)
    return label


# ── Readability ────────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _syllables(word: str) -> int:
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def readability(text: str) -> Dict[str, Optional[float]]:
    sentences = [s for s in re.split(r"[.!?]+", text) if _WORD_RE.search(s)]
    words = _WORD_RE.findall(text)
    if not sentences or not words:
        return {"flesch_reading_ease": None, "fk_grade": None}
    syl = sum(_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syl / len(words)
    return {
        "flesch_reading_ease": 206.835 - 1.015 * wps - 84.6 * spw,
        "fk_grade": 0.39 * wps + 11.8 * spw - 15.59,
    }


# ── Text cleaning / structure ──────────────────────────────────────────────
def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\[\]]+\]", "", text or "")


def headings_filled(summary: str, headings: List[str]) -> Dict[str, Any]:
    """Which required headings are present with non-placeholder content."""
    present = {}
    for i, h in enumerate(headings):
        m = re.search(rf"(?im)^\W*{re.escape(h)}\W*:\s*(.*)$", summary or "")
        if not m:
            present[h] = False
            continue
        # content = text until next heading
        start = m.start(1)
        nxt = len(summary)
        for h2 in headings:
            if h2 == h:
                continue
            m2 = re.search(rf"(?im)^\W*{re.escape(h2)}\W*:", summary[start:])
            if m2:
                nxt = min(nxt, start + m2.start())
        body = summary[start:nxt].strip()
        present[h] = bool(body) and "not stated in the provided text" not in body.lower()
    return {"filled": sum(present.values()), "total": len(headings), "by_heading": present}


def remove_headings(summary: str, headings: List[str]) -> str:
    out = summary or ""
    for h in headings:
        out = re.sub(rf"(?im)^\W*{re.escape(h)}\W*:\s*", "", out)
    return out


# ── Evidence selection for claim verification ──────────────────────────────
_STOP = set("the a an of to and in on for by with is was were be been that this it as at from or are which has have had its their his her not no".split())


def _content_tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP and len(w) > 2}


def evidence_for_claims(claims: List[str], blocks: List[Dict[str, Any]], per_claim: int = 2,
                        max_chars: int = 3000) -> str:
    """Union of the top-overlap source blocks per claim, in document order, capped."""
    btoks = [(_content_tokens(b["text"]), b) for b in blocks]
    picked: Dict[int, Dict[str, Any]] = {}
    for c in claims:
        ct = _content_tokens(c)
        if not ct:
            continue
        scored = sorted(btoks, key=lambda x: len(ct & x[0]) / (len(ct) or 1), reverse=True)
        for _, b in scored[:per_claim]:
            picked[b["order"]] = b
    out, used = [], 0
    for k in sorted(picked):
        t = picked[k]["text"]
        if used + len(t) > max_chars:
            continue
        out.append(t)
        used += len(t)
    return "\n\n".join(out)


def citation_check(summary: str, input_block_ids: List[str]) -> Dict[str, Any]:
    cited = extract_citations(summary)
    valid = [c for c in cited if c in input_block_ids]
    return {
        "cited_ids": cited,
        "valid_ids": valid,
        "invalid_ids": [c for c in cited if c not in input_block_ids],
        "has_citation": bool(cited),
    }


# ── Aggregation ────────────────────────────────────────────────────────────
# ── Document-type detection (Phase 1) ────────────────────────────────────────
# Judgment keyword patterns (high-confidence identifiers in Indian judgments)
_JUDGMENT_SIGNALS = re.compile(
    r"\b(?:IN\s+THE\s+SUPREME\s+COURT|CIVIL\s+APPEAL|CRIMINAL\s+APPEAL|SPECIAL\s+LEAVE|"
    r"SLP\s*No\.|Writ\s+Petition|Transfer\s+Petition|High\s+Court|Sessions\s+Court|"
    r"petitioner|respondent|appellant|learned\s+(?:senior\s+)?counsel|the\s+appeal\s+is|judgment|decree|conviction)",
    re.IGNORECASE,
)
_CONTRACT_SIGNALS = re.compile(
    r"\b(?:AGREEMENT|WHEREAS|Party\s+of\s+the\s+(?:First|Second)\s+Part|Seller|Buyer|Licensor|"
    r"Licensee|Contractor|Vendor|terms\s+and\s+conditions|clause|INDEMNITY|TERMINATION|GOVERNING\s+LAW|"
    r"THIS\s+AGREEMENT\s+is\s+made|consideration\s+of|representations\s+and\s+warranties)",
    re.IGNORECASE,
)
_STATUTE_SIGNALS = re.compile(
    r"\b(?:BE\s+IT\s+ENACTED|CHAPTER\s+[IVX]+|passed\s+by\s+(?:Parliament|Legislature)|"
    r"extent\s+and\s+commencement|Short\s+title|PREAMBLE|Act\s+shall\s+apply|come\s+into\s+force)",
    re.IGNORECASE,
)

DOC_TYPE_JUDGMENT = "court_judgment"
DOC_TYPE_CONTRACT = "contract"
DOC_TYPE_STATUTE  = "statute"
DOC_TYPE_OTHER    = "other_legal_text"


def detect_document_type(text: str) -> str:
    """Classify a legal document's type from its full text.

    Phase 1 Fix: Prevents contract clause/risk analysis from being applied
    to court judgments, and ensures judgment-specific logic is not applied
    to contracts or statutes.

    Returns one of: ``court_judgment``, ``contract``, ``statute``,
    ``other_legal_text``.
    """
    if not text:
        return DOC_TYPE_OTHER
    j = len(_JUDGMENT_SIGNALS.findall(text))
    c = len(_CONTRACT_SIGNALS.findall(text))
    s = len(_STATUTE_SIGNALS.findall(text))
    total = j + c + s
    if total == 0:
        return DOC_TYPE_OTHER
    if j >= c and j >= s:
        return DOC_TYPE_JUDGMENT
    if c >= j and c >= s:
        return DOC_TYPE_CONTRACT
    return DOC_TYPE_STATUTE


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate_summaries(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scorable = [r for r in rows if r["gold_disposition"] in SCORABLE]
    out = {
        "count": len(rows),
        "generation_failures": sum(1 for r in rows if r["generation_failed"]),
        "disposition_scorable": len(scorable),
        "disposition_accuracy": _mean([1.0 if r["pred_disposition"] == r["gold_disposition"] else 0.0 for r in scorable]),
        "disposition_not_stated_rate": _mean([1.0 if r["pred_disposition"] == "not_stated" else 0.0 for r in scorable]),
        "mean_support_rate": _mean([r["verification"]["support_rate"] for r in rows if r["verification"]["claims"]]),
        "total_supported_claims": sum(r["verification"]["supported_count"] for r in rows),
        "total_unsupported_claims": sum(r["verification"]["unsupported_count"] for r in rows),
        "total_unknown_claims": sum(r["verification"]["unknown_count"] for r in rows),
        "flesch_reading_ease": _mean([r["readability"]["flesch_reading_ease"] for r in rows]),
        "fk_grade": _mean([r["readability"]["fk_grade"] for r in rows]),
        "mean_words": _mean([r["n_words"] for r in rows]),
        "mean_compression": _mean([r["compression"] for r in rows]),
        "latency_median_s": None,
    }
    lats = sorted(r["latency_s"] for r in rows if r.get("latency_s"))
    if lats:
        out["latency_median_s"] = lats[len(lats) // 2]
    cit_rows = [r for r in rows if r.get("citations") is not None]
    if cit_rows:
        n_cited = sum(len(r["citations"]["cited_ids"]) for r in cit_rows)
        n_invalid = sum(len(r["citations"]["invalid_ids"]) for r in cit_rows)
        out["citation_rate"] = _mean([1.0 if r["citations"]["has_citation"] else 0.0 for r in cit_rows])
        out["invalid_citation_rate"] = (n_invalid / n_cited) if n_cited else None
    head_rows = [r for r in rows if r.get("headings") is not None]
    if head_rows:
        out["headings_filled_rate"] = _mean([r["headings"]["filled"] / r["headings"]["total"] for r in head_rows])
    return out


def parse_structured_sections(
    summary_text: str,
    input_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Parse structured summary headings and map page-level evidence for each section."""
    block_page_map: Dict[str, List[int]] = {}
    if input_blocks:
        for b in input_blocks:
            bid = b.get("block_id")
            if bid:
                block_page_map[bid] = b.get("pages", [])

    # Heading patterns to match both standard 4-heading and enhanced 6-heading summaries
    heading_pats = [
        ("facts", re.compile(r"(?:^|\n)\s*(?:Background\s+Facts|Facts|What\s+happened)\s*:\s*", re.I)),
        ("issues", re.compile(r"(?:^|\n)\s*(?:Legal\s+Issues|Issues|What\s+the\s+court\s+had\s+to\s+decide)\s*:\s*", re.I)),
        ("arguments", re.compile(r"(?:^|\n)\s*(?:Arguments\s+and\s+Submissions|Arguments|Submissions)\s*:\s*", re.I)),
        ("reasoning", re.compile(r"(?:^|\n)\s*(?:Court\s+Reasoning|Reasoning|Why\s+the\s+court\s+decided\s+as\s+it\s+did)\s*:\s*", re.I)),
        ("decision", re.compile(r"(?:^|\n)\s*(?:Decision\s+and\s+Disposition|Decision|Final\s+outcome|Final\s+Order)\s*:\s*", re.I)),
        ("relief", re.compile(r"(?:^|\n)\s*(?:Relief\s+and\s+Directions|Relief|Directions)\s*:\s*", re.I)),
    ]

    found_spans = []
    for key, pat in heading_pats:
        m = pat.search(summary_text)
        if m:
            found_spans.append((m.start(), m.end(), key))

    found_spans.sort(key=lambda x: x[0])

    sections: Dict[str, Any] = {}
    missing_fields: List[str] = []

    for i, (start, header_end, key) in enumerate(found_spans):
        text_end = found_spans[i + 1][0] if i + 1 < len(found_spans) else len(summary_text)
        sec_text = summary_text[header_end:text_end].strip()

        cites = extract_citations(sec_text)
        pages_set = set()
        for cid in cites:
            for p in block_page_map.get(cid, []):
                pages_set.add(p)

        is_not_stated = (
            "not clearly stated" in sec_text.lower()
            or "not stated in the" in sec_text.lower()
            or not sec_text
        )

        if is_not_stated:
            missing_fields.append(key)

        sections[key] = {
            "text": sec_text,
            "citations": cites,
            "pages": sorted(list(pages_set)),
            "is_stated": not is_not_stated,
        }

    return {
        "sections": sections,
        "missing_fields": missing_fields,
    }
