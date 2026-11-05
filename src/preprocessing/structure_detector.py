"""
LegalSaarthi — Legal Structure Detector (Phase 7)

Rule-based section classifier for Indian Supreme Court judgments.
Assigns each paragraph a section label from a controlled vocabulary using
regex pattern matching, positional heuristics, and contextual continuity.

Section labels:
    HEADER, FACTS, ISSUES, ARGUMENTS, LAW, PRECEDENTS,
    REASONING, DECISION, UNKNOWN

This is a heuristic baseline. It establishes a stable schema for
downstream phases (chunking, retrieval, summarization) and provides
a measurable starting point for future ML-based improvements.
"""
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_setup import get_logger

logger = get_logger("preprocessing.structure")


class SectionLabel(str, Enum):
    """Controlled vocabulary for legal document section types."""
    HEADER = "HEADER"
    FACTS = "FACTS"
    ISSUES = "ISSUES"
    ARGUMENTS = "ARGUMENTS"
    LAW = "LAW"
    PRECEDENTS = "PRECEDENTS"
    REASONING = "REASONING"
    DECISION = "DECISION"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Each entry: (SectionLabel, compiled regex, weight)
# Weight is used when multiple patterns match; highest weight wins.
# Patterns are case-insensitive.

_PATTERN_DEFS: List[Tuple[SectionLabel, str, float]] = [
    # --- HEADER patterns (case metadata, bench, title block) ---
    (SectionLabel.HEADER, r"(?:^|\b)(?:IN\s+THE\s+SUPREME\s+COURT)", 10.0),
    (SectionLabel.HEADER, r"(?:^|\b)(?:CIVIL\s+APPEAL|CRIMINAL\s+APPEAL|SPECIAL\s+LEAVE\s+PETITION|SLP|WRIT\s+PETITION|TRANSFER\s+PETITION)", 8.0),
    (SectionLabel.HEADER, r"\bCORAM\b", 9.0),
    (SectionLabel.HEADER, r"\bHON['\u2019]?BLE\b", 7.0),
    (SectionLabel.HEADER, r"\bJ\.?\s*J\.?\s*\]", 6.0),
    (SectionLabel.HEADER, r"(?:^|\b)(?:VERSUS|v\.)\s", 5.0),
    (SectionLabel.HEADER, r"(?:^|\b)(?:Petitioner|Respondent|Appellant)\s*$", 5.0),
    (SectionLabel.HEADER, r"^\s*JUDGMENT\s*$", 8.0),
    (SectionLabel.HEADER, r"^\s*O\s*R\s*D\s*E\s*R\s*$", 8.0),
    (SectionLabel.HEADER, r"^\[?\d{4}\]?\s*\d+\s*S\.?\s*C\.?\s*R\.?\s*\d+", 7.0),

    # --- FACTS patterns ---
    (SectionLabel.FACTS, r"(?:facts?\s+of\s+the\s+case|brief\s+facts|factual\s+(?:background|matrix|scenario))", 9.0),
    (SectionLabel.FACTS, r"(?:background\s+of\s+(?:the\s+)?case|genesis\s+of\s+(?:the\s+)?(?:case|matter|dispute))", 8.0),
    (SectionLabel.FACTS, r"(?:the\s+(?:brief|material|relevant|necessary)\s+facts)", 8.0),
    (SectionLabel.FACTS, r"(?:facts\s+(?:in\s+brief|briefly\s+stated|necessary\s+for|giving\s+rise))", 7.0),

    # --- ISSUES patterns ---
    (SectionLabel.ISSUES, r"(?:issues?\s+(?:for|that|to\s+be)\s+(?:determination|consideration|decided|adjudication))", 9.0),
    (SectionLabel.ISSUES, r"(?:(?:the\s+)?(?:point|question)s?\s+(?:for|that|which)\s+(?:determination|consideration|arise|fall))", 8.0),
    (SectionLabel.ISSUES, r"(?:(?:the\s+)?(?:moot|central|core|pivotal)\s+(?:question|issue|point))", 7.0),
    (SectionLabel.ISSUES, r"(?:framing\s+of\s+issues|issues?\s+framed)", 9.0),

    # --- ARGUMENTS patterns ---
    (SectionLabel.ARGUMENTS, r"(?:learned\s+(?:senior\s+)?counsel)", 7.0),
    (SectionLabel.ARGUMENTS, r"(?:(?:submitted|contended|argued|urged)\s+(?:that|by|on\s+behalf))", 6.0),
    (SectionLabel.ARGUMENTS, r"(?:submission(?:s)?\s+(?:of|on\s+behalf|made\s+by))", 7.0),
    (SectionLabel.ARGUMENTS, r"(?:on\s+behalf\s+of\s+the\s+(?:appellant|respondent|petitioner|plaintiff|defendant|accused))", 6.0),
    (SectionLabel.ARGUMENTS, r"(?:(?:Mr|Ms|Shri|Smt)\.?\s+[A-Z][a-z]+.*?(?:submitted|argued|contended))", 5.0),
    (SectionLabel.ARGUMENTS, r"(?:rival\s+(?:submissions|contentions|arguments))", 8.0),

    # --- LAW patterns (statutes, provisions) ---
    (SectionLabel.LAW, r"(?:(?:Section|Sec\.)\s+\d+[A-Za-z]?(?:\s*(?:\(\d+\)|\([a-z]\)))*)", 5.0),
    (SectionLabel.LAW, r"(?:Article\s+\d+[A-Za-z]?(?:\s*(?:\(\d+\)|\([a-z]\)))*)", 5.0),
    (SectionLabel.LAW, r"(?:Rule\s+\d+)", 4.0),
    (SectionLabel.LAW, r"(?:Order\s+(?:[IVXLC]+|\d+)\s+Rule\s+\d+)", 6.0),
    (SectionLabel.LAW, r"(?:(?:the\s+)?[A-Z][a-zA-Z\s]+Act,?\s*(?:19|20)\d{2})", 5.0),
    (SectionLabel.LAW, r"(?:Constitution\s+of\s+India)", 6.0),
    (SectionLabel.LAW, r"\b(?:IPC|CrPC|Cr\.P\.C|CPC|C\.P\.C)\b", 5.0),
    (SectionLabel.LAW, r"(?:Indian\s+(?:Penal|Evidence|Contract)\s+(?:Code|Act))", 5.0),

    # --- PRECEDENTS patterns (case citations) ---
    (SectionLabel.PRECEDENTS, r"\(\d{4}\)\s*\d+\s*S\.?\s*C\.?\s*C\.?", 6.0),
    (SectionLabel.PRECEDENTS, r"\(\d{4}\)\s*\d+\s*S\.?\s*C\.?\s*R\.?", 6.0),
    (SectionLabel.PRECEDENTS, r"\bAIR\s+\d{4}\s+S\.?\s*C\.?\b", 6.0),
    (SectionLabel.PRECEDENTS, r"(?:reported\s+in\b)", 5.0),
    (SectionLabel.PRECEDENTS, r"(?:relied\s+(?:upon|on)\s+.*?(?:decision|judgment|case|ruling))", 5.0),
    (SectionLabel.PRECEDENTS, r"(?:(?:this\s+Court|the\s+Court)\s+(?:in|has\s+in)\s+[A-Z])", 4.0),
    (SectionLabel.PRECEDENTS, r"(?:(?:the\s+)?(?:ratio|principle)\s+(?:laid\s+down|enunciated|propounded)\s+in)", 6.0),

    # --- REASONING patterns ---
    (SectionLabel.REASONING, r"(?:in\s+(?:our|my)\s+(?:opinion|view|considered\s+(?:opinion|view)))", 9.0),
    (SectionLabel.REASONING, r"(?:(?:we|I)\s+(?:find|hold|are\s+of|am\s+of|observe|notice|note)\s+that)", 8.0),
    (SectionLabel.REASONING, r"(?:having\s+(?:considered|heard|perused|examined))", 7.0),
    (SectionLabel.REASONING, r"(?:(?:on|upon)\s+(?:a\s+)?(?:careful|close|due)\s+(?:perusal|examination|consideration|scrutiny|analysis))", 8.0),
    (SectionLabel.REASONING, r"(?:(?:the\s+)?(?:above|aforesaid|foregoing)\s+(?:analysis|discussion|consideration|reasoning))", 7.0),
    (SectionLabel.REASONING, r"(?:(?:we|I)\s+(?:are|am)\s+(?:of\s+the\s+)?(?:opinion|view|firm\s+view))", 8.0),
    (SectionLabel.REASONING, r"(?:(?:it\s+is|we\s+are)\s+(?:well\s+)?(?:settled|established)\s+(?:that|in\s+law))", 6.0),
    (SectionLabel.REASONING, r"(?:(?:for\s+the\s+)?(?:reasons?\s+(?:aforesaid|aforementioned|stated|set\s+out)))", 7.0),

    # --- DECISION patterns ---
    (SectionLabel.DECISION, r"(?:(?:the\s+)?appeal(?:s)?\s+(?:is|are)\s+(?:accordingly\s+)?(?:allowed|dismissed|partly\s+allowed|disposed\s+of))", 10.0),
    (SectionLabel.DECISION, r"(?:(?:the\s+)?(?:writ\s+)?petition(?:s)?\s+(?:is|are)\s+(?:accordingly\s+)?(?:allowed|dismissed|disposed\s+of))", 10.0),
    (SectionLabel.DECISION, r"(?:(?:we|I)\s+(?:accordingly\s+)?(?:direct|order|allow|dismiss|dispose|set\s+aside))", 9.0),
    (SectionLabel.DECISION, r"(?:(?:in\s+the\s+result|accordingly|in\s+view\s+of\s+the\s+above),?\s+(?:the|this|we))", 8.0),
    (SectionLabel.DECISION, r"(?:ordered\s+accordingly)", 9.0),
    (SectionLabel.DECISION, r"(?:no\s+(?:order\s+as\s+to\s+)?costs)", 6.0),
    (SectionLabel.DECISION, r"(?:(?:the\s+)?(?:impugned\s+)?(?:order|judgment|decree)\s+(?:is|are|shall\s+be)\s+(?:set\s+aside|quashed|upheld|affirmed|modified))", 9.0),
]


class StructureDetector:
    """
    Rule-based legal document structure detector.

    Uses regex patterns, positional heuristics, and contextual continuity
    to assign a section label to each paragraph of a parsed judgment.
    """

    STRUCTURE_VERSION = "v1.0-rules"

    def __init__(
        self,
        header_page_threshold: int = 2,
        decision_tail_ratio: float = 0.15,
        continuity_weight: float = 3.0,
    ):
        """
        Args:
            header_page_threshold: Pages 1..N bias toward HEADER.
            decision_tail_ratio: Last X% of paragraphs bias toward DECISION.
            continuity_weight: Bonus score for maintaining the previous label.
        """
        self.header_page_threshold = header_page_threshold
        self.decision_tail_ratio = decision_tail_ratio
        self.continuity_weight = continuity_weight
        self._patterns = self._compile_patterns()

    # ------------------------------------------------------------------
    # Pattern compilation
    # ------------------------------------------------------------------
    @staticmethod
    def _compile_patterns() -> List[Tuple[SectionLabel, "re.Pattern[str]", float]]:
        """Pre-compile all regex patterns."""
        compiled = []
        for label, pattern_str, weight in _PATTERN_DEFS:
            compiled.append((label, re.compile(pattern_str, re.IGNORECASE), weight))
        return compiled

    # ------------------------------------------------------------------
    # Single paragraph classification
    # ------------------------------------------------------------------
    def classify_paragraph(
        self,
        text: str,
        position_ratio: float,
        page_num: int,
        prev_label: Optional[SectionLabel] = None,
    ) -> SectionLabel:
        """
        Classify a single paragraph into a section label.

        Args:
            text: The paragraph text.
            position_ratio: Position of this paragraph in the document
                            (0.0 = start, 1.0 = end).
            page_num: 1-indexed page number.
            prev_label: The label assigned to the previous paragraph.

        Returns:
            The best matching SectionLabel.
        """
        if not text or not text.strip():
            return SectionLabel.UNKNOWN

        # Score accumulator: label -> total score
        scores: Dict[SectionLabel, float] = {label: 0.0 for label in SectionLabel}

        # ---- 1. Pattern matching ----
        for label, pattern, weight in self._patterns:
            if pattern.search(text):
                scores[label] += weight

        # ---- 2. Positional heuristics ----
        # Early pages bias toward HEADER
        if page_num <= self.header_page_threshold:
            scores[SectionLabel.HEADER] += 3.0

        # Very first paragraphs on page 1 are almost certainly header
        if page_num == 1 and position_ratio < 0.02:
            scores[SectionLabel.HEADER] += 5.0

        # Tail of document biases toward DECISION
        if position_ratio > (1.0 - self.decision_tail_ratio):
            scores[SectionLabel.DECISION] += 2.0

        # Early-middle portion biases slightly toward FACTS
        if 0.03 < position_ratio < 0.25:
            scores[SectionLabel.FACTS] += 1.0

        # ---- 3. Contextual continuity ----
        # If the previous paragraph had a label and this paragraph has no
        # strong pattern match, favour continuing the same label.
        if prev_label is not None and prev_label != SectionLabel.UNKNOWN:
            scores[prev_label] += self.continuity_weight

        # ---- 4. Select best label ----
        # If no substantive label scored > 0, default to UNKNOWN
        substantive_max = max(
            score for label, score in scores.items()
            if label != SectionLabel.UNKNOWN
        )
        if substantive_max <= 0:
            return SectionLabel.UNKNOWN

        # Suppress UNKNOWN when any substantive label has signal
        scores[SectionLabel.UNKNOWN] = -1.0

        best_label = max(scores, key=lambda lbl: scores[lbl])
        return best_label

    # ------------------------------------------------------------------
    # Full document structure detection
    # ------------------------------------------------------------------
    def detect_structure(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a parsed document with section labels on every paragraph.

        Args:
            parsed_doc: Output of PDFParser.parse_pdf(), with keys
                        'case_id', 'total_pages', 'pages', etc.

        Returns:
            A new dict (deep copy semantics on paragraphs) with:
            - 'section_label' added to each paragraph
            - 'sections_summary' top-level field with label counts
            - 'structure_version' field
        """
        case_id = parsed_doc.get("case_id", "unknown")

        # Flatten paragraphs for position calculation
        all_paras: List[Tuple[int, int, Dict[str, Any]]] = []
        for page in parsed_doc.get("pages", []):
            page_num = page.get("page_num", 0)
            for pi, para in enumerate(page.get("paragraphs", [])):
                all_paras.append((page_num, pi, para))

        total_paras = len(all_paras)
        if total_paras == 0:
            logger.warning(f"Case {case_id}: No paragraphs found for structure detection.")
            return {
                **parsed_doc,
                "structure_version": self.STRUCTURE_VERSION,
                "sections_summary": {lbl.value: 0 for lbl in SectionLabel},
            }

        # Classify each paragraph
        prev_label: Optional[SectionLabel] = None
        label_counts: Dict[str, int] = {lbl.value: 0 for lbl in SectionLabel}
        # Build a lookup: (page_num, para_index) -> label
        label_map: Dict[Tuple[int, int], str] = {}

        for idx, (page_num, para_idx, para) in enumerate(all_paras):
            position_ratio = idx / total_paras
            label = self.classify_paragraph(
                text=para.get("text", ""),
                position_ratio=position_ratio,
                page_num=page_num,
                prev_label=prev_label,
            )
            label_map[(page_num, para_idx)] = label.value
            label_counts[label.value] += 1
            prev_label = label

        # Build enriched output
        enriched_pages = []
        for page in parsed_doc.get("pages", []):
            page_num = page.get("page_num", 0)
            enriched_paras = []
            for pi, para in enumerate(page.get("paragraphs", [])):
                enriched_paras.append({
                    **para,
                    "section_label": label_map.get((page_num, pi), SectionLabel.UNKNOWN.value),
                })
            enriched_pages.append({
                "page_num": page_num,
                "paragraphs": enriched_paras,
            })

        distinct_labels = sum(1 for v in label_counts.values() if v > 0)
        logger.info(
            f"Case {case_id}: {total_paras} paragraphs, "
            f"{distinct_labels} distinct labels detected."
        )

        return {
            "case_id": case_id,
            "total_pages": parsed_doc.get("total_pages", 0),
            "structure_version": self.STRUCTURE_VERSION,
            "sections_summary": label_counts,
            "pages": enriched_pages,
            "parse_status": parsed_doc.get("parse_status", "success"),
            "error_msg": parsed_doc.get("error_msg", ""),
        }

    # ------------------------------------------------------------------
    # Phase 1: Party extraction from HEADER paragraphs
    # ------------------------------------------------------------------
    @staticmethod
    def extract_parties(
        enriched_doc: Dict[str, Any],
        fallback: str = "Not clearly stated in the document",
    ) -> Dict[str, str]:
        """Extract petitioner/appellant and respondent names from HEADER paragraphs.

        Phase 1 Fix: Uses strict positional patterns on HEADER-labelled text
        only. If no match is found in any HEADER paragraph, returns the
        ``fallback`` string for both fields rather than extracting an arbitrary
        sentence fragment.

        Args:
            enriched_doc: Output of :meth:`detect_structure` (has section_label
                on every paragraph).
            fallback: String returned when extraction fails.

        Returns:
            dict with keys ``petitioner`` and ``respondent``.
        """
        # Collect all HEADER paragraphs from the first 3 pages (title block)
        header_texts: List[str] = []
        for page in enriched_doc.get("pages", []):
            if page.get("page_num", 99) > 3:
                break
            for para in page.get("paragraphs", []):
                if para.get("section_label") == SectionLabel.HEADER.value:
                    t = (para.get("text") or "").strip()
                    if t:
                        header_texts.append(t)

        full_header = "\n".join(header_texts)

        # Pattern 1: "X VERSUS Y" or "X v. Y" with proper noun heuristic
        versus_pat = re.compile(
            r"(?:^|\n)\s*([A-Z][^\n\r]{3,80}?)\s*\n?\s*(?:VERSUS|Vs\.?|v\.)\s*\n?\s*([A-Z][^\n\r]{3,80})",
            re.IGNORECASE | re.MULTILINE,
        )
        m = versus_pat.search(full_header)
        if m:
            petitioner = m.group(1).strip().rstrip(".,;")
            respondent = m.group(2).strip().rstrip(".,;")
            # Reject if fragment is suspiciously short or starts with a lowercase word
            if (
                len(petitioner) >= 4
                and len(respondent) >= 4
                and petitioner[0].isupper()
                and respondent[0].isupper()
            ):
                return {"petitioner": petitioner, "respondent": respondent}

        # Pattern 2: Lines ending with "...Petitioner" / "...Appellant" / "...Respondent"
        party_lines = re.compile(
            r"^(.{4,80}?)\s*\n?\s*(?:\.\.\.|—|--)?\s*"
            r"(Petitioner(?:s)?|Appellant(?:s)?|Respondent(?:s)?|Complainant(?:s)?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        petitioner_val = fallback
        respondent_val = fallback
        for m2 in party_lines.finditer(full_header):
            name = m2.group(1).strip().rstrip(".,;")
            role = m2.group(2).strip().lower()
            if len(name) < 4 or not name[0].isupper():
                continue
            if "petitioner" in role or "appellant" in role or "complainant" in role:
                petitioner_val = name
            elif "respondent" in role:
                respondent_val = name

        return {"petitioner": petitioner_val, "respondent": respondent_val}


# ---------------------------------------------------------------------------
# Module-level helpers (Phase 1)
# ---------------------------------------------------------------------------

def extract_decision_text(enriched_doc: Dict[str, Any]) -> str:
    """Collect all DECISION-section paragraph text from a structured document.

    Used by ``predict_disposition`` to obtain section-aware operative text,
    reducing false positives from intermediate discussion of prior court orders.

    Args:
        enriched_doc: Output of :meth:`StructureDetector.detect_structure`.

    Returns:
        Concatenated text of all DECISION-labelled paragraphs, or empty string
        if none are found.
    """
    decision_parts: List[str] = []
    for page in enriched_doc.get("pages", []):
        for para in page.get("paragraphs", []):
            if para.get("section_label") == SectionLabel.DECISION.value:
                t = (para.get("text") or "").strip()
                if t:
                    decision_parts.append(t)
    return " ".join(decision_parts)


def extract_case_number(header_text: str) -> str:
    """Extract case number (Appeal / Petition) from header paragraphs."""
    pats = [
        r"(?:Civil|Criminal)\s+Appeal\s+No[s\.]*.*?of\s+\d{4}",
        r"(?:Special\s+Leave\s+Petition|SLP)\s*\(.*?\)\s*No[s\.]*.*?of\s+\d{4}",
        r"Writ\s+Petition\s*\(.*?\)\s*No[s\.]*.*?of\s+\d{4}",
        r"Transfer\s+Petition\s*\(.*?\)\s*No[s\.]*.*?of\s+\d{4}",
        r"Review\s+Petition\s*\(.*?\)\s*No[s\.]*.*?of\s+\d{4}",
    ]
    for p in pats:
        m = re.search(p, header_text, re.IGNORECASE)
        if m:
            return m.group(0).strip().rstrip(".,;")
    return "Not clearly stated in the document"


def extract_court_name(header_text: str) -> str:
    """Extract court name from header paragraphs."""
    if re.search(r"\b(?:SUPREME\s+COURT|S\.?C\.?R\.?|INSC)\b", header_text, re.IGNORECASE):
        return "Supreme Court of India"
    hc_m = re.search(
        r"\bHIGH\s+COURT\s+(?:OF\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?|AT\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b",
        header_text,
    )
    if hc_m:
        return hc_m.group(0).strip().title()
    return "Supreme Court of India"


def extract_judgment_date(header_text: str, full_doc: Optional[Dict[str, Any]] = None) -> str:
    """Extract judgment date from header or last page."""
    date_pats = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)),?\s+\d{4}\b",
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
    ]
    for p in date_pats:
        m = re.search(p, header_text, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    if full_doc and full_doc.get("pages"):
        last_page = full_doc["pages"][-1]
        last_text = " ".join(p.get("text", "") for p in last_page.get("paragraphs", []))
        for p in date_pats:
            m = re.search(p, last_text, re.IGNORECASE)
            if m:
                return m.group(0).strip()

    return "Not clearly stated in the document"


def extract_bench(header_text: str) -> List[str]:
    """Extract judge names from header Coram / Bench text."""
    m = re.search(r"\[\s*([A-Z\.\s,]+?)\s*,\s*JJ?\.?\s*\]", header_text)
    if m:
        raw = m.group(1).replace(" AND ", ", ")
        judges = [j.strip() for j in raw.split(",") if len(j.strip()) > 2]
        if judges:
            return judges

    hon_matches = re.findall(
        r"HON['\u2019]?BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+([A-Z\.\s]+?)(?=\s+(?:AND|,\s*HON|\.|\n|$))",
        header_text,
        re.IGNORECASE,
    )
    if hon_matches:
        return [f"Justice {j.strip().title()}" for j in hon_matches if len(j.strip()) > 2]

    return []


def extract_precedents_list(enriched_doc: Dict[str, Any], max_items: int = 15) -> List[str]:
    """Extract cited case precedents from PRECEDENTS-labelled sections with strict normalization."""
    precedents = []
    seen = set()

    # 1. Standard Indian law reports citations (High confidence)
    law_report_pat = re.compile(
        r"(?:\bAIR\s+\d{4}\s+SC\s+\d+|\(?\d{4}\)?\s*\d+\s*S\.?C\.?C\.?\s*(?:\([A-Za-z0-9\s]+\)\s*)?\d+|"
        r"\[\d{4}\]\s*\d+\s*S\.?C\.?R\.?\s*\d+|\b\d{4}\s+INSC\s+\d+|\b\d{4}\s+SCC\s+OnLine\s+SC\s+\d+)",
        re.IGNORECASE,
    )

    # 2. Party vs Party format (Must be capitalized title tokens, not sentence fragments)
    party_pat = re.compile(
        r"\b((?:[A-Z][A-Za-z0-9\.\'\-]+(?:\s+(?:[A-Z][A-Za-z0-9\.\'\-]+|of|and|&|the|for)){0,5})\s+(?:v\.|vs\.|versus)\s+(?:[A-Z][A-Za-z0-9\.\'\-]+(?:\s+(?:[A-Z][A-Za-z0-9\.\'\-]+|of|and|&|the|for)){0,5}))"
    )

    # Forbidden words/phrases that indicate a sentence fragment, not a genuine case title
    invalid_fragments = {
        "court held", "we held", "it was held", "learned counsel", "in the case", "appeal is", "appeal was",
        "held that", "observed that", "stated that", "noted that", "found that", "judgment of", "order of",
        "furthermore", "however", "therefore", "in our opinion", "we are of the view", "ratio decidendi"
    }

    for page in enriched_doc.get("pages", []):
        for para in page.get("paragraphs", []):
            if para.get("section_label") == SectionLabel.PRECEDENTS.value:
                text = (para.get("text") or "").strip()
                if not text:
                    continue

                # Check law reports first
                for match in law_report_pat.finditer(text):
                    cite = match.group(0).strip().rstrip(".,;")
                    if len(cite) >= 5 and cite.lower() not in seen:
                        seen.add(cite.lower())
                        precedents.append(cite)
                        if len(precedents) >= max_items:
                            return precedents

                # Check party citations
                for match in party_pat.finditer(text):
                    cite = match.group(1).strip().rstrip(".,;")
                    # Strip leading prepositions (e.g., 'In Kesavananda' -> 'Kesavananda')
                    cite = re.sub(r"^(?:in|see|vide|cf|per|re)\s+", "", cite, flags=re.IGNORECASE).strip()
                    cite_low = cite.lower()
                    # Validate: not too long, not an invalid sentence fragment
                    if 8 <= len(cite) <= 75 and not any(frag in cite_low for frag in invalid_fragments):
                        if cite_low not in seen:
                            seen.add(cite_low)
                            precedents.append(cite)
                            if len(precedents) >= max_items:
                                return precedents

    return precedents


def extract_statutory_provisions(enriched_doc: Dict[str, Any], max_items: int = 15) -> List[str]:
    """Extract statutory provisions from LAW-labelled sections."""
    statutes = []
    seen = set()
    pat = re.compile(
        r"(?:(?:Section|Sec\.)\s+\d+[A-Za-z]?(?:\s*(?:\(\d+\)|\([a-z]\)))*(?:\s+of\s+the\s+[A-Za-z\s]+Act)?)|"
        r"(?:Article\s+\d+[A-Za-z]?(?:\s*(?:\(\d+\)|\([a-z]\)))*)|"
        r"(?:Order\s+(?:[IVXLC]+|\d+)\s+Rule\s+\d+)|"
        r"(?:[A-Z][a-zA-Z\s]{2,30}Act(?:,\s*\d{4})?)",
        re.IGNORECASE,
    )
    for page in enriched_doc.get("pages", []):
        for para in page.get("paragraphs", []):
            if para.get("section_label") in (SectionLabel.LAW.value, SectionLabel.REASONING.value):
                text = (para.get("text") or "").strip()
                for match in pat.finditer(text):
                    stat = match.group(0).strip().rstrip(".,;")
                    if len(stat) > 4 and stat.lower() not in seen:
                        seen.add(stat.lower())
                        statutes.append(stat)
                        if len(statutes) >= max_items:
                            return statutes
    return statutes


def extract_full_case_metadata(enriched_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Collect complete court, case number, date, bench, and parties metadata."""
    header_texts: List[str] = []
    for page in enriched_doc.get("pages", []):
        if page.get("page_num", 99) > 3:
            break
        for para in page.get("paragraphs", []):
            if para.get("section_label") == SectionLabel.HEADER.value:
                t = (para.get("text") or "").strip()
                if t:
                    header_texts.append(t)
    full_header = " ".join(header_texts)

    parties = StructureDetector.extract_parties(enriched_doc)
    case_no = extract_case_number(full_header)
    court = extract_court_name(full_header)
    date_str = extract_judgment_date(full_header, full_doc=enriched_doc)
    bench = extract_bench(full_header)

    petitioner = parties.get("petitioner", "Not clearly stated in the document")
    respondent = parties.get("respondent", "Not clearly stated in the document")
    if petitioner != "Not clearly stated in the document" and respondent != "Not clearly stated in the document":
        case_title = f"{petitioner} v. {respondent}"
    else:
        case_title = enriched_doc.get("case_id", "Indian Court Judgment")

    return {
        "case_title": case_title,
        "court": court,
        "case_number": case_no,
        "judgment_date": date_str,
        "bench": bench,
        "petitioner": petitioner,
        "respondent": respondent,
    }
