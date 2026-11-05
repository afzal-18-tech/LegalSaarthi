"""
LegalSaarthi — Phase 15 Structured Summarization.

Four systems, all given the SAME input character budget:
  extractive_lead : no LLM; first N chars of non-header body text.
  generic_llm     : LLM summarizes the lead of the judgment (free-form).
  structured_llm  : structure-aware input selection (Facts/Issues/Reasoning/
                    Decision quotas) + sectioned output with [block_id] citations.
  plain_llm       : same input as structured_llm, plain-language prompt (H4).

Known confound: generic_llm vs structured_llm differ in BOTH input selection
and output format. Report this; an ablation can separate them later.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generation.llm_client import LLMClient
from src.summarization.prompts import (
    GENERIC_SUMM_SYSTEM, GENERIC_SUMM_USER,
    PLAIN_SUMM_SYSTEM, PLAIN_SUMM_USER,
    STRUCTURED_SUMM_SYSTEM, STRUCTURED_SUMM_USER,
    SUMM_BLOCK_TEMPLATE,
)

SKIP_LABELS = {"HEADER"}
BLOCK_MAX_CHARS = 700

# Share of the input budget per section for structure-aware selection.
SECTION_QUOTAS = {
    "FACTS": 0.25,
    "ISSUES": 0.10,
    "REASONING": 0.30,
    "DECISION": 0.20,
}
# Sections taken from the END of the judgment (the operative order is last).
TAIL_SECTIONS = {"DECISION"}


def load_document(path: Path) -> Dict[str, Any]:
    """Load a Phase 7 structured judgment into a flat paragraph list."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    paras = []
    for page in d.get("pages", []):
        for p in page.get("paragraphs", []):
            text = (p.get("text") or "").strip()
            if not text:
                continue
            paras.append({
                "para_id": p["para_id"],
                "page": page.get("page_num"),
                "section": p.get("section_label", "UNKNOWN"),
                "text": text,
            })
    return {"case_id": d["case_id"], "paragraphs": paras}


def build_blocks(paragraphs: List[Dict[str, Any]], max_chars: int = BLOCK_MAX_CHARS) -> List[Dict[str, Any]]:
    """Merge consecutive same-section paragraphs (excluding HEADER) into blocks.

    block_id is the first para_id; every block keeps its para_ids and pages
    so citations trace back to page/paragraph.
    """
    blocks: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for p in paragraphs:
        if p["section"] in SKIP_LABELS:
            cur = None
            continue
        if cur and cur["section"] == p["section"] and len(cur["text"]) + len(p["text"]) + 1 <= max_chars:
            cur["text"] += " " + p["text"]
            cur["para_ids"].append(p["para_id"])
            if p["page"] not in cur["pages"]:
                cur["pages"].append(p["page"])
        else:
            cur = {
                "block_id": p["para_id"],
                "section": p["section"],
                "text": p["text"],
                "para_ids": [p["para_id"]],
                "pages": [p["page"]],
            }
            blocks.append(cur)
    for i, b in enumerate(blocks):
        b["order"] = i
    return blocks


def select_lead(blocks: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    out, used = [], 0
    for b in blocks:
        if used + len(b["text"]) > budget:
            break
        out.append(b)
        used += len(b["text"])
    return out


def select_structured(blocks: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    """Fill per-section quotas, then spend leftover budget in document order."""
    chosen: Dict[int, Dict[str, Any]] = {}
    used = 0
    for section, share in SECTION_QUOTAS.items():
        quota = int(budget * share)
        cands = [b for b in blocks if b["section"] == section]
        if section in TAIL_SECTIONS:
            cands = list(reversed(cands))
        spent = 0
        for b in cands:
            if spent + len(b["text"]) > quota:
                break
            chosen[b["order"]] = b
            spent += len(b["text"])
        used += spent
    for b in blocks:
        if b["order"] in chosen:
            continue
        if used + len(b["text"]) > budget:
            continue
        chosen[b["order"]] = b
        used += len(b["text"])
    return [chosen[k] for k in sorted(chosen)]


def format_blocks(blocks: List[Dict[str, Any]], with_ids: bool) -> str:
    if not with_ids:
        return "\n\n".join(b["text"] for b in blocks)
    return "\n\n".join(
        SUMM_BLOCK_TEMPLATE.format(block_id=b["block_id"], section=b["section"],
                                   pages=",".join(str(p) for p in b["pages"]), text=b["text"])
        for b in blocks
    )


def extractive_lead(blocks: List[Dict[str, Any]], out_chars: int) -> str:
    """Classic lead baseline: first out_chars of body text, cut at a sentence end."""
    text = " ".join(b["text"] for b in blocks)
    if len(text) <= out_chars:
        return text
    cut = text[:out_chars]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: end + 1] if end > out_chars // 2 else cut


class Summarizer:
    SYSTEMS = ("extractive_lead", "generic_llm", "structured_llm", "plain_llm")
    # Phase 21 ablation: swap input selection and output format to separate their effects.
    #   structured_input_free_prompt : structure-aware input, free-form generic prompt (no headings/IDs)
    #   lead_input_structured_prompt : lead input, headed+cited structured prompt
    ABLATION_SYSTEMS = ("structured_input_free_prompt", "lead_input_structured_prompt")

    def __init__(self, llm: Optional[LLMClient], input_budget: int = 6000, extractive_chars: int = 1200):
        self.llm = llm
        self.input_budget = input_budget
        self.extractive_chars = extractive_chars

    def summarize(self, system: str, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if system == "extractive_lead":
            inp = select_lead(blocks, self.input_budget)
            return {"summary": extractive_lead(inp, self.extractive_chars), "input_blocks": inp,
                    "generation_failed": False, "latency_s": 0.0, "model_used": None}
        if system == "generic_llm":
            inp = select_lead(blocks, self.input_budget)
            sysp, user = GENERIC_SUMM_SYSTEM, GENERIC_SUMM_USER.format(text=format_blocks(inp, False))
        elif system == "structured_llm":
            inp = select_structured(blocks, self.input_budget)
            sysp, user = STRUCTURED_SUMM_SYSTEM, STRUCTURED_SUMM_USER.format(context=format_blocks(inp, True))
        elif system == "plain_llm":
            inp = select_structured(blocks, self.input_budget)
            sysp, user = PLAIN_SUMM_SYSTEM, PLAIN_SUMM_USER.format(context=format_blocks(inp, True))
        elif system == "structured_input_free_prompt":
            inp = select_structured(blocks, self.input_budget)
            sysp, user = GENERIC_SUMM_SYSTEM, GENERIC_SUMM_USER.format(text=format_blocks(inp, False))
        elif system == "lead_input_structured_prompt":
            inp = select_lead(blocks, self.input_budget)
            sysp, user = STRUCTURED_SUMM_SYSTEM, STRUCTURED_SUMM_USER.format(context=format_blocks(inp, True))
        else:
            raise ValueError(f"unknown system {system}")
        out = self.llm.generate(user, system_prompt=sysp)
        return {"summary": out, "input_blocks": inp, "generation_failed": out.startswith("ERROR:"),
                "latency_s": self.llm.last_latency_s, "model_used": self.llm.last_model_used}
