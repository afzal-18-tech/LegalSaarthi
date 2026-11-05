"""Phase 15 summarization unit tests (offline, no LLM calls)."""
from src.summarization import summary_metrics as M
from src.summarization.prompts import STRUCTURED_HEADINGS
from src.summarization.summarizers import (
    Summarizer, build_blocks, extractive_lead, select_lead, select_structured,
)


def _paras():
    rows = [("HEADER", "IN THE SUPREME COURT OF INDIA")]
    rows += [("FACTS", f"The appellant was employed in year {i}.") for i in range(5)]
    rows += [("ISSUES", "Whether the dismissal was lawful.")]
    rows += [("ARGUMENTS", "Counsel argued at length about procedure. " * 3)]
    rows += [("REASONING", f"The court found the inquiry defective {i}.") for i in range(5)]
    rows += [("DECISION", "The appeal is allowed.")]
    return [{"para_id": f"p1_{i}", "page": 1, "section": s, "text": t} for i, (s, t) in enumerate(rows)]


class FakeLLM:
    last_latency_s = 0.1
    last_model_used = "fake"

    def __init__(self, out):
        self.out, self.calls = out, []

    def generate(self, prompt, system_prompt=None):
        self.calls.append((prompt, system_prompt))
        return self.out


def test_blocks_skip_header_and_keep_provenance():
    blocks = build_blocks(_paras())
    assert all(b["section"] != "HEADER" for b in blocks)
    assert blocks[0]["block_id"] == "p1_1"
    assert all(b["para_ids"] and b["pages"] for b in blocks)


def test_blocks_respect_max_chars():
    blocks = build_blocks(_paras(), max_chars=50)
    assert all(len(b["text"]) <= 50 or len(b["para_ids"]) == 1 for b in blocks)


def test_select_lead_respects_budget():
    blocks = build_blocks(_paras())
    sel = select_lead(blocks, 100)
    assert sum(len(b["text"]) for b in sel) <= 100


def test_select_structured_includes_decision_and_budget():
    blocks = build_blocks(_paras(), max_chars=60)
    sel = select_structured(blocks, 200)
    assert sum(len(b["text"]) for b in sel) <= 200
    assert any(b["section"] == "DECISION" for b in sel)
    assert [b["order"] for b in sel] == sorted(b["order"] for b in sel)


def test_extractive_lead_cuts_at_sentence():
    blocks = build_blocks(_paras())
    out = extractive_lead(blocks, 80)
    assert len(out) <= 80 and out.endswith(".")


def test_summarizer_extractive_no_llm():
    s = Summarizer(llm=None)
    r = s.summarize("extractive_lead", build_blocks(_paras()))
    assert r["summary"] and r["model_used"] is None


def test_summarizer_structured_passes_block_ids():
    llm = FakeLLM("Facts: x [p1_1]")
    s = Summarizer(llm, input_budget=2000)
    s.summarize("structured_llm", build_blocks(_paras()))
    assert "[p1_1]" in llm.calls[0][0]


def test_gold_disposition_normalization():
    assert M.normalize_gold_disposition("Appeal(s) allowed") == "allowed"
    assert M.normalize_gold_disposition("Case Partly allowed") == "partly_allowed"
    assert M.normalize_gold_disposition("Disposed off") == "disposed"
    assert M.normalize_gold_disposition("Matter referred to larger bench") == "other"
    assert M.normalize_gold_disposition(None) == "other"


def test_predict_disposition():
    assert M.predict_disposition("The appeal is allowed.") == "allowed"
    assert M.predict_disposition("The appeal is partly allowed.") == "partly_allowed"
    assert M.predict_disposition("The petition was dismissed.") == "dismissed"
    assert M.predict_disposition("The appeal is not allowed and is dismissed.") == "dismissed"
    assert M.predict_disposition("The court discussed evidence.") == "not_stated"
    # last-mentioned outcome wins (High Court dismissed, SC allowed)
    assert M.predict_disposition("The High Court dismissed the writ. This appeal is allowed.") == "allowed"


def test_readability_simple_vs_complex():
    easy = M.readability("The cat sat. The dog ran.")
    hard = M.readability("Notwithstanding the aforementioned jurisprudential considerations, adjudication necessitates deliberation.")
    assert easy["flesch_reading_ease"] > hard["flesch_reading_ease"]
    assert M.readability("")["fk_grade"] is None


def test_headings_filled_and_placeholder():
    s = "Facts: A sued B [p1_1].\nIssues: Not stated in the provided text.\nReasoning: Defective inquiry [p1_9].\nDecision: Allowed [p1_14]."
    h = M.headings_filled(s, STRUCTURED_HEADINGS)
    assert h["filled"] == 3 and h["by_heading"]["Issues"] is False


def test_strip_citations_and_headings():
    s = "Facts: A sued B [p1_1].\nDecision: Allowed [p1_2; p1_3]."
    clean = M.remove_headings(M.strip_citations(s), STRUCTURED_HEADINGS)
    assert "[" not in clean and "Facts:" not in clean


def test_citation_check():
    c = M.citation_check("x [p1_1]. y [p9_9].", ["p1_1", "p1_2"])
    assert c["valid_ids"] == ["p1_1"] and c["invalid_ids"] == ["p9_9"]


def test_evidence_for_claims_picks_overlap_and_caps():
    blocks = build_blocks(_paras(), max_chars=60)
    ev = M.evidence_for_claims(["The court found the inquiry defective."], blocks, max_chars=500)
    assert "inquiry defective" in ev and len(ev) <= 500 + 10


def test_aggregate():
    row = {
        "generation_failed": False, "gold_disposition": "allowed", "pred_disposition": "allowed",
        "verification": {"claims": ["a"], "support_rate": 1.0, "supported_count": 1, "unsupported_count": 0, "unknown_count": 0},
        "readability": {"flesch_reading_ease": 50.0, "fk_grade": 10.0}, "n_words": 100, "compression": 0.1,
        "latency_s": 2.0, "citations": {"cited_ids": ["a", "b"], "invalid_ids": ["b"], "has_citation": True},
        "headings": {"filled": 3, "total": 4},
    }
    other = dict(row, gold_disposition="other", pred_disposition="dismissed", citations=None, headings=None)
    agg = M.aggregate_summaries([row, other])
    assert agg["disposition_scorable"] == 1 and agg["disposition_accuracy"] == 1.0
    assert agg["invalid_citation_rate"] == 0.5 and agg["headings_filled_rate"] == 0.75
