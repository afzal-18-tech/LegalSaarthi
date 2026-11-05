"""
LegalSaarthi — Basic RAG Pipeline Tests (Phase 13)
"""
import pytest

from src.generation.llm_client import LLMClient
from src.generation.rag_pipeline import BasicRAGPipeline, extract_citations
from src.evaluation.generation_metrics import LLMEvaluator


CHUNKS = {
    "CASE A_section_aware_0": {
        "chunk_id": "CASE A_section_aware_0", "case_id": "CASE A", "pages": [3],
        "paragraphs": ["p3_1"], "sections": ["DECISION"],
        "source_reference": "CASE A, paras p3_1", "text": "The appeal is allowed.",
    },
    "CASE A_section_aware_1": {
        "chunk_id": "CASE A_section_aware_1", "case_id": "CASE A", "pages": [2],
        "paragraphs": ["p2_4"], "sections": ["FACTS"],
        "source_reference": "CASE A, paras p2_4", "text": "The appellant was convicted.",
    },
}


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.last_query = None

    def retrieve(self, query, top_k=10):
        self.last_query = query
        return self.hits[:top_k]


class StubLLM(LLMClient):
    def __init__(self, reply):
        super().__init__(model_name="stub")
        self.reply = reply
        self.last_prompt = None

    def generate(self, prompt, system_prompt=None, retries=3):
        self.last_prompt = prompt
        self.last_model_used = "stub"
        self.last_latency_s = 0.01
        return self.reply


def test_extract_citations_dedup_and_multi():
    ans = "Allowed [X_1]. Convicted [X_2; X_1]. Again [X_2]."
    assert extract_citations(ans) == ["X_1", "X_2"]


def test_extract_citations_none():
    assert extract_citations("No citations here.") == []


def test_pipeline_provenance_and_valid_citation():
    retr = FakeRetriever([("CASE A_section_aware_0", 9.0), ("CASE A_section_aware_1", 4.0)])
    llm = StubLLM("The appeal was allowed [CASE A_section_aware_0].")
    out = BasicRAGPipeline(retr, CHUNKS, llm, top_k=2, retriever_name="fake").answer("What happened?")

    # Evidence and provenance are in the prompt and the output
    assert "[CASE A_section_aware_0] (case: CASE A; pages: 3; section: DECISION)" in llm.last_prompt
    assert [r["chunk_id"] for r in out["retrieved"]] == ["CASE A_section_aware_0", "CASE A_section_aware_1"]
    assert out["retrieved"][0]["paragraphs"] == ["p3_1"]
    assert out["retrieved"][0]["source_reference"] == "CASE A, paras p3_1"
    assert out["citations"]["valid_ids"] == ["CASE A_section_aware_0"]
    assert out["citations"]["invalid_ids"] == []
    assert out["model_used"] == "stub"
    assert out["prompt_version"] == "rag_basic_v1"


def test_pipeline_flags_fabricated_citation():
    retr = FakeRetriever([("CASE A_section_aware_0", 9.0)])
    llm = StubLLM("Allowed [CASE A_section_aware_0] per [CASE Z_section_aware_99].")
    out = BasicRAGPipeline(retr, CHUNKS, llm, top_k=1).answer("Q")
    assert out["citations"]["invalid_ids"] == ["CASE Z_section_aware_99"]


def test_pipeline_records_missing_chunks_not_silently_dropped():
    retr = FakeRetriever([("UNKNOWN_0", 5.0), ("CASE A_section_aware_1", 3.0)])
    out = BasicRAGPipeline(retr, CHUNKS, StubLLM("x"), top_k=2).answer("Q")
    assert out["missing_chunk_ids"] == ["UNKNOWN_0"]
    assert len(out["retrieved"]) == 1


def test_pipeline_uses_separate_retrieval_query():
    retr = FakeRetriever([("CASE A_section_aware_0", 1.0)])
    BasicRAGPipeline(retr, CHUNKS, StubLLM("x"), top_k=1).answer("Q display", retrieval_query="raw query")
    assert retr.last_query == "raw query"


def test_rag_abstention_phrase_detected():
    judge = StubLLM("CORRECTNESS: 5\nFACTUALITY: 5")
    res = LLMEvaluator(judge).evaluate("Q", "Gold", "Insufficient evidence to answer.")
    assert res["abstention"] == 1


def test_judge_parse_failure_is_flagged_not_scored():
    judge = StubLLM("")  # e.g. truncated judge output
    res = LLMEvaluator(judge).evaluate("Q", "Gold", "Some answer")
    assert res["judge_parse_failure"] is True
    assert res["correctness"] is None and res["factuality"] is None


def test_qwen3_detection_and_think_stripping():
    assert LLMClient._is_qwen3("ollama/qwen3:8b") is True
    assert LLMClient._is_qwen3("gemini/gemini-2.5-flash") is False
    assert LLMClient._clean_output("<think>\nhidden\n</think>\n\nAnswer") == "Answer"
    # An output consumed entirely by think tags must return None so retry/fallback activates
    assert LLMClient._clean_output("<think>\nonly hidden reasoning\n</think>") is None


def test_extract_citations_chunk_prefix_and_parentheses():
    ans = "The court observed [CHUNK 2015 INSC 177_section_aware_1] and (CHUNK 2015 INSC 177_section_aware_2)."
    assert extract_citations(ans) == ["2015 INSC 177_section_aware_1", "2015 INSC 177_section_aware_2"]


