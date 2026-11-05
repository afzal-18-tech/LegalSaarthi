"""
LegalSaarthi — Verified-RAG Pipeline (Phase 14)

Extends the Basic RAG pipeline (Phase 13) with three verification layers:

1. **Hybrid Retrieval** — BM25+Dense RRF fusion (Phase 11 HybridRetriever)
   to leverage both lexical precision and semantic coverage.

2. **Citation Verification** — Post-generation check that every [chunk_id]
   citation in the answer corresponds to a chunk that was actually retrieved.
   Hallucinated citations are stripped and flagged.

3. **Claim-Level Verification** — Each factual sentence is checked for
   entailment against the cited evidence chunks. Unsupported claims are
   flagged. If the proportion of unsupported claims exceeds a threshold,
   the system abstains rather than outputting misleading content.

Research Hypotheses Tested:
  H2: Structure-aware retrieval improves coverage (hybrid + section metadata).
  H3: Explicit citation/claim verification reduces unsupported claims and
      fabricated legal references.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.generation.llm_client import LLMClient
from src.generation.prompts import (
    RAG_EVIDENCE_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    VERIFIED_RAG_PROMPT_VERSION,
    VERIFIED_RAG_SYSTEM_PROMPT,
    VERIFIED_RAG_USER_PROMPT,
)
from src.generation.rag_pipeline import Retriever, extract_citations
from src.utils.logging_setup import get_logger

logger = get_logger("generation.verified_rag_pipeline")

# ── Claim verification prompt ──────────────────────────────────────────────
CLAIM_VERIFY_SYSTEM = """You are a legal fact-checker. For each claim below, decide if it is SUPPORTED or UNSUPPORTED by the evidence passage.
Output ONLY one line per claim in the format:
CLAIM <N>: SUPPORTED
or
CLAIM <N>: UNSUPPORTED
Do not add any other text."""

CLAIM_VERIFY_USER = """Evidence passage:
{evidence}

Claims to verify:
{claims}

Verify each claim against the evidence."""

_CLAIM_VERDICT_RE = re.compile(r"CLAIM\s*(\d+)\s*:\s*(SUPPORTED|UNSUPPORTED)", re.IGNORECASE)


class CitationVerifier:
    """Post-generation citation verification.

    Checks every [chunk_id] in the generated answer against the set of
    actually-retrieved chunk IDs. Returns clean answer text, lists of
    valid/invalid citations, and a cleaned version with hallucinated
    citations stripped.
    """

    @staticmethod
    def verify(answer: str, retrieved_ids: List[str]) -> Dict[str, Any]:
        """Verify citations in the answer against retrieved evidence.

        Returns dict with:
          - cited_ids: all cited IDs in order of first appearance
          - valid_ids: cited AND present in retrieved evidence
          - invalid_ids: cited but NOT in retrieved evidence (hallucinated)
          - has_citation: bool
          - cleaned_answer: answer with hallucinated citations removed
          - citations_stripped: count of hallucinated citations removed
        """
        cited = extract_citations(answer)
        valid = [c for c in cited if c in retrieved_ids]
        invalid = [c for c in cited if c not in retrieved_ids]

        # Build cleaned answer by removing only hallucinated citation tokens
        cleaned = answer
        stripped_count = 0
        for inv_id in invalid:
            # Remove [inv_id] or (inv_id) with optional CHUNK prefix
            pattern = rf"(?:\[|\()(?:\s*CHUNK\s+)?{re.escape(inv_id)}(?:\]|\))"
            cleaned, n = re.subn(pattern, "", cleaned, flags=re.IGNORECASE)
            stripped_count += n

        # Clean up any resulting double spaces or trailing spaces
        cleaned = re.sub(r"  +", " ", cleaned).strip()

        return {
            "cited_ids": cited,
            "valid_ids": valid,
            "invalid_ids": invalid,
            "has_citation": len(cited) > 0,
            "cleaned_answer": cleaned,
            "citations_stripped": stripped_count,
        }


class ClaimVerifier:
    """LLM-based claim-level entailment verification.

    Splits a generated answer into individual factual claims (sentences),
    then asks the LLM to judge each claim against its cited evidence.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    @staticmethod
    def extract_claims(answer: str) -> List[str]:
        """Split answer into individual claim sentences.

        Filters out very short fragments and abstention phrases.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = []
        for s in sentences:
            s = s.strip()
            # Skip very short fragments, headers, or abstention
            if len(s) < 15:
                continue
            if any(p in s.lower() for p in ("insufficient evidence", "i don't know")):
                continue
            claims.append(s)
        return claims

    def verify_claims(
        self,
        claims: List[str],
        evidence_text: str,
    ) -> Dict[str, Any]:
        """Verify each claim against the evidence.

        Returns dict with:
          - claims: list of claim strings
          - verdicts: list of 'SUPPORTED' | 'UNSUPPORTED' | 'UNKNOWN'
          - supported_count: int
          - unsupported_count: int
          - unknown_count: int
          - support_rate: float (0.0 to 1.0)
        """
        if not claims:
            return {
                "claims": [],
                "verdicts": [],
                "supported_count": 0,
                "unsupported_count": 0,
                "unknown_count": 0,
                "support_rate": 1.0,  # vacuously true
            }

        # Format claims as numbered list
        claims_text = "\n".join(f"CLAIM {i+1}: {c}" for i, c in enumerate(claims))

        # Truncate evidence if very long to stay within context window
        max_evidence_chars = 3000
        if len(evidence_text) > max_evidence_chars:
            evidence_text = evidence_text[:max_evidence_chars] + "\n[...truncated...]"

        prompt = CLAIM_VERIFY_USER.format(evidence=evidence_text, claims=claims_text)
        raw = self.llm.generate(prompt, system_prompt=CLAIM_VERIFY_SYSTEM)

        # Parse verdicts
        verdicts = ["UNKNOWN"] * len(claims)
        for match in _CLAIM_VERDICT_RE.finditer(raw):
            idx = int(match.group(1)) - 1  # 1-indexed → 0-indexed
            verdict = match.group(2).upper()
            if 0 <= idx < len(claims):
                verdicts[idx] = verdict

        supported = sum(1 for v in verdicts if v == "SUPPORTED")
        unsupported = sum(1 for v in verdicts if v == "UNSUPPORTED")
        unknown = sum(1 for v in verdicts if v == "UNKNOWN")

        return {
            "claims": claims,
            "verdicts": verdicts,
            "supported_count": supported,
            "unsupported_count": unsupported,
            "unknown_count": unknown,
            "support_rate": supported / len(claims) if claims else 1.0,
        }


class VerifiedRAGPipeline:
    """Retrieve → Generate → Verify Citations → Verify Claims → Abstain/Accept.

    This is the proposed system for testing H2 (structure-aware retrieval)
    and H3 (explicit verification reduces unsupported claims).
    """

    def __init__(
        self,
        retriever: Retriever,
        chunk_store: Dict[str, Dict[str, Any]],
        gen_client: LLMClient,
        verify_client: LLMClient,
        top_k: int = 5,
        retriever_name: str = "hybrid",
        abstention_threshold: float = 0.5,
    ):
        """
        Args:
            retriever: Any retriever implementing the Retriever protocol.
            chunk_store: chunk_id → chunk dict mapping.
            gen_client: LLM client for answer generation.
            verify_client: LLM client for claim verification (may be same).
            top_k: Number of evidence chunks to retrieve.
            retriever_name: Tag for provenance tracking.
            abstention_threshold: If support_rate < this, abstain.
        """
        self.retriever = retriever
        self.chunk_store = chunk_store
        self.gen_client = gen_client
        self.citation_verifier = CitationVerifier()
        self.claim_verifier = ClaimVerifier(verify_client)
        self.top_k = top_k
        self.retriever_name = retriever_name
        self.abstention_threshold = abstention_threshold

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for c in chunks:
            blocks.append(RAG_EVIDENCE_TEMPLATE.format(
                chunk_id=c["chunk_id"],
                case_id=c.get("case_id", ""),
                pages=",".join(str(p) for p in c.get("pages", [])),
                sections=",".join(c.get("sections", [])),
                text=c.get("text", ""),
            ))
        return "\n\n".join(blocks)

    def _build_evidence_text(self, retrieved: List[Dict[str, Any]], cited_valid: List[str]) -> str:
        """Build combined evidence text from validly-cited chunks."""
        if cited_valid:
            chunks = [c for c in retrieved if c["chunk_id"] in cited_valid]
        else:
            # No valid citations; use all retrieved as evidence
            chunks = retrieved
        return "\n\n".join(c.get("text", "") for c in chunks)

    def answer(self, question: str, retrieval_query: Optional[str] = None) -> Dict[str, Any]:
        """Full Verified-RAG pipeline: Retrieve → Generate → Verify → Decide."""
        query = retrieval_query or question

        # ── Step 1: Retrieve ──────────────────────────────────────────────
        hits = self.retriever.retrieve(query, top_k=self.top_k)
        retrieved: List[Dict[str, Any]] = []
        missing: List[str] = []
        for chunk_id, score in hits:
            chunk = self.chunk_store.get(chunk_id)
            if chunk is None:
                missing.append(chunk_id)
                continue
            retrieved.append({**chunk, "score": score})
        if missing:
            logger.warning(f"{len(missing)} retrieved chunk IDs not in chunk store: {missing[:3]}")

        if not retrieved:
            return {
                "question": question,
                "retrieval_query": query,
                "answer": "Insufficient evidence to answer.",
                "raw_answer": "Insufficient evidence to answer.",
                "generation_failed": False,
                "model_used": "abstention_calibrator",
                "generation_latency_s": 0.0,
                "prompt_version": VERIFIED_RAG_PROMPT_VERSION,
                "retriever": self.retriever_name,
                "top_k": self.top_k,
                "retrieved": [],
                "citations": {
                    "cited_ids": [],
                    "valid_ids": [],
                    "invalid_ids": [],
                    "has_citation": False,
                    "cleaned_answer": "Insufficient evidence to answer.",
                    "citations_stripped": 0,
                },
                "verification": {
                    "claims_total": 0,
                    "supported_count": 0,
                    "unsupported_count": 0,
                    "support_rate": 0.0,
                    "claims": [],
                    "evidence_used": "",
                    "abstained": True,
                    "abstention_reason": "no_evidence_retrieved",
                    "abstention_threshold": self.abstention_threshold,
                    "verification_latency_s": 0.0,
                },
            }

        # ── Step 2: Generate ──────────────────────────────────────────────
        context = self._format_context(retrieved)
        prompt = VERIFIED_RAG_USER_PROMPT.format(context=context, question=question)
        raw_answer = self.gen_client.generate(prompt, system_prompt=VERIFIED_RAG_SYSTEM_PROMPT)
        gen_failed = raw_answer.startswith("ERROR:")
        gen_latency = self.gen_client.last_latency_s
        gen_model = self.gen_client.last_model_used

        # ── Step 3: Verify Citations ──────────────────────────────────────
        retrieved_ids = [c["chunk_id"] for c in retrieved]
        cit_result = self.citation_verifier.verify(raw_answer, retrieved_ids)
        cleaned_answer = cit_result["cleaned_answer"]

        # ── Step 4: Verify Claims ─────────────────────────────────────────
        claims = ClaimVerifier.extract_claims(cleaned_answer)
        evidence_text = self._build_evidence_text(retrieved, cit_result["valid_ids"])
        claim_result = self.claim_verifier.verify_claims(claims, evidence_text)
        verify_latency = self.claim_verifier.llm.last_latency_s
        verify_model = self.claim_verifier.llm.last_model_used

        # ── Step 5: Abstention Decision ───────────────────────────────────
        should_abstain = False
        abstention_reason = None
        if gen_failed:
            should_abstain = True
            abstention_reason = "generation_failed"
        elif not claims:
            # The answer was too short to contain verifiable claims
            # Check if it's already an abstention phrase
            lower = cleaned_answer.lower().strip()
            if any(p in lower for p in ("insufficient evidence", "i don't know")):
                should_abstain = True
                abstention_reason = "model_abstained"
        elif claim_result["support_rate"] < self.abstention_threshold:
            should_abstain = True
            abstention_reason = (
                f"support_rate={claim_result['support_rate']:.2f} "
                f"< threshold={self.abstention_threshold:.2f}"
            )

        final_answer = (
            "Insufficient evidence to answer."
            if should_abstain and abstention_reason != "generation_failed"
            else cleaned_answer
        )

        return {
            "question": question,
            "retrieval_query": query,
            "answer": final_answer,
            "raw_answer": raw_answer,
            "generation_failed": gen_failed,
            "model_used": gen_model,
            "generation_latency_s": gen_latency,
            "verify_model_used": verify_model,
            "verify_latency_s": verify_latency,
            "prompt_version": VERIFIED_RAG_PROMPT_VERSION,
            "retriever": self.retriever_name,
            "top_k": self.top_k,
            "retrieved": [
                {
                    "chunk_id": c["chunk_id"],
                    "case_id": c.get("case_id"),
                    "pages": c.get("pages"),
                    "paragraphs": c.get("paragraphs"),
                    "sections": c.get("sections"),
                    "source_reference": c.get("source_reference"),
                    "score": c["score"],
                }
                for c in retrieved
            ],
            "missing_chunk_ids": missing,
            "citations": {
                "cited_ids": cit_result["cited_ids"],
                "valid_ids": cit_result["valid_ids"],
                "invalid_ids": cit_result["invalid_ids"],
                "has_citation": cit_result["has_citation"],
                "citations_stripped": cit_result["citations_stripped"],
            },
            "verification": {
                "claims": claim_result["claims"],
                "verdicts": claim_result["verdicts"],
                "supported_count": claim_result["supported_count"],
                "unsupported_count": claim_result["unsupported_count"],
                "unknown_count": claim_result["unknown_count"],
                "support_rate": claim_result["support_rate"],
                "abstained": should_abstain,
                "abstention_reason": abstention_reason,
                "abstention_threshold": self.abstention_threshold,
            },
        }
