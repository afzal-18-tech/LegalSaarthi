"""
LegalSaarthi — Generation Metrics Evaluator.
Uses an LLM-as-a-judge to evaluate generated answers against gold standard answers.
"""
import re
from typing import Dict, Any, Optional, Tuple
from src.generation.llm_client import LLMClient
from src.utils.logging_setup import get_logger

logger = get_logger("evaluation.generation_metrics")

ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "cannot be verified",
    "insufficient evidence",
)

JUDGE_SYSTEM_PROMPT = """You are an expert legal evaluator. Your task is to grade the correctness and factuality of an AI-generated answer against a gold standard answer.

You must score the generated answer on two dimensions from 1 to 5:
1. Correctness: Does the generated answer correctly address the question based on the gold answer? (1 = completely wrong, 5 = perfectly correct)
2. Factuality: Does the generated answer contain hallucinations or false claims not present in the gold answer? (1 = severe hallucinations, 5 = entirely factual/no hallucinations)

If the generated answer is "I don't know" or a similar abstention, score both as 0 (Abstention).

Output your evaluation exactly in this format (and nothing else):
CORRECTNESS: <score>
FACTUALITY: <score>
"""

JUDGE_USER_PROMPT = """Question: {question}

Gold Standard Answer: {gold_answer}

Generated Answer: {generated_answer}

Evaluate the generated answer.
"""

class LLMEvaluator:
    """Evaluates generation quality using an LLM judge."""

    def __init__(self, judge_client: LLMClient):
        self.judge_client = judge_client

    def evaluate(self, question: str, gold_answer: str, generated_answer: str) -> Dict[str, Any]:
        """Scores a single generation."""
        # Simple heuristic for abstention
        lower_gen = generated_answer.lower().strip()
        if any(p in lower_gen for p in ABSTENTION_PHRASES):
            return {"correctness": 0, "factuality": 0, "abstention": 1,
                    "judge_parse_failure": False, "raw_judge_output": "Auto-scored: Abstention"}

        prompt = JUDGE_USER_PROMPT.format(
            question=question, 
            gold_answer=gold_answer, 
            generated_answer=generated_answer
        )
        
        judge_output = self.judge_client.generate(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        
        # Parse output
        correctness = self._extract_score("CORRECTNESS", judge_output)
        factuality = self._extract_score("FACTUALITY", judge_output)
        
        return {
            "correctness": correctness,
            "factuality": factuality,
            "abstention": 0,
            "judge_parse_failure": correctness is None or factuality is None,
            "raw_judge_output": judge_output
        }
        
    def _extract_score(self, key: str, text: str) -> Optional[int]:
        match = re.search(rf"{key}:\s*(\d)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Never invent a score: report the failure so it can be excluded and counted.
        logger.warning(f"Failed to extract {key} from judge output: {text}")
        return None
