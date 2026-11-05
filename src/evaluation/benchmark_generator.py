"""
LegalSaarthi — Benchmark Generator (Phase 9, revised for Phase 10)

Synthesizes a QA benchmark from generated chunks for retrieval evaluation.
Creates Easy, Medium, and Hard queries using CONTENT-BASED extraction
so that retrieval systems can match on actual text signals.
"""
import random
import re
from typing import Dict, List, Any

from src.utils.logging_setup import get_logger

logger = get_logger("evaluation.benchmark")


def _extract_key_phrases(text: str, n_words: int = 15) -> str:
    """Extract a representative phrase from the chunk text.
    
    Takes a segment from the middle of the text (avoids headers/footers)
    and cleans it up for use as a query fragment.
    """
    words = text.split()
    if len(words) <= n_words:
        return " ".join(words)
    
    # Take from the middle third for richer content
    start = len(words) // 3
    return " ".join(words[start:start + n_words])


def _extract_legal_entities(text: str) -> List[str]:
    """Extract statute/section references and case citations from text."""
    entities = []
    # Statute references: "Section 302", "Article 21", etc.
    statutes = re.findall(r"(?:Section|Article|Order|Rule)\s+\d+[A-Za-z]?(?:\(\d+\))?", text, re.IGNORECASE)
    entities.extend(statutes[:3])
    # Case citations: patterns like "(2015) 5 SCC 123" or "AIR 1990 SC 123"
    citations = re.findall(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+|\bAIR\s+\d{4}\s+SC\s+\d+", text)
    entities.extend(citations[:2])
    return entities


class BenchmarkGenerator:
    """Generates content-based synthetic queries targeting specific chunks."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(self.seed)
        # Build a chunk_id -> chunk dict for content extraction
        self._chunk_map: Dict[str, Dict[str, Any]] = {}

    def _group_chunks_by_case_and_section(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
        """Groups chunk IDs by case_id and section_label, also builds chunk_map."""
        case_section_map = {}
        for chunk in chunks:
            case_id = chunk["case_id"]
            chunk_id = chunk["chunk_id"]
            self._chunk_map[chunk_id] = chunk
            
            if case_id not in case_section_map:
                case_section_map[case_id] = {}
            
            section = chunk["sections"][0] if chunk["sections"] else "UNKNOWN"
            if section not in case_section_map[case_id]:
                case_section_map[case_id][section] = []
            
            case_section_map[case_id][section].append(chunk_id)
        
        return case_section_map

    def generate_easy_queries(self, case_section_map: Dict[str, Dict[str, List[str]]], target_count: int = 20) -> List[Dict[str, Any]]:
        """
        Easy: 1 chunk required. Query is constructed from the chunk's own content.
        """
        queries = []
        cases = list(case_section_map.keys())
        random.shuffle(cases)
        
        query_idx = 1
        for case_id in cases:
            if len(queries) >= target_count:
                break
                
            sections = case_section_map[case_id]
            
            # Try FACTS first, then DECISION
            target_section = None
            q_type = None
            if "FACTS" in sections and sections["FACTS"]:
                target_section = "FACTS"
                q_type = "factual"
            elif "DECISION" in sections and sections["DECISION"]:
                target_section = "DECISION"
                q_type = "dispositional"
            
            if target_section:
                chunk_id = random.choice(sections[target_section])
                chunk_text = self._chunk_map[chunk_id]["text"]
                key_phrase = _extract_key_phrases(chunk_text, n_words=20)
                
                queries.append({
                    "query_id": f"EASY_{query_idx:03d}",
                    "case_id": case_id,
                    "question_text": key_phrase,
                    "question_type": q_type,
                    "difficulty": "Easy",
                    "gold_evidence": [chunk_id]
                })
                query_idx += 1

        return queries

    def generate_medium_queries(self, case_section_map: Dict[str, Dict[str, List[str]]], target_count: int = 20) -> List[Dict[str, Any]]:
        """
        Medium: 2-3 chunks required from the same section.
        Query combines content from multiple chunks.
        """
        queries = []
        cases = list(case_section_map.keys())
        random.shuffle(cases)
        
        query_idx = 1
        for case_id in cases:
            if len(queries) >= target_count:
                break
                
            sections = case_section_map[case_id]
            if "ARGUMENTS" in sections and len(sections["ARGUMENTS"]) >= 2:
                k = min(3, len(sections["ARGUMENTS"]))
                chunk_ids = random.sample(sections["ARGUMENTS"], k)
                
                # Build query from content of each gold chunk
                phrases = []
                for cid in chunk_ids:
                    chunk_text = self._chunk_map[cid]["text"]
                    phrases.append(_extract_key_phrases(chunk_text, n_words=10))
                
                query_text = " ".join(phrases)
                
                queries.append({
                    "query_id": f"MED_{query_idx:03d}",
                    "case_id": case_id,
                    "question_text": query_text,
                    "question_type": "arguments",
                    "difficulty": "Medium",
                    "gold_evidence": chunk_ids
                })
                query_idx += 1

        return queries

    def generate_hard_queries(self, case_section_map: Dict[str, Dict[str, List[str]]], target_count: int = 10) -> List[Dict[str, Any]]:
        """
        Hard: Chunks spanning LAW, PRECEDENTS, and REASONING.
        Query combines legal entities and reasoning phrases.
        """
        queries = []
        cases = list(case_section_map.keys())
        random.shuffle(cases)
        
        query_idx = 1
        for case_id in cases:
            if len(queries) >= target_count:
                break
                
            sections = case_section_map[case_id]
            has_law = "LAW" in sections and sections["LAW"]
            has_prec = "PRECEDENTS" in sections and sections["PRECEDENTS"]
            has_reas = "REASONING" in sections and sections["REASONING"]
            
            if has_law and has_prec and has_reas:
                law_cid = random.choice(sections["LAW"])
                prec_cid = random.choice(sections["PRECEDENTS"])
                reas_cid = random.choice(sections["REASONING"])
                
                # Build query combining legal entities from LAW + content from REASONING
                law_text = self._chunk_map[law_cid]["text"]
                reas_text = self._chunk_map[reas_cid]["text"]
                prec_text = self._chunk_map[prec_cid]["text"]
                
                entities = _extract_legal_entities(law_text) + _extract_legal_entities(prec_text)
                reas_phrase = _extract_key_phrases(reas_text, n_words=12)
                
                entity_str = " ".join(entities[:3]) if entities else _extract_key_phrases(law_text, n_words=8)
                query_text = f"{entity_str} {reas_phrase}"
                
                chunk_ids = [law_cid, prec_cid, reas_cid]
                queries.append({
                    "query_id": f"HARD_{query_idx:03d}",
                    "case_id": case_id,
                    "question_text": query_text,
                    "question_type": "legal_reasoning",
                    "difficulty": "Hard",
                    "gold_evidence": chunk_ids
                })
                query_idx += 1

        return queries

    def generate_benchmark(self, chunks: List[Dict[str, Any]], easy_n: int = 20, med_n: int = 20, hard_n: int = 10) -> List[Dict[str, Any]]:
        self._chunk_map = {}
        case_section_map = self._group_chunks_by_case_and_section(chunks)
        
        easy_q = self.generate_easy_queries(case_section_map, easy_n)
        med_q = self.generate_medium_queries(case_section_map, med_n)
        hard_q = self.generate_hard_queries(case_section_map, hard_n)
        
        all_queries = easy_q + med_q + hard_q
        logger.info(f"Generated {len(all_queries)} queries (Easy: {len(easy_q)}, Medium: {len(med_q)}, Hard: {len(hard_q)})")
        
        return all_queries
