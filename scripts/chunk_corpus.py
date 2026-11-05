#!/usr/bin/env python3
"""
LegalSaarthi — Corpus Chunking Script (Phase 8)

Reads structured JSON documents, applies the selected chunking strategy,
and outputs a flat JSONL file ready for embedding/retrieval.
"""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.chunker import DocumentChunker
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger("preprocessing.chunking")

def parse_args():
    parser = argparse.ArgumentParser(description="Chunk structured JSON documents.")
    parser.add_argument("--strategy", type=str, choices=["paragraph", "fixed_size", "section_aware"], 
                        help="Chunking strategy to use (overrides config).")
    parser.add_argument("--max_words", type=int, help="Max words per chunk (overrides config).")
    parser.add_argument("--overlap", type=int, help="Overlap words (overrides config).")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (for testing).")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    setup_logging(config)
    
    # Configuration
    strategy = args.strategy or config.get("chunking", {}).get("default_strategy", "section_aware")
    max_words = args.max_words or config.get("chunking", {}).get("max_words", 300)
    overlap_words = args.overlap if args.overlap is not None else config.get("chunking", {}).get("overlap_words", 50)
    
    input_dir = PROJECT_ROOT / config.get("structure_detection", {}).get("output_dir", "data/processed/structured")
    output_dir = PROJECT_ROOT / config.get("chunking", {}).get("output_dir", "data/processed/chunks")
    
    # Save into strategy-specific subfolder
    strategy_out_dir = output_dir / f"strategy={strategy}"
    strategy_out_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = strategy_out_dir / "chunks.jsonl"
    stats_file = strategy_out_dir / "chunking_stats.json"
    
    logger.info(f"Starting chunking (strategy: {strategy}, max_words: {max_words}, overlap: {overlap_words})")
    
    chunker = DocumentChunker(max_words=max_words, overlap_words=overlap_words)
    
    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        logger.error(f"No structured JSON files found in {input_dir}")
        sys.exit(1)
        
    if args.limit > 0:
        json_files = json_files[:args.limit]
        
    total_docs = 0
    total_chunks = 0
    total_words = 0
    section_counts = Counter()
    doc_chunk_counts = Counter()
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for json_path in tqdm(json_files, desc="Chunking"):
            try:
                doc = json.loads(json_path.read_text(encoding="utf-8"))
                case_id = doc.get("case_id", json_path.stem)
                
                if strategy == "paragraph":
                    chunks = chunker.chunk_by_paragraph(doc)
                elif strategy == "fixed_size":
                    chunks = chunker.chunk_fixed_size(doc)
                elif strategy == "section_aware":
                    chunks = chunker.chunk_by_section(doc)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                doc_chunk_counts[case_id] = len(chunks)
                total_docs += 1
                
                for chunk in chunks:
                    total_chunks += 1
                    total_words += chunk.word_count
                    for sec in chunk.sections:
                        section_counts[sec] += 1
                        
                    # Write as JSONL
                    out_f.write(json.dumps(chunk.to_dict()) + "\n")
                    
            except Exception as e:
                logger.error(f"Failed to chunk {json_path.name}: {e}")
                
    # Compile statistics
    stats = {
        "strategy": strategy,
        "max_words": max_words,
        "overlap_words": overlap_words,
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "average_chunks_per_doc": round(total_chunks / max(1, total_docs), 2),
        "average_words_per_chunk": round(total_words / max(1, total_chunks), 2),
        "section_distribution": dict(section_counts)
    }
    
    stats_file.write_text(json.dumps(stats, indent=2))
    
    logger.info("=" * 60)
    logger.info("CHUNKING REPORT")
    logger.info("=" * 60)
    logger.info(f"Documents processed:   {total_docs}")
    logger.info(f"Total chunks created:  {total_chunks}")
    logger.info(f"Avg words per chunk:   {stats['average_words_per_chunk']}")
    logger.info(f"Output saved to:       {output_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
