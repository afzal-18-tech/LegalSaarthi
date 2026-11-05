#!/usr/bin/env python3
"""
LegalSaarthi — Benchmark Generation Script (Phase 9)

Synthesizes a gold-standard QA benchmark for retrieval evaluation.
Only samples cases from the 'val' and 'test' splits to prevent data leakage.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_generator import BenchmarkGenerator
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger("evaluation.benchmark")

def parse_args():
    parser = argparse.ArgumentParser(description="Build QA Benchmark from chunks.")
    parser.add_argument("--easy", type=int, default=20, help="Number of easy queries.")
    parser.add_argument("--medium", type=int, default=20, help="Number of medium queries.")
    parser.add_argument("--hard", type=int, default=10, help="Number of hard queries.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    setup_logging(config)
    
    # Paths
    splits_file = PROJECT_ROOT / "data/metadata/split_manifest.csv"
    chunks_file = PROJECT_ROOT / config.get("chunking", {}).get("output_dir", "data/processed/chunks") / "strategy=section_aware" / "chunks.jsonl"
    output_dir = PROJECT_ROOT / "data/benchmark"
    output_file = output_dir / "qa_ground_truth.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not splits_file.exists():
        logger.error(f"Splits file not found: {splits_file}. Run Phase 5.")
        sys.exit(1)
        
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}. Run Phase 8.")
        sys.exit(1)
        
    # Load allowed case IDs (val + test)
    import pandas as pd
    df = pd.read_csv(splits_file)
    val_cases = set(df[df["split"] == "val"]["case_id"])
    test_cases = set(df[df["split"] == "test"]["case_id"])
    allowed_cases = val_cases.union(test_cases)
    
    logger.info(f"Loaded {len(allowed_cases)} allowed cases (val: {len(val_cases)}, test: {len(test_cases)})")
    
    # Load chunks belonging ONLY to allowed cases
    eligible_chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["case_id"] in allowed_cases:
                eligible_chunks.append(chunk)
                
    logger.info(f"Loaded {len(eligible_chunks)} eligible chunks from val/test cases.")
    
    # Generate benchmark
    generator = BenchmarkGenerator(seed=42)
    benchmark = generator.generate_benchmark(
        chunks=eligible_chunks,
        easy_n=args.easy,
        med_n=args.medium,
        hard_n=args.hard
    )
    
    # Save output
    output_file.write_text(json.dumps(benchmark, indent=2))
    
    logger.info("=" * 60)
    logger.info("BENCHMARK CONSTRUCTION REPORT")
    logger.info("=" * 60)
    logger.info(f"Target queries:      {args.easy + args.medium + args.hard}")
    logger.info(f"Generated queries:   {len(benchmark)}")
    logger.info(f"Output saved to:     {output_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
