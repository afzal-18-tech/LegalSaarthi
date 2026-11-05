"""
LegalSaarthi — Generic LLM Baseline (Phase 12)
Runs the generic zero-shot LLM baseline on the Phase 9 benchmark.
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

from src.generation.llm_client import LLMClient
from src.generation.prompts import GENERIC_BASELINE_SYSTEM_PROMPT, GENERIC_BASELINE_USER_PROMPT
from src.evaluation.generation_metrics import LLMEvaluator
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger
from dotenv import load_dotenv

logger = get_logger("generation.baselines")


def main():
    parser = argparse.ArgumentParser(description="Run Generic LLM Baseline (Zero-Shot)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to test")
    args = parser.parse_args()

    load_dotenv()
    setup_logging()
    config = load_config()

    gen_config = config.get("generation", {})
    model_name = gen_config.get("model_name", "gemini/gemini-1.5-flash")
    fallback_model = gen_config.get("fallback_model", None)
    temperature = gen_config.get("temperature", 0.0)
    max_tokens = gen_config.get("max_tokens", 512)
    judge_model = gen_config.get("judge_model", model_name) # default to same model for judging

    logger.info("=" * 60)
    logger.info("GENERIC LLM BASELINE")
    logger.info("=" * 60)
    logger.info(f"Primary Model: {model_name} (Temp: {temperature}, Max Tokens: {max_tokens})")
    if fallback_model:
        logger.info(f"Fallback Standby Model: {fallback_model}")
    logger.info(f"Judge Model: {judge_model}")

    # Initialize clients
    llm_client = LLMClient(model_name=model_name, fallback_model=fallback_model, temperature=temperature, max_tokens=max_tokens)
    judge_client = LLMClient(model_name=judge_model, fallback_model=fallback_model, temperature=0.0, max_tokens=100)
    evaluator = LLMEvaluator(judge_client=judge_client)

    # Load benchmark
    benchmark_path = Path("data/benchmark/qa_ground_truth.json")
    if not benchmark_path.exists():
        logger.error(f"Benchmark not found at {benchmark_path}. Run Phase 9 first.")
        return
        
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
        
    # Load chunks to construct the gold answer text
    chunks_path = Path("data/processed/chunks/strategy=section_aware/chunks.jsonl")
    chunk_map = {}
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                chunk_map[c["chunk_id"]] = c["text"]
    else:
        logger.error(f"Chunks not found at {chunks_path}. Run Phase 8.")
        return
        
    if args.limit:
        benchmark = benchmark[:args.limit]
        logger.info(f"Limiting to {args.limit} queries.")
        
    results = []
    
    for item in tqdm(benchmark, desc="Generating & Evaluating"):
        query_text = item["question_text"]
        q_type = item["question_type"]
        case_id = item["case_id"]
        difficulty = item["difficulty"]
        evidence_ids = item["gold_evidence"]
        
        # Reconstruct a semantic question for the LLM
        if q_type == "factual":
            query = f"What are the facts of case {case_id} involving: '{query_text}'?"
        elif q_type == "dispositional":
            query = f"What was the final decision in case {case_id} regarding: '{query_text}'?"
        elif q_type == "arguments":
            query = f"What were the arguments in case {case_id} related to: '{query_text}'?"
        elif q_type == "legal_reasoning":
            query = f"What was the legal reasoning in case {case_id} involving: '{query_text}'?"
        else:
            query = f"Provide details about case {case_id} regarding: '{query_text}'?"

        # Construct gold answer from chunk text
        gold_answer = "\n\n".join([chunk_map.get(cid, "") for cid in evidence_ids])
        
        # 1. Generate (Zero-shot)
        prompt = GENERIC_BASELINE_USER_PROMPT.format(question=query)
        generated_answer = llm_client.generate(prompt, system_prompt=GENERIC_BASELINE_SYSTEM_PROMPT)
        
        # 2. Evaluate
        eval_result = evaluator.evaluate(
            question=query,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )
        
        # 3. Store
        results.append({
            "query_id": item["query_id"],
            "case_id": case_id,
            "query": query,
            "difficulty": difficulty,
            "model_used": llm_client.last_model_used,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "evaluation": eval_result
        })
        
        # Pacing delay to avoid API rate limit spikes
        time.sleep(2.0)

    # Calculate Aggregates
    metrics = {"overall": {"count": 0, "correctness": 0.0, "factuality": 0.0, "abstention": 0}}
    
    for r in results:
        diff = r["difficulty"]
        if diff not in metrics:
            metrics[diff] = {"count": 0, "correctness": 0.0, "factuality": 0.0, "abstention": 0}
            
        metrics["overall"]["count"] += 1
        metrics[diff]["count"] += 1
        
        metrics["overall"]["correctness"] += r["evaluation"]["correctness"]
        metrics["overall"]["factuality"] += r["evaluation"]["factuality"]
        metrics["overall"]["abstention"] += r["evaluation"]["abstention"]
        
        metrics[diff]["correctness"] += r["evaluation"]["correctness"]
        metrics[diff]["factuality"] += r["evaluation"]["factuality"]
        metrics[diff]["abstention"] += r["evaluation"]["abstention"]

    # Average metrics
    for key, agg in metrics.items():
        if agg["count"] > 0:
            agg["correctness"] /= agg["count"]
            agg["factuality"] /= agg["count"]
            # abstention is left as absolute count or can be a rate, let's make it a rate
            agg["abstention_rate"] = agg["abstention"] / agg["count"]

    # Report
    exp_dir = Path("experiments/phase12")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    full_output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "judge_model": judge_model,
            "usage": llm_client.get_usage_stats(),
            "judge_usage": judge_client.get_usage_stats(),
        },
        "metrics": metrics,
        "results": results
    }
    
    with open(exp_dir / "generic_llm_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # Markdown Report
    report_lines = [
        "# Generic LLM Baseline Report (Zero-Shot)",
        "",
        f"**Generated**: {full_output['metadata']['timestamp']}",
        f"**Model**: {model_name}",
        f"**Judge**: {judge_model}",
        "",
        "## Overall Metrics",
        "",
        "| Correctness (1-5) | Factuality (1-5) | Abstention Rate |",
        "|-------------------|------------------|-----------------|",
        f"| {metrics['overall']['correctness']:.2f} | {metrics['overall']['factuality']:.2f} | {metrics['overall']['abstention_rate']:.1%} |",
        "",
        "## Breakdown by Difficulty",
        "",
        "| Difficulty | Correctness | Factuality | Abstention Rate | Count |",
        "|------------|-------------|------------|-----------------|-------|",
    ]
    
    for diff in ["Easy", "Medium", "Hard"]:
        if diff in metrics:
            m = metrics[diff]
            report_lines.append(
                f"| {diff} | {m['correctness']:.2f} | {m['factuality']:.2f} | {m['abstention_rate']:.1%} | {m['count']} |"
            )
            
    report_lines.extend([
        "",
        "## Cost & Usage",
        f"- **Generator Cost**: ${full_output['metadata']['usage']['cost_usd']:.4f} ({full_output['metadata']['usage']['total_tokens']} tokens)",
        f"- **Judge Cost**: ${full_output['metadata']['judge_usage']['cost_usd']:.4f} ({full_output['metadata']['judge_usage']['total_tokens']} tokens)",
    ])

    report_path = Path("reports/generic_llm_baseline_report.md")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    logger.info(f"Report saved to {report_path}")
    logger.info("=" * 60)
    logger.info("GENERIC LLM BASELINE COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
