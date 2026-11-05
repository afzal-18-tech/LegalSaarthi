"""
LegalSaarthi — Unified generation evaluation harness (Phases 12–14).

Runs one system over the Phase 9 benchmark under a fixed protocol:
  --system generic      : zero-shot LLM, no retrieval (Phase 12)
  --system rag          : Basic RAG baseline (Phase 13), --retriever dense|bm25
  --system verified-rag : Verified-RAG (Phase 14), --retriever dense|bm25|hybrid

Same questions, same judge prompt, same metric definitions for every system.
Fallback models are DISABLED by default so one run = one generator
(mixing generators would confound comparisons). Raw predictions are kept.

Usage:
  python -m scripts.run_generation_eval --system rag --retriever dense
  python -m scripts.run_generation_eval --system verified-rag --retriever hybrid
  python -m scripts.run_generation_eval --system generic
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.evaluation.generation_eval import (
    aggregate, aggregate_by, build_question, fmt, load_chunk_store, load_split_map,
)
from src.evaluation.generation_metrics import JUDGE_SYSTEM_PROMPT, LLMEvaluator
from src.generation.llm_client import LLMClient
from src.generation.prompts import (
    GENERIC_BASELINE_SYSTEM_PROMPT, GENERIC_BASELINE_USER_PROMPT,
    RAG_PROMPT_VERSION, RAG_SYSTEM_PROMPT,
    VERIFIED_RAG_PROMPT_VERSION, VERIFIED_RAG_SYSTEM_PROMPT,
)
from src.generation.rag_pipeline import BasicRAGPipeline
from src.generation.verified_rag_pipeline import VerifiedRAGPipeline
from src.utils.config import load_config
from src.utils.logging_setup import get_logger, setup_logging

logger = get_logger("evaluation.generation_harness")

BENCHMARK = Path("data/benchmark/qa_ground_truth.json")
CHUNKS = Path("data/processed/chunks/strategy=section_aware/chunks.jsonl")
SPLITS = Path("data/metadata/split_manifest.csv")
INDEX_DIRS = {"dense": Path("experiments/phase10/dense_index"), "bm25": Path("experiments/phase10/bm25_index")}


def set_index_root(root: str) -> None:
    """Point retrievers at a different index root (e.g. experiments/phase19 for the 51k-chunk corpus)."""
    INDEX_DIRS["dense"] = Path(root) / "dense_index"
    INDEX_DIRS["bm25"] = Path(root) / "bm25_index"


def model_tag(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def load_retriever(name: str, dense_model: str, hybrid_cfg: dict = None):
    if name == "hybrid":
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        bm25 = BM25Retriever()
        bm25.load_index(INDEX_DIRS["bm25"])
        dense = DenseRetriever(model_name=dense_model)
        dense.load_index(INDEX_DIRS["dense"])
        hcfg = hybrid_cfg or {}
        return HybridRetriever(
            bm25, dense,
            fusion_method=hcfg.get("fusion_method", "rrf"),
            k_rrf=hcfg.get("k_rrf", 60),
            alpha=hcfg.get("alpha", 0.5),
            candidate_k=hcfg.get("candidate_k", 50),
        )
    elif name == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever
        r = BM25Retriever()
    else:
        from src.retrieval.dense_retriever import DenseRetriever
        r = DenseRetriever(model_name=dense_model)
    r.load_index(INDEX_DIRS[name])
    return r


def main():
    p = argparse.ArgumentParser(description="Unified generation evaluation harness")
    p.add_argument("--system", choices=["generic", "rag", "verified-rag"], required=True)
    p.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="dense")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="Smoke-test on first N queries")
    p.add_argument("--model", default=None, help="Override generator model")
    p.add_argument("--judge", default=None, help="Override judge model")
    p.add_argument("--allow-fallback", action="store_true", help="Permit fallback model (confounds results)")
    p.add_argument("--phase-dir", default=None, help="Override output experiments phase directory (e.g. phase16)")
    p.add_argument("--index-root", default=None, help="Index root holding bm25_index/ and dense_index/ (default: experiments/phase10)")
    p.add_argument("--phase-num", type=int, default=None, help="Phase number recorded in run_config")
    args = p.parse_args()
    if args.index_root:
        set_index_root(args.index_root)

    load_dotenv()
    setup_logging()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    gen_cfg = cfg.get("generation", {})

    gen_model = args.model or gen_cfg["model_name"]
    judge_model = args.judge or gen_cfg.get("judge_model", gen_model)
    fallback = gen_cfg.get("fallback_model") if args.allow_fallback else None
    temperature = gen_cfg.get("temperature", 0.0)
    max_tokens = gen_cfg.get("max_tokens", 512)

    if args.system == "generic":
        system_name = "generic"
    elif args.system == "verified-rag":
        system_name = f"verified-rag-{args.retriever}-k{args.top_k}"
    else:
        system_name = f"rag-{args.retriever}-k{args.top_k}"
    run_id = f"{system_name}__{model_tag(gen_model)}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.limit:
        run_id += f"__smoke{args.limit}"
    if args.phase_dir:
        phase_dir = args.phase_dir
    else:
        phase_dir = "phase14" if args.system == "verified-rag" else "phase13"
    out_dir = Path(f"experiments/{phase_dir}") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"RUN {run_id}")
    logger.info(f"Generator={gen_model} Judge={judge_model} Fallback={fallback} Temp={temperature}")
    logger.info("=" * 60)

    llm = LLMClient(gen_model, fallback_model=fallback, temperature=temperature, max_tokens=max_tokens)
    judge = LLMClient(judge_model, fallback_model=fallback, temperature=0.0, max_tokens=512)
    evaluator = LLMEvaluator(judge)

    chunk_store = load_chunk_store(CHUNKS)
    split_map = load_split_map(SPLITS)
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if args.limit:
        benchmark = benchmark[: args.limit]

    pipeline = None
    verified_pipeline = None
    dense_model = cfg.get("retrieval", {}).get("dense", {}).get("model_name", "all-MiniLM-L6-v2")
    hybrid_cfg = cfg.get("retrieval", {}).get("hybrid", {})

    if args.system == "rag":
        pipeline = BasicRAGPipeline(load_retriever(args.retriever, dense_model), chunk_store, llm,
                                    top_k=args.top_k, retriever_name=args.retriever)
    elif args.system == "verified-rag":
        verify_client = LLMClient(gen_model, fallback_model=fallback, temperature=0.0, max_tokens=256)
        verified_pipeline = VerifiedRAGPipeline(
            retriever=load_retriever(args.retriever, dense_model, hybrid_cfg),
            chunk_store=chunk_store,
            gen_client=llm,
            verify_client=verify_client,
            top_k=args.top_k,
            retriever_name=args.retriever,
        )

    # Snapshot the exact configuration before generating anything.
    phase_num = args.phase_num or (14 if args.system == "verified-rag" else 13)
    if args.system == "verified-rag":
        pv, sp = VERIFIED_RAG_PROMPT_VERSION, VERIFIED_RAG_SYSTEM_PROMPT
    elif args.system == "rag":
        pv, sp = RAG_PROMPT_VERSION, RAG_SYSTEM_PROMPT
    else:
        pv, sp = "generic_v1", GENERIC_BASELINE_SYSTEM_PROMPT
    run_config = {
        "index_root": args.index_root or "experiments/phase10", "n_chunks_in_store": len(chunk_store),
        "run_id": run_id, "phase": phase_num, "system": system_name, "generator": gen_model, "judge": judge_model,
        "fallback_model": fallback, "temperature": temperature, "max_tokens": max_tokens,
        "retriever": args.retriever if args.system != "generic" else None,
        "top_k": args.top_k if args.system != "generic" else None,
        "prompt_version": pv,
        "system_prompt": sp,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "benchmark": str(BENCHMARK), "n_queries": len(benchmark), "limit": args.limit,
        "python": platform.python_version(), "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    results = []
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as pred_f:
        for item in tqdm(benchmark, desc=system_name):
            question = build_question(item)
            gold_ids = item["gold_evidence"]
            gold_answer = "\n\n".join(chunk_store.get(cid, {}).get("text", "") for cid in gold_ids)

            if verified_pipeline is not None:
                rec = verified_pipeline.answer(question)
                retrieved_ids = [r["chunk_id"] for r in rec["retrieved"]]
                hit = [g for g in gold_ids if g in retrieved_ids]
                rec["retrieval_eval"] = {
                    "gold_hit": len(hit) > 0,
                    "gold_recall": len(hit) / len(gold_ids) if gold_ids else 0.0,
                    "cited_gold": any(c in gold_ids for c in rec["citations"]["valid_ids"]),
                }
            elif pipeline is not None:
                rec = pipeline.answer(question)
                retrieved_ids = [r["chunk_id"] for r in rec["retrieved"]]
                hit = [g for g in gold_ids if g in retrieved_ids]
                rec["retrieval_eval"] = {
                    "gold_hit": len(hit) > 0,
                    "gold_recall": len(hit) / len(gold_ids) if gold_ids else 0.0,
                    "cited_gold": any(c in gold_ids for c in rec["citations"]["valid_ids"]),
                }
            else:
                answer = llm.generate(GENERIC_BASELINE_USER_PROMPT.format(question=question),
                                      system_prompt=GENERIC_BASELINE_SYSTEM_PROMPT)
                rec = {"answer": answer, "generation_failed": answer.startswith("ERROR:"),
                       "model_used": llm.last_model_used, "generation_latency_s": llm.last_latency_s}

            if rec["generation_failed"]:
                ev = {"correctness": None, "factuality": None, "abstention": 0,
                      "judge_parse_failure": False, "raw_judge_output": "Skipped: generation failed"}
            else:
                ev = evaluator.evaluate(question, gold_answer, rec["answer"])

            row = {
                "query_id": item["query_id"], "case_id": item["case_id"],
                "split": split_map.get(item["case_id"], "unknown"),
                "difficulty": item["difficulty"], "question_type": item["question_type"],
                "question": question, "gold_evidence": gold_ids, "gold_answer": gold_answer,
                **rec, "judge_model_used": judge.last_model_used, "evaluation": ev,
            }
            results.append(row)
            pred_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_f.flush()  # keep partial raw predictions if the run is interrupted

    metrics = {
        "overall": aggregate(results),
        "by_difficulty": aggregate_by(results, "difficulty"),
        "by_split": aggregate_by(results, "split"),
        "by_question_type": aggregate_by(results, "question_type"),
        "usage": {"generator": llm.get_usage_stats(), "judge": judge.get_usage_stats()},
        "models_used": sorted({r["model_used"] or "none" for r in results}),
    }
    run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    o = metrics["overall"]
    lines = [
        f"# Generation Eval — {system_name}", "",
        f"- Run ID: `{run_id}`", f"- Generator: `{gen_model}` | Judge: `{judge_model}` | Fallback: `{fallback}`",
        f"- Prompt version: `{run_config['prompt_version']}` | Queries: {o['count']}",
        f"- Models actually used: {metrics['models_used']}", "",
        "## Overall", "",
        "| Metric | Value |", "|---|---|",
        f"| Abstention rate | {fmt(o.get('abstention_rate'), True)} ({o.get('abstained')}/{o['count']}) |",
        f"| Correctness, answered only (1-5) | {fmt(o.get('correctness_answered'))} |",
        f"| Factuality, answered only (1-5) | {fmt(o.get('factuality_answered'))} |",
        f"| Correctness, all (abstain=0) | {fmt(o.get('correctness_all'))} |",
        f"| Judge parse failures | {o.get('judge_parse_failures')} |",
        f"| Generation failures | {o.get('generation_failures')} |",
        f"| Latency median / p95 (s) | {fmt(o.get('latency_median_s'))} / {fmt(o.get('latency_p95_s'))} |",
    ]
    if "gold_hit_rate" in o:
        lines += [
            f"| Gold evidence hit@{args.top_k} | {fmt(o['gold_hit_rate'], True)} |",
            f"| Gold evidence recall@{args.top_k} | {fmt(o['gold_recall'], True)} |",
            f"| Answers with >=1 citation | {fmt(o['citation_rate'], True)} |",
            f"| Invalid (non-retrieved) citation IDs | {fmt(o['invalid_citation_rate'], True)} |",
            f"| Answers citing a gold chunk | {fmt(o['cited_gold_rate'], True)} |",
        ]
    lines += ["", "## By difficulty", "", "| Difficulty | n | Abstain | Corr(ans) | Fact(ans) | Corr(all) |",
              "|---|---|---|---|---|---|"]
    for k, m in metrics["by_difficulty"].items():
        lines.append(f"| {k} | {m['count']} | {fmt(m['abstention_rate'], True)} | {fmt(m['correctness_answered'])} | "
                     f"{fmt(m['factuality_answered'])} | {fmt(m['correctness_all'])} |")
    lines += ["", "## By split (test split is evaluation-only; never used for tuning)", "",
              "| Split | n | Abstain | Corr(ans) | Fact(ans) |", "|---|---|---|---|---|"]
    for k, m in metrics["by_split"].items():
        lines.append(f"| {k} | {m['count']} | {fmt(m['abstention_rate'], True)} | "
                     f"{fmt(m['correctness_answered'])} | {fmt(m['factuality_answered'])} |")
    lines += ["", "## Caveats",
              "- LLM-as-judge on a single 1-5 scale; judge may share biases with the generator.",
              "- Gold answers are raw gold-chunk text, and benchmark queries are extracted from that text "
              "(high lexical overlap favours lexical retrieval).",
              "- n=50; differences of a few points are not statistically established."]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"Saved: {out_dir}")
    print(json.dumps({"run_id": run_id, "overall": o}, indent=2))


if __name__ == "__main__":
    main()
