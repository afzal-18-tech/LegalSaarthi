"""
LegalSaarthi — Phase 18 Robustness & Evidence-Failure runner.

Runs Verified-RAG (BM25 k=5, tau=0.5 frozen from Phase 17) under controlled
evidence-failure conditions. Defaults to the VALIDATION split only; the locked
test split is excluded unless --splits includes it explicitly.

Usage:
  python -m scripts.run_robustness_eval --limit 3 --conditions clean no_evidence   # smoke
  python -m scripts.run_robustness_eval
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from scripts.run_generation_eval import BENCHMARK, CHUNKS, SPLITS, load_retriever, model_tag
from src.evaluation.generation_eval import build_question, load_chunk_store, load_split_map
from src.evaluation.generation_metrics import JUDGE_SYSTEM_PROMPT, LLMEvaluator
from src.evaluation.robustness import (
    CONDITIONS, EXPECT_ABSTAIN, OOD_QUESTIONS, PerturbingRetriever, summarize_condition,
)
from src.generation.llm_client import LLMClient
from src.generation.verified_rag_pipeline import VerifiedRAGPipeline
from src.utils.config import load_config
from src.utils.logging_setup import get_logger, setup_logging

logger = get_logger("evaluation.robustness_runner")


def pct(x):
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def f2(x):
    return f"{x:.2f}" if x is not None else "n/a"


def main():
    p = argparse.ArgumentParser(description="Phase 18 robustness evaluation")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--splits", nargs="+", default=["val"])
    p.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    load_dotenv()
    setup_logging()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    gen_cfg = cfg.get("generation", {})
    gen_model = gen_cfg["model_name"]
    judge_model = gen_cfg.get("judge_model", gen_model)

    run_id = f"robustness__{args.retriever}-k{args.top_k}__{model_tag(gen_model)}__" \
             f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.limit:
        run_id += f"__smoke{args.limit}"
    out_dir = Path("experiments/phase18") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(gen_model, fallback_model=None, temperature=gen_cfg.get("temperature", 0.0),
                    max_tokens=gen_cfg.get("max_tokens", 512))
    verify = LLMClient(gen_model, fallback_model=None, temperature=0.0, max_tokens=256)
    judge = LLMClient(judge_model, fallback_model=None, temperature=0.0, max_tokens=512)
    evaluator = LLMEvaluator(judge)

    chunk_store = load_chunk_store(CHUNKS)
    split_map = load_split_map(SPLITS)
    benchmark = [b for b in json.loads(BENCHMARK.read_text(encoding="utf-8"))
                 if split_map.get(b["case_id"], "unknown") in args.splits]
    if args.limit:
        benchmark = benchmark[: args.limit]

    dense_model = cfg.get("retrieval", {}).get("dense", {}).get("model_name", "all-MiniLM-L6-v2")
    base = load_retriever(args.retriever, dense_model, cfg.get("retrieval", {}).get("hybrid", {}))

    run_config = {
        "run_id": run_id, "phase": 18, "generator": gen_model, "judge": judge_model,
        "retriever": args.retriever, "top_k": args.top_k, "tau": args.tau, "seed": args.seed,
        "conditions": args.conditions, "splits": args.splits, "n_queries": len(benchmark),
        "ood_questions": OOD_QUESTIONS, "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "python": platform.python_version(), "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    summaries = {}
    for cond in args.conditions:
        store = dict(chunk_store)  # isolated copy per condition
        pr = PerturbingRetriever(base, store, cond, seed=args.seed)
        pipe = VerifiedRAGPipeline(pr, store, llm, verify, top_k=args.top_k,
                                   retriever_name=f"{args.retriever}+{cond}",
                                   abstention_threshold=args.tau)
        items = ([{"query_id": f"OOD_{i+1:03d}", "case_id": "", "gold_evidence": [],
                   "question": q, "difficulty": "n/a", "question_type": "out_of_domain"}
                  for i, q in enumerate(OOD_QUESTIONS)]
                 if cond == "out_of_domain" else benchmark)
        rows = []
        with open(out_dir / f"predictions_{cond}.jsonl", "w", encoding="utf-8") as f:
            for item in tqdm(items, desc=cond):
                question = item["question"] if cond == "out_of_domain" else build_question(item)
                gold = item["gold_evidence"]
                pr.set_gold(gold, item["case_id"])
                rec = pipe.answer(question)
                rids = [r["chunk_id"] for r in rec["retrieved"]]
                rec["gold_in_context"] = any(g in rids for g in gold)
                rec["tampered_in_context"] = any(i.endswith("__tampered") for i in rids)
                if cond in EXPECT_ABSTAIN or rec["generation_failed"] or rec["verification"]["abstained"]:
                    ev = {"correctness": None, "factuality": None, "note": "judge skipped"}
                else:
                    gold_answer = "\n\n".join(chunk_store.get(g, {}).get("text", "") for g in gold)
                    ev = evaluator.evaluate(question, gold_answer, rec["answer"])
                rec.update({"query_id": item["query_id"], "case_id": item["case_id"],
                            "condition": cond, "gold_evidence": gold, "evaluation": ev})
                rows.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                pr.restore()
        s = summarize_condition(rows, cond)
        if cond == "contradictory":
            s["tampered_in_context_rate"] = sum(r["tampered_in_context"] for r in rows) / max(len(rows), 1)
        summaries[cond] = s
        logger.info(f"{cond}: {s}")

    (out_dir / "metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    lines = [f"# Phase 18 — Robustness & Evidence-Failure Report", "",
             f"- Run ID: `{run_id}`", f"- Generator/Judge: `{gen_model}` / `{judge_model}`",
             f"- Retriever: {args.retriever} k={args.top_k}, tau={args.tau}, seed={args.seed}",
             f"- Splits: {args.splits} (n={len(benchmark)} queries; OOD probes n={len(OOD_QUESTIONS)})", "",
             "| Condition | n | Abstain | Correct-abstain* | Unsupported-answer* | Gold in ctx | Mean support | Invalid cit. | Corr(ans) | Fact(ans) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c, s in summaries.items():
        lines.append(f"| {c} | {s['n']} | {pct(s.get('abstention_rate'))} | {pct(s.get('correct_abstention_rate'))} | "
                     f"{pct(s.get('unsupported_answer_rate'))} | {pct(s.get('gold_in_context_rate'))} | "
                     f"{pct(s.get('mean_support_rate'))} | {pct(s.get('invalid_citation_rate'))} | "
                     f"{f2(s.get('correctness_answered'))} | {f2(s.get('factuality_answered'))} |")
    lines += ["", "\\* only defined for conditions where abstention is the correct behaviour "
              "(no_evidence, distractor, out_of_domain).", "",
              "## Caveats",
              "- Perturbations are synthetic; real retrieval failures may differ.",
              "- `contradictory` has no abstention ground truth; read abstention and tampered-citation rows descriptively.",
              "- Claim verifier is the same LLM as the generator; shared blind spots are possible.",
              "- Small n; no statistical significance claimed. Locked test split not used."]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
