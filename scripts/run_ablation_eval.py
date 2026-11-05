"""
LegalSaarthi — Phase 21 ablation evaluation.

Derives component-isolation tables from raw Phase 20 / Phase 21 predictions.
Only two things need new LLM calls here:
  * judging the raw answers that the claim-verifier gate withheld (counterfactual tau=0), and
  * the runs launched by scripts/run_phase21_runs.ps1 (verified-dense, summarization swaps).

Usage:
  python -m scripts.run_ablation_eval            # full analysis (judges gated raw answers)
  python -m scripts.run_ablation_eval --no-judge # skip judging (counterfactual will report missing)
"""
import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.evaluation import ablation as A
from src.evaluation.generation_metrics import LLMEvaluator
from src.generation.llm_client import LLMClient
from src.generation.verified_rag_pipeline import CitationVerifier
from src.utils.config import load_config
from src.utils.logging_setup import get_logger, setup_logging

logger = get_logger("evaluation.ablation")

PH20 = Path("experiments/phase20")
PH21 = Path("experiments/phase21")


def latest(pattern: str) -> Optional[Path]:
    hits = sorted(p for p in glob.glob(pattern) if "smoke" not in p
                  and (Path(p) / "predictions.jsonl").exists() and (Path(p) / "metrics.json").exists())
    return Path(hits[-1]) if hits else None


def load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def f2(x: Optional[float], nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def ci_str(ci: Dict[str, Optional[float]]) -> str:
    return "n/a" if ci["mean"] is None else f"{ci['mean']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"


def diff_str(d: Dict[str, Any]) -> str:
    if not d["n_paired"]:
        return "n/a (no paired queries)"
    flag = "excludes 0" if d["excludes_zero"] else "includes 0"
    return f"{d['mean_diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] ({flag}; paired n={d['n_paired']})"


def judge_gated(rows: List[Dict[str, Any]], evaluator: LLMEvaluator, chunk_store_gold) -> Dict[str, Dict[str, Any]]:
    """Judge the citation-cleaned raw answer of every query withheld solely by the claim gate."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in A.verifier_gated_queries(rows):
        retrieved_ids = [c["chunk_id"] for c in r["retrieved"]]
        cleaned = CitationVerifier.verify(r["raw_answer"], retrieved_ids)["cleaned_answer"]
        out[r["query_id"]] = {**evaluator.evaluate(r["question"], r["gold_answer"], cleaned),
                              "judged_text": cleaned}
    return out


def qa_section(rows_by_sys: Dict[str, List[Dict[str, Any]]], raw_j: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"systems": {}, "paired": {}, "claim_gate": {}, "citation": {}, "support_vs_quality": {}}
    for name, rows in rows_by_sys.items():
        fact = A.scored_vector(rows, "factuality")
        corr = A.scored_vector(rows, "correctness")
        hit = [1.0 if r.get("retrieval_eval", {}).get("gold_hit") else 0.0 for r in rows
               if "retrieval_eval" in r]
        res["systems"][name] = {
            "n": len(rows),
            "abstention": A.abstention_breakdown(rows),
            "factuality_answered": A.bootstrap_mean_ci(list(fact.values())),
            "correctness_answered": A.bootstrap_mean_ci(list(corr.values())),
            "gold_hit_at_5": (sum(hit) / len(hit)) if hit else None,
        }
    # Prompt/pipeline effect at fixed retriever: Basic RAG -> Verified-RAG (as run, tau=0.5, stripping on)
    for ret in ("dense", "bm25", "hybrid"):
        b, v = f"basic-{ret}", f"verified-{ret}"
        if b in rows_by_sys and v in rows_by_sys:
            res["paired"][f"{ret}: basic -> verified (factuality)"] = A.paired_bootstrap_diff(
                A.scored_vector(rows_by_sys[b], "factuality"), A.scored_vector(rows_by_sys[v], "factuality"))
            res["paired"][f"{ret}: basic -> verified (correctness)"] = A.paired_bootstrap_diff(
                A.scored_vector(rows_by_sys[b], "correctness"), A.scored_vector(rows_by_sys[v], "correctness"))
    # Retriever effect at fixed pipeline (answered-in-both factuality)
    for pipe in ("basic", "verified"):
        for a_, b_ in (("bm25", "hybrid"), ("bm25", "dense"), ("hybrid", "dense")):
            ka, kb = f"{pipe}-{a_}", f"{pipe}-{b_}"
            if ka in rows_by_sys and kb in rows_by_sys:
                res["paired"][f"{pipe}: {a_} -> {b_} (factuality)"] = A.paired_bootstrap_diff(
                    A.scored_vector(rows_by_sys[ka], "factuality"), A.scored_vector(rows_by_sys[kb], "factuality"))
                ha = {r["query_id"]: 1.0 if r["retrieval_eval"]["gold_hit"] else 0.0 for r in rows_by_sys[ka]}
                hb = {r["query_id"]: 1.0 if r["retrieval_eval"]["gold_hit"] else 0.0 for r in rows_by_sys[kb]}
                res["paired"][f"{pipe}: {a_} -> {b_} (gold hit@5)"] = A.paired_bootstrap_diff(ha, hb)
    for name, rows in rows_by_sys.items():
        if not name.startswith("verified-"):
            continue
        res["citation"][name] = A.citation_integrity(rows)
        res["support_vs_quality"][name] = A.support_vs_judged_quality(rows, "factuality")
        j = raw_j.get(name, {})
        res["claim_gate"][name] = {
            "tau_0.0 (no claim gate)": A.claim_gate_counterfactual(rows, 0.0, j),
            "tau_0.5 (frozen policy)": A.claim_gate_counterfactual(rows, 0.5, j),
            "n_gated_judged": len(j),
            "n_gated_total": len(A.verifier_gated_queries(rows)),
        }
    return res


def summ_section() -> Optional[Dict[str, Any]]:
    d = latest(str(PH21 / "summ-val__*"))
    if d is None:
        return None
    rows = load_rows(d)
    from src.summarization.summary_metrics import SCORABLE
    by_sys: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in rows:
        by_sys.setdefault(r["system"], {"acc": {}, "support": {}, "fre": {}})
        if r["gold_disposition"] in SCORABLE:
            by_sys[r["system"]]["acc"][r["case_id"]] = 1.0 if r["pred_disposition"] == r["gold_disposition"] else 0.0
        by_sys[r["system"]]["support"][r["case_id"]] = r["verification"]["support_rate"]
        fre = r["readability"].get("flesch_reading_ease")
        if fre is not None:
            by_sys[r["system"]]["fre"][r["case_id"]] = fre
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    out: Dict[str, Any] = {"run_dir": str(d), "systems": {k: metrics[k] for k in by_sys if k in metrics}, "paired": {}}
    g = by_sys.get("generic_llm")
    if g:
        for label, key in (("input selection only (structured_input_free_prompt - generic)", "structured_input_free_prompt"),
                           ("output format only (lead_input_structured_prompt - generic)", "lead_input_structured_prompt"),
                           ("input + format (structured_llm - generic)", "structured_llm")):
            if key in by_sys:
                out["paired"][label] = {
                    "disposition_accuracy": A.paired_bootstrap_diff(g["acc"], by_sys[key]["acc"]),
                    "claim_support": A.paired_bootstrap_diff(g["support"], by_sys[key]["support"]),
                }
    return out


def render(res: Dict[str, Any], summ: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    L: List[str] = [f"# Phase 21 Ablation Report — {meta['run_id']}", "",
                    f"- Generator/judge/verifier: `{meta['model']}` | tau* frozen at 0.5 | bootstrap: seed 42, 10,000 resamples, 95% percentile CIs",
                    "- All QA ablations use the 50 Phase 9 queries on the 51,231-chunk index. Ablations here are analyses of frozen runs; "
                    "no configuration was selected using these numbers.", "",
                    "## A. Retriever x pipeline grid (QA, n=50)", "",
                    "| System | Gold hit@5 | Abstained | Model refusal | Claim-gate | Correctness (ans) [95% CI] | Factuality (ans) [95% CI] |",
                    "|---|---|---|---|---|---|---|"]
    for name, s in res["systems"].items():
        ab = s["abstention"]
        L.append(f"| {name} | {pct(s['gold_hit_at_5'])} | {ab['abstained']}/{ab['total']} | {ab['model_refusal']} | "
                 f"{ab['claim_verifier_gate']} | {ci_str(s['correctness_answered'])} | {ci_str(s['factuality_answered'])} |")
    L += ["", "## B. Paired differences (mean of second minus first; answered-in-both queries)", "",
          "| Comparison | Diff [95% CI] |", "|---|---|"]
    for k, v in res["paired"].items():
        L.append(f"| {k} | {diff_str(v)} |")
    L += ["", "## C. Citation verifier (invalid-citation rate before vs after, answered queries)", "",
          "| Run | Cited (before) | Invalid (before) | Rate before | Cited (delivered) | Invalid (delivered) | Rate after |",
          "|---|---|---|---|---|---|---|"]
    for k, c in res["citation"].items():
        L.append(f"| {k} | {c['cited_before']} | {c['invalid_before']} | {pct(c['invalid_rate_before'])} | "
                 f"{c['cited_after']} | {c['invalid_after']} | {pct(c['invalid_rate_after'])} |")
    L += ["", "## D. Claim verifier gate (counterfactual re-decision from stored support rates)", "",
          "| Run | tau | Answered | Abstained | Coverage | Scored | Missing judgement | Correctness (ans) | Factuality (ans) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, g in res["claim_gate"].items():
        for lab in ("tau_0.0 (no claim gate)", "tau_0.5 (frozen policy)"):
            p = g[lab]
            L.append(f"| {k} | {lab} | {p['answered']} | {p['abstained']} | {pct(p['coverage'])} | {p['scored_answered']} | "
                     f"{p['missing_judgement']} | {f2(p['correctness_answered'])} | {f2(p['factuality_answered'])} |")
        L.append(f"| {k} | _gated raw answers judged_ | {g['n_gated_judged']}/{g['n_gated_total']} | | | | | | |")
    L += ["", "Spearman(support_rate, judged factuality), answered queries:", ""]
    for k, s in res["support_vs_quality"].items():
        L.append(f"- {k}: rho = {f2(s['spearman'])} (n={s['n']})")
    if summ:
        L += ["", "## E. Structure awareness: input selection vs output format (summarization, ORIGINAL 29 validation cases)", "",
              f"- Source: `{summ['run_dir']}`", "",
              "| System | Disposition acc | Not stated | Claim support | Flesch RE |", "|---|---|---|---|---|"]
        for k, m in summ["systems"].items():
            L.append(f"| {k} | {pct(m.get('disposition_accuracy'))} | {pct(m.get('disposition_not_stated_rate'))} | "
                     f"{pct(m.get('mean_support_rate'))} | {f2(m.get('flesch_reading_ease'), 1)} |")
        L += ["", "| Effect vs generic_llm | Disposition accuracy diff | Claim support diff |", "|---|---|---|"]
        for k, v in summ["paired"].items():
            L.append(f"| {k} | {diff_str(v['disposition_accuracy'])} | {diff_str(v['claim_support'])} |")
    L += ["", "## Caveats", "",
          "- n=50 QA queries and 29 summarization cases: most differences are within sampling noise; formal tests are Phase 22.",
          "- Basic->Verified comparisons change the prompt as well as the verifier layers (prompt effect is not separated from gating).",
          "- Judge = generator = verifier (same Qwen3-8B): shared-bias risk; no human validation yet (Phase 23).",
          "- Disposition accuracy is a regex over summary text vs metadata `disposal_nature`; headed formats may "
          "score differently simply because an explicit Decision/Final-outcome line exists.",
          "- Counterfactual tau=0 rows depend on judged raw answers only for the queries listed as judged above."]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--model", default=None)
    args = p.parse_args()
    load_dotenv()
    setup_logging()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config().get("generation", {})
    model = args.model or cfg["model_name"]

    runs = {
        "generic": latest(str(PH20 / "generic__*")),
        "basic-dense": latest(str(PH20 / "rag-dense-k5__*")),
        "basic-bm25": latest(str(PH20 / "rag-bm25-k5__*")),
        "basic-hybrid": latest(str(PH20 / "rag-hybrid-k5__*")),
        "verified-dense": latest(str(PH21 / "verified-rag-dense-k5__*")),
        "verified-bm25": latest(str(PH20 / "verified-rag-bm25-k5__*")),
        "verified-hybrid": latest(str(PH20 / "verified-rag-hybrid-k5__*")),
    }
    rows_by_sys = {k: load_rows(v) for k, v in runs.items() if v is not None and k != "generic"}
    missing = [k for k, v in runs.items() if v is None]
    logger.info(f"runs found: {[k for k, v in runs.items() if v]} missing: {missing}")

    run_id = f"ablation__{model.replace('/', '-').replace(':', '-')}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = PH21 / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_j: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not args.no_judge:
        evaluator = LLMEvaluator(LLMClient(cfg.get("judge_model", model), temperature=0.0, max_tokens=512))
        with open(out_dir / "raw_judgements.jsonl", "w", encoding="utf-8") as jf:
            for name, rows in rows_by_sys.items():
                if not name.startswith("verified-"):
                    continue
                raw_j[name] = judge_gated(rows, evaluator, None)
                for qid, ev in raw_j[name].items():
                    jf.write(json.dumps({"system": name, "query_id": qid, **ev}, ensure_ascii=False) + "\n")

    res = qa_section(rows_by_sys, raw_j)
    summ = summ_section()
    meta = {"run_id": run_id, "model": model, "missing_runs": missing,
            "source_runs": {k: str(v) for k, v in runs.items() if v}}
    (out_dir / "metrics.json").write_text(json.dumps({"meta": meta, "qa": res, "summarization": summ}, indent=2,
                                                      default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(render(res, summ, meta), encoding="utf-8")
    logger.info(f"Saved: {out_dir}")
    print(render(res, summ, meta))


if __name__ == "__main__":
    main()
