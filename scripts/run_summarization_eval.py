"""
LegalSaarthi — Phase 15 summarization evaluation harness.

Runs the 4 summarization systems on one split under a fixed protocol and
records reference-free metrics (see src/summarization/summary_metrics.py).

The locked test split is refused unless --final is passed (Phase 20 only).

Usage:
  python -m scripts.run_summarization_eval --split val --limit 3     # smoke
  python -m scripts.run_summarization_eval --split val               # full val
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.generation.llm_client import LLMClient
from src.generation.verified_rag_pipeline import ClaimVerifier
from src.summarization import summary_metrics as M
from src.summarization.prompts import PLAIN_HEADINGS, STRUCTURED_HEADINGS, SUMM_PROMPT_VERSION
from src.summarization.summarizers import Summarizer, build_blocks, load_document
from src.utils.config import load_config
from src.utils.logging_setup import get_logger, setup_logging

logger = get_logger("evaluation.summarization_harness")

STRUCTURED_DIR = Path("data/processed/structured")
SPLITS = Path("data/metadata/split_manifest.csv")
PARQUET = Path("data/metadata/parquet")


def model_tag(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def load_disposals() -> dict:
    frames = []
    for f in sorted(PARQUET.glob("year=*/*.parquet")):
        frames.append(pd.read_parquet(f, columns=["case_id", "disposal_nature"]))
    df = pd.concat(frames).drop_duplicates("case_id")
    return dict(zip(df["case_id"], df["disposal_nature"]))


def find_doc(case_id: str):
    hits = list(STRUCTURED_DIR.glob(f"year=*/{case_id}.json"))
    return hits[0] if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "val", "test"], default="val")
    p.add_argument("--final", action="store_true", help="Required to touch the locked test split")
    p.add_argument("--systems", default=",".join(Summarizer.SYSTEMS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--input-budget", type=int, default=6000)
    p.add_argument("--model", default=None)
    p.add_argument("--restrict-manifest", default=None,
                   help="Only evaluate cases that this manifest assigns to --split (e.g. split_manifest_prototype.csv)")
    p.add_argument("--phase-dir", default="phase15")
    p.add_argument("--phase-num", type=int, default=15)
    args = p.parse_args()

    if args.split == "test" and not args.final:
        sys.exit("Refusing to run on the LOCKED test split without --final (Phase 20 only).")

    load_dotenv()
    setup_logging()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config().get("generation", {})
    gen_model = args.model or cfg["model_name"]
    systems = [s.strip() for s in args.systems.split(",")]

    splits = pd.read_csv(SPLITS)
    case_ids = sorted(splits.loc[splits["split"] == args.split, "case_id"].tolist())
    if args.restrict_manifest:
        rm = pd.read_csv(args.restrict_manifest)
        allowed = set(rm.loc[rm["split"] == args.split, "case_id"])
        case_ids = [c for c in case_ids if c in allowed]
    if args.limit:
        case_ids = case_ids[: args.limit]
    disposals = load_disposals()

    run_id = f"summ-{args.split}__{model_tag(gen_model)}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.limit:
        run_id += f"__smoke{args.limit}"
    out_dir = Path("experiments") / args.phase_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = LLMClient(gen_model, temperature=0.0, max_tokens=600)
    verifier = ClaimVerifier(LLMClient(gen_model, temperature=0.0, max_tokens=256))
    summarizer = Summarizer(gen, input_budget=args.input_budget)

    config = {
        "run_id": run_id, "phase": args.phase_num, "split": args.split, "n_cases": len(case_ids),
        "restrict_manifest": args.restrict_manifest,
        "systems": systems, "generator": gen_model, "verifier": gen_model,
        "temperature": 0.0, "max_tokens": 600, "input_budget_chars": args.input_budget,
        "extractive_chars": summarizer.extractive_chars, "prompt_version": SUMM_PROMPT_VERSION,
        "reference_summaries": "none (reference-free evaluation; ROUGE/BERTScore not applicable)",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info(f"RUN {run_id} | {len(case_ids)} cases x {len(systems)} systems")

    rows, skipped = [], []
    pred_f = open(out_dir / "predictions.jsonl", "w", encoding="utf-8")
    for case_id in tqdm(case_ids, desc="cases"):
        path = find_doc(case_id)
        if path is None:
            skipped.append({"case_id": case_id, "reason": "structured doc not found"})
            continue
        doc = load_document(path)
        blocks = build_blocks(doc["paragraphs"])
        if not blocks:
            skipped.append({"case_id": case_id, "reason": "no non-header blocks"})
            continue
        doc_words = sum(len(b["text"].split()) for b in blocks)
        gold_raw = disposals.get(case_id)
        for system in systems:
            res = summarizer.summarize(system, blocks)
            summary = res["summary"]
            headings = None
            if system in ("structured_llm", "lead_input_structured_prompt"):
                headings = STRUCTURED_HEADINGS
            elif system == "plain_llm":
                headings = PLAIN_HEADINGS
            clean = M.strip_citations(summary)
            if headings:
                clean = M.remove_headings(clean, headings)
            claims = [] if res["generation_failed"] else ClaimVerifier.extract_claims(clean)
            claims = [c for c in claims if "not stated in the provided text" not in c.lower()]
            evidence = M.evidence_for_claims(claims, blocks)
            ver = verifier.verify_claims(claims, evidence)
            row = {
                "case_id": case_id, "split": args.split, "system": system,
                "summary": summary, "generation_failed": res["generation_failed"],
                "model_used": res["model_used"], "latency_s": res["latency_s"],
                "input_blocks": [{"block_id": b["block_id"], "section": b["section"], "pages": b["pages"],
                                  "para_ids": b["para_ids"]} for b in res["input_blocks"]],
                "gold_disposal_raw": gold_raw,
                "gold_disposition": M.normalize_gold_disposition(gold_raw),
                "pred_disposition": M.predict_disposition(clean),
                "verification": ver,
                "readability": M.readability(clean),
                "n_words": len(clean.split()),
                "compression": len(clean.split()) / doc_words if doc_words else None,
                "citations": M.citation_check(summary, [b["block_id"] for b in res["input_blocks"]])
                if headings else None,
                "headings": M.headings_filled(summary, headings) if headings else None,
            }
            rows.append(row)
            pred_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_f.flush()
    pred_f.close()

    metrics = {s: M.aggregate_summaries([r for r in rows if r["system"] == s]) for s in systems}
    metrics["_skipped"] = skipped
    metrics["_models_used"] = sorted({r["model_used"] for r in rows if r["model_used"]})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(config, metrics, systems), encoding="utf-8")
    logger.info(f"Saved: {out_dir}")
    print(json.dumps({s: metrics[s] for s in systems}, indent=2))


def _f(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v*100:.1f}%" if pct else f"{v:.2f}"


def render_report(cfg, metrics, systems):
    rows = [
        ("Generation failures", "generation_failures", None),
        ("Disposition accuracy (vs metadata)", "disposition_accuracy", True),
        ("Disposition not stated", "disposition_not_stated_rate", True),
        ("Mean claim support rate", "mean_support_rate", True),
        ("Supported / unsupported / unknown claims", None, None),
        ("Invalid citation rate", "invalid_citation_rate", True),
        ("Headings filled", "headings_filled_rate", True),
        ("Flesch Reading Ease (higher=easier)", "flesch_reading_ease", False),
        ("Flesch-Kincaid Grade", "fk_grade", False),
        ("Mean words", "mean_words", False),
        ("Median latency (s)", "latency_median_s", False),
    ]
    lines = [f"# Phase 15 Summarization Eval — {cfg['run_id']}", "",
             f"- Split: `{cfg['split']}` | Cases: {cfg['n_cases']} | Generator/verifier: `{cfg['generator']}`",
             f"- Prompt version: `{cfg['prompt_version']}` | Input budget: {cfg['input_budget_chars']} chars", "",
             "| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for label, key, pct in rows:
        if key is None:
            vals = [f"{metrics[s]['total_supported_claims']}/{metrics[s]['total_unsupported_claims']}/"
                    f"{metrics[s]['total_unknown_claims']}" for s in systems]
        elif pct is None:
            vals = [str(metrics[s].get(key)) for s in systems]
        else:
            vals = [_f(metrics[s].get(key), pct) for s in systems]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", f"Skipped cases: {len(metrics['_skipped'])} {metrics['_skipped']}", "", "## Caveats",
              "- No reference summaries: ROUGE/BERTScore not computed.",
              "- Claim support uses the same LLM as the generator (unvalidated verifier); extractive is supported by construction.",
              "- Disposition is matched by regex against metadata `disposal_nature`.",
              "- generic_llm vs structured_llm differ in both input selection and output format (confound).",
              "- Structure labels come from the rule-based Phase 7 detector (FACTS/ISSUES are sparse)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
