import json
import glob
import os

runs = [
    ("E1 Generic LLM", "generic__ollama-qwen3-8b__20261024T090117Z"),
    ("E2 Basic RAG Dense", "rag-dense-k5__ollama-qwen3-8b__20261024T090501Z"),
    ("E3 Basic RAG BM25", "rag-bm25-k5__ollama-qwen3-8b__20261024T091159Z"),
    ("E4 Basic RAG Hybrid", "rag-hybrid-k5__ollama-qwen3-8b__20261024T092452Z"),
    ("E5 Verified-RAG BM25", "verified-rag-bm25-k5__ollama-qwen3-8b__20261024T093708Z"),
    ("E6 Verified-RAG Hybrid", "verified-rag-hybrid-k5__ollama-qwen3-8b__20261024T095500Z"),
]

print("=" * 105)
print(f"{'System & Configuration':<24} | {'Split':<6} | {'n':<3} | {'Abst%':<6} | {'Corr(ans)':<9} | {'Fact(ans)':<9} | {'Hit@5':<6} | {'Recall@5':<8} | {'ClaimSupp':<9} | {'InvCit%':<7}")
print("=" * 105)

for label, folder in runs:
    mpath = os.path.join("experiments/phase20", folder, "metrics.json")
    with open(mpath, encoding="utf-8") as f:
        data = json.load(f)
    
    # Overall
    o = data["overall"]
    abst = f"{o.get('abstention_rate', 0)*100:.1f}%"
    corr = f"{o.get('correctness_answered', 0):.2f}" if o.get("correctness_answered") is not None else "N/A"
    fact = f"{o.get('factuality_answered', 0):.2f}" if o.get("factuality_answered") is not None else "N/A"
    hit = f"{o.get('gold_hit_rate', 0)*100:.1f}%" if "gold_hit_rate" in o else "N/A"
    rec = f"{o.get('gold_recall', 0)*100:.1f}%" if "gold_recall" in o else "N/A"
    supp = f"{o.get('mean_support_rate', 0)*100:.1f}%" if "mean_support_rate" in o else "N/A"
    inv = f"{o.get('invalid_citation_rate', 0)*100:.1f}%" if "invalid_citation_rate" in o else "N/A"
    print(f"{label:<24} | {'ALL':<6} | {o.get('count', 0):<3} | {abst:<6} | {corr:<9} | {fact:<9} | {hit:<6} | {rec:<8} | {supp:<9} | {inv:<7}")

    # Test split only
    if "by_split" in data and "test" in data["by_split"]:
        t = data["by_split"]["test"]
        t_abst = f"{t.get('abstention_rate', 0)*100:.1f}%"
        t_corr = f"{t.get('correctness_answered', 0):.2f}" if t.get("correctness_answered") is not None else "N/A"
        t_fact = f"{t.get('factuality_answered', 0):.2f}" if t.get("factuality_answered") is not None else "N/A"
        t_hit = f"{t.get('gold_hit_rate', 0)*100:.1f}%" if "gold_hit_rate" in t else "N/A"
        t_rec = f"{t.get('gold_recall', 0)*100:.1f}%" if "gold_recall" in t else "N/A"
        t_supp = f"{t.get('mean_support_rate', 0)*100:.1f}%" if "mean_support_rate" in t else "N/A"
        t_inv = f"{t.get('invalid_citation_rate', 0)*100:.1f}%" if "invalid_citation_rate" in t else "N/A"
        print(f"{'  -> Locked Test Split':<24} | {'TEST':<6} | {t.get('count', 0):<3} | {t_abst:<6} | {t_corr:<9} | {t_fact:<9} | {t_hit:<6} | {t_rec:<8} | {t_supp:<9} | {t_inv:<7}")

print("=" * 105)

# Summarization
print("\n" + "=" * 115)
print(f"{'Summarization System':<24} | {'Test n':<6} | {'DisposAcc':<9} | {'NotStated':<9} | {'ClaimSupp':<9} | {'FleschEase':<10} | {'GradeLevel':<10} | {'MeanWords':<9} | {'InvCit%':<7}")
print("=" * 115)

# E7 (generic_llm, structured_llm)
e7_path = "experiments/phase20/summ-test__ollama-qwen3-8b__20261024T101022Z/metrics.json"
with open(e7_path, encoding="utf-8") as f:
    e7_data = json.load(f)

for sname, label in [("generic_llm", "E7 Generic Summarizer"), ("structured_llm", "E7 Structured Summarizer")]:
    sd = e7_data[sname]
    d_acc = f"{sd.get('disposition_accuracy', 0)*100:.1f}%"
    d_ns = f"{sd.get('disposition_not_stated_rate', 0)*100:.1f}%"
    supp = f"{sd.get('mean_support_rate', 0)*100:.1f}%"
    fl = f"{sd.get('flesch_reading_ease', 0):.1f}"
    gr = f"{sd.get('fk_grade', 0):.1f}"
    w = f"{sd.get('mean_words', 0):.1f}"
    inv = f"{sd.get('invalid_citation_rate', 0)*100:.1f}%" if "invalid_citation_rate" in sd else "N/A"
    print(f"{label:<24} | {sd.get('count', 0):<6} | {d_acc:<9} | {d_ns:<9} | {supp:<9} | {fl:<10} | {gr:<10} | {w:<9} | {inv:<7}")

# E8 (plain_llm)
e8_path = "experiments/phase20/summ-test__ollama-qwen3-8b__20261024T103554Z/metrics.json"
with open(e8_path, encoding="utf-8") as f:
    e8_data = json.load(f)

sd = e8_data["plain_llm"]
d_acc = f"{sd.get('disposition_accuracy', 0)*100:.1f}%"
d_ns = f"{sd.get('disposition_not_stated_rate', 0)*100:.1f}%"
supp = f"{sd.get('mean_support_rate', 0)*100:.1f}%"
fl = f"{sd.get('flesch_reading_ease', 0):.1f}"
gr = f"{sd.get('fk_grade', 0):.1f}"
w = f"{sd.get('mean_words', 0):.1f}"
inv = f"{sd.get('invalid_citation_rate', 0)*100:.1f}%" if "invalid_citation_rate" in sd else "N/A"
print(f"{'E8 Plain-Language Summ':<24} | {sd.get('count', 0):<6} | {d_acc:<9} | {d_ns:<9} | {supp:<9} | {fl:<10} | {gr:<10} | {w:<9} | {inv:<7}")
print("=" * 115)
