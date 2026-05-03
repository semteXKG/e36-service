#!/usr/bin/env python3
"""
benchmark.py -- Compare LLM providers on speed and output quality.

Usage:
    python benchmark.py
    python benchmark.py --runs 3
"""
import argparse
import json
import os
import pathlib
import time

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent / ".env")
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("ERROR: pip install openai")

from query import tokenize, compute_tag_score, compute_phrase_bonus, TAG_BOOST, deduplicate_by_chapter, load_tags, clean_text
from rank_bm25 import BM25Okapi

# ── Providers to compare ──────────────────────────────────────────────────────

PROVIDERS = [
    {
        "name":     "Groq / Llama-3.3-70B",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key":  os.environ.get("GROQ_API_KEY") or (os.environ.get("LLM_API_KEY") if "groq" in os.environ.get("LLM_BASE_URL","") else None),
        "model":    "llama-3.3-70b-versatile",
    },
    {
        "name":     "Kimi K2 / moonshot.ai",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key":  os.environ.get("MOONSHOT_API_KEY") or (os.environ.get("LLM_API_KEY") if "moonshot" in os.environ.get("LLM_BASE_URL","") else None),
        "model":    "kimi-k2.6",
    },
]

# ── Test queries ───────────────────────────────────────────────────────────────

QUERIES = [
    "what brake fluid should I use?",
    "cylinder head bolt torque specs",
    "how to bleed the cooling system",
]

# ── Build search index once ───────────────────────────────────────────────────

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
INDEX_PATH = BASE_DIR / "index.json"

print("Loading index...", end=" ", flush=True)
with open(INDEX_PATH, encoding="utf-8") as f:
    _records = json.load(f)["records"]
_tags   = load_tags()
_corpus = [tokenize(r["text"]) for r in _records]
_bm25   = BM25Okapi(_corpus)
print(f"done ({len(_records)} pages)")


def get_top5(q):
    qt = tokenize(q)
    scores = _bm25.get_scores(qt)
    final = [
        float(scores[i])
        + TAG_BOOST * compute_tag_score(_tags.get(str(r["page"]), []), qt)
        + compute_phrase_bonus(r["text"], qt)
        for i, r in enumerate(_records)
    ]
    ranked = sorted(range(len(final)), key=lambda i: final[i], reverse=True)
    ranked = [i for i in ranked if final[i] > 0][:15]
    pre    = [{**_records[i], "score": final[i]} for i in ranked]
    return deduplicate_by_chapter(pre, _tags)[:5]


def build_prompt(q, top5):
    parts   = ["[Page " + str(r["page"]) + "]\n" + clean_text(r["text"]) for r in top5]
    context = "\n\n---\n\n".join(parts)
    return (
        "You are a BMW E36 service manual assistant. "
        "Answer the question using only the provided manual excerpts. "
        "Format your response in Markdown: use **bold** for critical values (torque specs, fluid types, part numbers, temperatures), "
        "and numbered lists for procedures. "
        "Cite page numbers in your answer (e.g. Page 73).\n\n"
        + context + "\n\nQuestion: " + q
    )


def run_once(provider, prompt):
    if not provider["api_key"]:
        return None
    client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
    t_start = time.perf_counter()
    t_first = None
    tokens  = 0
    text    = ""
    try:
        stream = client.chat.completions.create(
            model=provider["model"],
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if t_first is None:
                    t_first = time.perf_counter()
                text   += chunk.choices[0].delta.content
                tokens += 1
    except Exception as e:
        return {"error": str(e)}

    t_end = time.perf_counter()
    return {
        "ttft":     round((t_first - t_start) if t_first else 0, 2),
        "total":    round(t_end - t_start, 2),
        "tokens":   tokens,
        "tok_s":    round(tokens / (t_end - (t_first or t_start)), 1) if tokens else 0,
        "answer":   text.strip(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2, help="Runs per provider per query (default 2)")
    args = ap.parse_args()

    active = [p for p in PROVIDERS if p["api_key"]]
    if not active:
        raise SystemExit("No API keys found. Set LLM_API_KEY / GROQ_API_KEY / MOONSHOT_API_KEY in .env")

    print(f"\nProviders : {', '.join(p['name'] for p in active)}")
    print(f"Queries   : {len(QUERIES)}")
    print(f"Runs each : {args.runs}\n")

    sep  = "-" * 72
    results = {p["name"]: [] for p in active}

    for q in QUERIES:
        top5   = get_top5(q)
        prompt = build_prompt(q, top5)
        print(sep)
        print(f"Query: \"{q}\"  (context: pages {[r['page'] for r in top5]})")
        print(sep)

        for p in active:
            run_times = []
            for run in range(args.runs):
                r = run_once(p, prompt)
                if r is None:
                    continue
                if "error" in r:
                    print(f"  {p['name']}: ERROR — {r['error']}")
                    break
                run_times.append(r)
                label = f"  run {run+1}"
                print(f"  [{p['name']}] {label}:  TTFT {r['ttft']}s  |  total {r['total']}s  |  {r['tokens']} tokens  |  {r['tok_s']} tok/s")

            if run_times:
                avg_ttft  = round(sum(r["ttft"]  for r in run_times) / len(run_times), 2)
                avg_total = round(sum(r["total"] for r in run_times) / len(run_times), 2)
                avg_toks  = round(sum(r["tok_s"] for r in run_times) / len(run_times), 1)
                results[p["name"]].append({"ttft": avg_ttft, "total": avg_total, "tok_s": avg_toks})
                print(f"  [{p['name']}] avg:    TTFT {avg_ttft}s  |  total {avg_total}s  |  {avg_toks} tok/s")
                print(f"\n  Answer: {run_times[-1]['answer'][:300]}{'...' if len(run_times[-1]['answer']) > 300 else ''}\n")

    # Summary table
    print("=" * 72)
    print("SUMMARY (averages across all queries)")
    print("=" * 72)
    print(f"{'Provider':<30} {'TTFT':>8} {'Total':>8} {'Tok/s':>8}")
    print("-" * 56)
    for p in active:
        rs = results[p["name"]]
        if not rs:
            continue
        print(f"{p['name']:<30} {round(sum(r['ttft'] for r in rs)/len(rs),2):>7}s {round(sum(r['total'] for r in rs)/len(rs),2):>7}s {round(sum(r['tok_s'] for r in rs)/len(rs),1):>7}")
    print("=" * 72)


if __name__ == "__main__":
    main()
